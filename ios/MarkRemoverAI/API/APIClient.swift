import Foundation

enum APIError: LocalizedError {
    case http(Int, String)
    case transport(Error)
    case decoding
    case needsVerification(String)
    case notAuthenticated
    case outOfCredits
    case workerOffline

    var errorDescription: String? {
        switch self {
        case .http(_, let message): return message
        case .transport: return "Can't reach ObjectRemoverAI. Check your connection."
        case .decoding: return "The server sent something unexpected."
        case .needsVerification(let email): return "Verify \(email) first. Check your inbox."
        case .notAuthenticated: return "Sign in to keep going."
        case .outOfCredits: return "You've used today's free videos. Two more tomorrow."
        case .workerOffline: return "The GPU worker isn't answering. Try again in a moment."
        }
    }
}

actor APIClient {
    static let shared = APIClient()

    static let baseURLDefaultsKey = "api_base_url"

    /// The app talks to Railway directly rather than through the marketing
    /// domain. Public and school Wi-Fi filters routinely refuse to resolve
    /// `markremoverai.com` - it is a young, uncategorised domain - while the
    /// Railway host answers fine. The website being unreachable is not a
    /// reason for the app to stop working, so the apex is only a fallback.
    static let railwayBase = "https://user-interface-ui-production.up.railway.app"
    static let productionBase = "https://markremoverai.com"

    /// Probed in order at launch; the first that answers is remembered.
    static var candidateHosts: [String] { [railwayBase, productionBase] }

    private let session: URLSession

    private init() {
        let config = URLSessionConfiguration.default
        // The backend authenticates with a Flask session cookie, so the cookie
        // jar is the whole login state. It persists across launches for free.
        config.httpCookieAcceptPolicy = .always
        config.httpShouldSetCookies = true
        config.httpCookieStorage = .shared
        // Tags every call as coming from the app. The backend hands out the
        // daily free allowance on this header alone, so the website - which
        // has its own pricing - is left untouched.
        config.httpAdditionalHeaders = ["X-Client": "objectremoverai-ios"]
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 600
        // Deliberately false: with it on, an unresolvable host reports
        // "waiting" forever instead of failing, which strands the launch check.
        config.waitsForConnectivity = false
        session = URLSession(configuration: config)
    }

    nonisolated var baseURL: URL {
        let stored = UserDefaults.standard.string(forKey: Self.baseURLDefaultsKey)
        return URL(string: stored ?? Self.railwayBase)!
    }

    /// Some networks (school and library DNS filters) refuse to resolve the
    /// apex domain while the Railway host answers fine. Let the user flip.
    nonisolated static func setBaseURL(_ value: String) {
        UserDefaults.standard.set(value, forKey: baseURLDefaultsKey)
    }

    nonisolated static var currentBaseURL: String {
        UserDefaults.standard.string(forKey: baseURLDefaultsKey) ?? railwayBase
    }

    /// A `-api_base_url` launch argument only lives for that one launch, which
    /// makes it look like the setting "forgot" as soon as the app is opened
    /// from the home screen. Persist it so it survives.
    nonisolated static func bootstrap() {
        if let launchValue = UserDefaults.standard.string(forKey: baseURLDefaultsKey) {
            UserDefaults.standard.set(launchValue, forKey: baseURLDefaultsKey)
            AppLog.info(.app, "API host: \(launchValue)")
        } else {
            AppLog.info(.app, "API host: \(railwayBase) (default)")
        }
    }

    // MARK: - Core

    func request(
        _ path: String,
        method: String = "GET",
        json: [String: Any]? = nil
    ) async throws -> Data {
        var req = URLRequest(url: baseURL.appendingPathComponent(path))
        req.httpMethod = method
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        if let json {
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: json)
        }

        let started = Date()
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: req)
        } catch {
            AppLog.error(.net, "\(method) \(path) failed: \(error.localizedDescription)")

            if Self.isHostUnreachable(error), await switchToFallbackHost() {
                AppLog.info(.net, "Retrying \(path) on \(Self.currentBaseURL)")
                return try await request(path, method: method, json: json)
            }
            throw APIError.transport(error)
        }

        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        let ms = Int(Date().timeIntervalSince(started) * 1000)

        guard (200..<300).contains(code) else {
            let failure = Self.mapFailure(code: code, data: data)
            AppLog.error(.net, "\(method) \(path) → \(code) (\(ms)ms): \(failure.localizedDescription)")
            throw failure
        }

        AppLog.debug(.net, "\(method) \(path) → \(code) (\(ms)ms)")
        return data
    }

    /// DNS refusing the apex domain is the failure mode on filtered networks,
    /// and it is indistinguishable from the site being down to the user. Swap
    /// to the Railway host once and remember it.
    nonisolated static func isHostUnreachable(_ error: Error) -> Bool {
        let code = (error as NSError).code
        return (error as NSError).domain == NSURLErrorDomain
            && (code == NSURLErrorCannotFindHost
                || code == NSURLErrorDNSLookupFailed
                || code == NSURLErrorCannotConnectToHost)
    }

    private func switchToFallbackHost() async -> Bool {
        let current = Self.currentBaseURL
        guard let winner = await Self.firstReachableHost(excluding: current) else {
            AppLog.error(.net, "No API host is reachable from this network")
            return false
        }
        AppLog.info(.net, "\(current) is unreachable - switching to \(winner)")
        Self.setBaseURL(winner)
        return true
    }

    /// Probes the known hosts at once and returns whichever answers
    /// `/api/health` first. Racing rather than trying them in sequence means a
    /// blackholed DNS lookup on one host cannot hold up a host that works.
    static func firstReachableHost(excluding: String? = nil) async -> String? {
        let hosts = candidateHosts.filter { $0 != excluding }
        guard !hosts.isEmpty else { return nil }

        return await withTaskGroup(of: String?.self) { group in
            for host in hosts {
                group.addTask { await probe(host) ? host : nil }
            }
            for await result in group {
                if let result {
                    group.cancelAll()
                    return result
                }
            }
            return nil
        }
    }

    private static func probe(_ host: String) async -> Bool {
        guard let url = URL(string: host)?.appendingPathComponent("api/health") else { return false }

        var req = URLRequest(url: url)
        req.timeoutInterval = 8
        req.setValue("application/json", forHTTPHeaderField: "Accept")

        do {
            let (_, response) = try await URLSession.shared.data(for: req)
            let ok = ((response as? HTTPURLResponse)?.statusCode ?? 0) == 200
            AppLog.debug(.net, "Probe \(host): \(ok ? "reachable" : "no")")
            return ok
        } catch {
            AppLog.debug(.net, "Probe \(host): \(error.localizedDescription)")
            return false
        }
    }

    /// Called once at launch so the app settles on a host that works before the
    /// user touches anything.
    static func resolveHost() async {
        if let stored = UserDefaults.standard.string(forKey: baseURLDefaultsKey), await probe(stored) {
            AppLog.info(.net, "Using \(stored)")
            return
        }
        if let winner = await firstReachableHost() {
            setBaseURL(winner)
            AppLog.info(.net, "Resolved API host: \(winner)")
        }
    }

    private static func mapFailure(code: Int, data: Data) -> APIError {
        let decoded = try? JSONDecoder().decode(APIErrorResponse.self, from: data)
        let message = decoded?.text ?? "Request failed (\(code))."

        switch code {
        case 401: return .notAuthenticated
        case 402: return .outOfCredits
        case 403 where decoded?.needsVerification == true:
            return .needsVerification(message)
        case 504: return .workerOffline
        default: return .http(code, message)
        }
    }

    func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        do { return try JSONDecoder().decode(T.self, from: data) }
        catch { throw APIError.decoding }
    }

    // MARK: - Health

    func health() async throws -> HealthResponse {
        try decode(HealthResponse.self, from: try await request("api/health"))
    }

    // MARK: - Auth

    func authStatus() async throws -> AuthStatusResponse {
        try decode(AuthStatusResponse.self, from: try await request("api/auth/status"))
    }

    func login(email: String, password: String) async throws -> User {
        let data = try await request(
            "api/auth/login",
            method: "POST",
            json: ["email": email, "password": password]
        )
        return try decode(AuthSuccessResponse.self, from: data).user
    }

    func register(email: String, password: String, name: String) async throws {
        _ = try await request(
            "api/auth/register",
            method: "POST",
            json: ["email": email, "password": password, "name": name]
        )
    }

    /// Asks the server to mail a reset link. It answers the same way whether or
    /// not the address is on file, so the caller must not treat success as
    /// proof the account exists.
    func requestPasswordReset(email: String) async throws {
        _ = try await request(
            "api/auth/forgot-password",
            method: "POST",
            json: ["email": email]
        )
    }

    /// Trades the one-time code from the native Google flow for a real session
    /// cookie on this URLSession.
    func exchangeGoogleCode(_ code: String) async throws -> User {
        let data = try await request(
            "api/auth/exchange",
            method: "POST",
            json: ["code": code]
        )
        return try decode(AuthSuccessResponse.self, from: data).user
    }

    /// Apple only reveals the name and address on the first authorization, so
    /// they are forwarded alongside the token rather than looked up later.
    func signInWithApple(identityToken: String, email: String?, name: String?) async throws -> User {
        var payload: [String: Any] = ["identity_token": identityToken]
        if let email, !email.isEmpty { payload["email"] = email }
        if let name, !name.isEmpty { payload["name"] = name }

        let data = try await request("api/auth/apple", method: "POST", json: payload)
        return try decode(AuthSuccessResponse.self, from: data).user
    }

    /// Apple requires an in-app way to delete an account wherever one can be
    /// created (guideline 5.1.1(v)). The server removes the user record along
    /// with their uploads and results.
    func deleteAccount() async throws {
        _ = try await request("api/auth/delete-account", method: "POST")
        HTTPCookieStorage.shared.cookies(for: baseURL)?.forEach {
            HTTPCookieStorage.shared.deleteCookie($0)
        }
    }

    func logout() async {
        _ = try? await request("api/auth/logout", method: "POST")
        // Clear the jar too, so a stale cookie can't resurrect the session.
        HTTPCookieStorage.shared.cookies(for: baseURL)?.forEach {
            HTTPCookieStorage.shared.deleteCookie($0)
        }
    }

    // MARK: - Upload

    /// Mirrors the web flow: ask the API for a B2 ticket, PUT the bytes straight
    /// to B2 (Railway never sees them), then tell the API it landed.
    func upload(fileURL: URL, filename: String, contentType: String) async throws -> String {
        let ticket: UploadURLResponse = try decode(
            UploadURLResponse.self,
            from: try await request(
                "api/get-upload-url",
                method: "POST",
                json: ["filename": filename, "content_type": contentType]
            )
        )

        try await uploadToB2(fileURL: fileURL, ticket: ticket, contentType: contentType)

        let done: UploadCompleteResponse = try decode(
            UploadCompleteResponse.self,
            from: try await request(
                "api/upload-complete",
                method: "POST",
                json: [
                    "task_id": ticket.taskId,
                    "remote_path": ticket.remotePath,
                    "cdn_url": ticket.cdnURL,
                    "filename": filename
                ]
            )
        )
        return done.taskId
    }

    private func uploadToB2(fileURL: URL, ticket: UploadURLResponse, contentType: String) async throws {
        try await uploadToB2(
            fileURL: fileURL,
            uploadURL: ticket.uploadURL,
            authToken: ticket.authToken,
            remotePath: ticket.remotePath,
            contentType: contentType
        )
    }

    func uploadToB2(
        fileURL: URL,
        uploadURL: String,
        authToken: String,
        remotePath: String,
        contentType: String
    ) async throws {
        guard let url = URL(string: uploadURL) else { throw APIError.decoding }

        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue(authToken, forHTTPHeaderField: "Authorization")
        req.setValue(contentType, forHTTPHeaderField: "Content-Type")
        // B2 wants the path percent-encoded, and the backend opts out of the
        // checksum the same way the web client does.
        let encodedPath = remotePath
            .addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? remotePath
        req.setValue(encodedPath, forHTTPHeaderField: "X-Bz-File-Name")
        req.setValue("do_not_verify", forHTTPHeaderField: "X-Bz-Content-Sha1")

        let (data, response): (Data, URLResponse)
        do {
            // Streaming from the file keeps a 4K video off the heap.
            (data, response) = try await session.upload(for: req, fromFile: fileURL)
        } catch {
            throw APIError.transport(error)
        }

        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(code) else {
            let body = String(data: data, encoding: .utf8) ?? ""
            throw APIError.http(code, "Upload to storage failed (\(code)). \(body.prefix(120))")
        }
    }

    // MARK: - Selection & processing

    /// Round-trips one frame to the SAM2 worker to preview what a set of taps
    /// selects. Returns the base64 PNG mask, or nil when the worker is quiet.
    func previewMask(
        frameBase64PNG: String,
        frameIndex: Int,
        points: [SelectionPoint],
        videoWidth: Int,
        videoHeight: Int
    ) async throws -> String? {
        let payload: [String: Any] = [
            "frame_data": frameBase64PNG,
            "frame_index": frameIndex,
            "points": points.map(\.payload),
            "video_width": videoWidth,
            "video_height": videoHeight
        ]
        let data = try await request("api/sam2/select-object", method: "POST", json: payload)
        return try decode(SelectObjectResponse.self, from: data).mask
    }

    func processVideo(
        taskId: String,
        points: [SelectionPoint],
        videoWidth: Int,
        videoHeight: Int,
        frameIndex: Int
    ) async throws -> String {
        let payload: [String: Any] = [
            "task_id": taskId,
            "prompt_mode": "point",
            "points": points.map(\.payload),
            "video_width": videoWidth,
            "video_height": videoHeight,
            "frame_index": frameIndex
        ]
        let data = try await request("api/sam2/process-video", method: "POST", json: payload)
        let result = try decode(ProcessVideoResponse.self, from: data)
        guard let jobId = result.jobId else {
            throw APIError.http(500, result.message ?? "The server didn't return a job id.")
        }
        return jobId
    }

    /// Static path: the drawn area is repeated on every frame
    /// server-side, so SAM2 never runs.
    func processStaticMask(
        taskId: String,
        maskBase64PNG: String,
        videoWidth: Int,
        videoHeight: Int,
        frameCount: Int
    ) async throws -> String {
        let payload: [String: Any] = [
            "task_id": taskId,
            "mask_base64": maskBase64PNG,
            "video_width": videoWidth,
            "video_height": videoHeight,
            "frame_count": frameCount
        ]
        let data = try await request("api/process-static-mask", method: "POST", json: payload)
        let result = try decode(ProcessVideoResponse.self, from: data)
        guard let jobId = result.jobId else {
            throw APIError.http(500, result.message ?? "The server didn't return a job id.")
        }
        return jobId
    }

    func jobStatus(jobId: String) async throws -> JobStatusResponse {
        try decode(JobStatusResponse.self, from: try await request("api/sam2/status/\(jobId)"))
    }

    /// Result paths come back either absolute or relative to the API host.
    nonisolated func absoluteResultURL(_ raw: String) -> URL? {
        if raw.hasPrefix("http") { return URL(string: raw) }
        return URL(string: raw, relativeTo: baseURL)?.absoluteURL
    }

    func download(_ url: URL) async throws -> URL {
        do {
            let (temp, _) = try await session.download(from: url)
            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent("markremover-\(UUID().uuidString).mp4")
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.moveItem(at: temp, to: dest)
            return dest
        } catch {
            throw APIError.transport(error)
        }
    }
}
