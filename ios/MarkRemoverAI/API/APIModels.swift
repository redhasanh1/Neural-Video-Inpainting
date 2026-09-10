import Foundation

// MARK: - Auth

struct User: Codable, Equatable {
    let id: Int
    let email: String
    let name: String?
    let credits: Double
    let emailVerified: Bool

    enum CodingKeys: String, CodingKey {
        case id, email, name, credits
        case emailVerified = "email_verified"
    }

    init(id: Int, email: String, name: String?, credits: Double, emailVerified: Bool) {
        self.id = id
        self.email = email
        self.name = name
        self.credits = credits
        self.emailVerified = emailVerified
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(Int.self, forKey: .id)
        email = try c.decode(String.self, forKey: .email)
        name = try c.decodeIfPresent(String.self, forKey: .name)
        // The server has handed back credits as both Int and Double over its life.
        credits = (try? c.decode(Double.self, forKey: .credits)) ?? 0
        emailVerified = (try? c.decode(Bool.self, forKey: .emailVerified)) ?? false
    }
}

extension User {
    /// Rebuilds the value with a fresh balance — used after a purchase so the
    /// UI updates without refetching the whole profile.
    func withCredits(_ balance: Double) -> User {
        User(id: id, email: email, name: name, credits: balance, emailVerified: emailVerified)
    }
}

struct AuthStatusResponse: Codable {
    let authenticated: Bool
    let user: User?
}

struct AuthSuccessResponse: Codable {
    let status: String
    let user: User
}

/// The server answers a failed login with `error`, and a login blocked on an
/// unverified address with `needs_verification` alongside it.
struct APIErrorResponse: Codable {
    let error: String?
    let message: String?
    let needsVerification: Bool?

    enum CodingKeys: String, CodingKey {
        case error, message
        case needsVerification = "needs_verification"
    }

    var text: String { error ?? message ?? "Something went wrong." }
}

// MARK: - Health

struct HealthResponse: Codable {
    let status: String
    let message: String?
}

// MARK: - Upload

struct UploadURLResponse: Codable {
    let status: String
    let taskId: String
    let uploadURL: String
    let authToken: String
    let remotePath: String
    let cdnURL: String

    enum CodingKeys: String, CodingKey {
        case status
        case taskId = "task_id"
        case uploadURL = "upload_url"
        case authToken = "auth_token"
        case remotePath = "remote_path"
        case cdnURL = "cdn_url"
    }
}

struct UploadCompleteResponse: Codable {
    let status: String
    let taskId: String
    let videoURL: String

    enum CodingKeys: String, CodingKey {
        case status
        case taskId = "task_id"
        case videoURL = "video_url"
    }
}

// MARK: - Selection & processing

/// A tap on the frame. `label` 1 marks the object to erase, 0 marks something
/// to keep — the same convention SAM2 uses on the web.
struct SelectionPoint: Codable, Identifiable, Equatable {
    var id = UUID()
    let x: Int
    let y: Int
    let label: Int

    enum CodingKeys: String, CodingKey { case x, y, label }
}

struct SelectObjectResponse: Codable {
    let status: String
    /// Base64-encoded PNG mask, full video resolution.
    let mask: String?
    let score: Double?
    let message: String?
}

struct ProcessVideoResponse: Codable {
    let status: String
    let jobId: String?
    let message: String?

    enum CodingKeys: String, CodingKey {
        case status
        case jobId = "job_id"
        case message
    }
}

struct JobStatusResponse: Codable {
    let status: String
    let progress: Int?
    let message: String?
    let resultURL: String?
    let error: String?
    let newCredits: Double?

    enum CodingKeys: String, CodingKey {
        case status, progress, message, error
        case resultURL = "result_url"
        case newCredits = "new_credits"
    }
}

// MARK: - Purchases

struct RedeemResponse: Codable {
    let status: String
    let credits: Double
    let creditsAdded: Int?
    let alreadyRedeemed: Bool?

    enum CodingKeys: String, CodingKey {
        case status, credits
        case creditsAdded = "credits_added"
        case alreadyRedeemed = "already_redeemed"
    }
}
