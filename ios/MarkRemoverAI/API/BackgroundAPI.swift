import SwiftUI

/// Background replacement runs on its own job pipeline (`objrem/…`), separate
/// from the object-removal one — different upload, different job id, different
/// status endpoint. Keeping it in its own file stops the two getting confused.
extension APIClient {

    struct BackgroundJob {
        let jobId: String
        let uploadURL: String
        let authToken: String
        let remotePath: String
        let cdnURL: String
    }

    func backgroundUpload(
        fileURL: URL,
        filename: String,
        contentType: String,
        width: Int,
        height: Int,
        fps: Double,
        duration: Double
    ) async throws -> String {
        let ticket: BackgroundTicket = try decode(
            BackgroundTicket.self,
            from: try await request(
                "api/object-removal/get-upload-url",
                method: "POST",
                json: ["filename": filename]
            )
        )

        try await uploadToB2(
            fileURL: fileURL,
            uploadURL: ticket.uploadURL,
            authToken: ticket.authToken,
            remotePath: ticket.remotePath,
            contentType: contentType
        )

        _ = try await request(
            "api/object-removal/upload-complete",
            method: "POST",
            json: [
                "job_id": ticket.jobId,
                "cdn_url": ticket.cdnURL,
                "width": width,
                "height": height,
                "fps": fps,
                "duration": duration,
                "frame_count": max(1, Int((duration * fps).rounded()))
            ]
        )
        return ticket.jobId
    }

    /// Marks the subject to keep, then follows it through the clip.
    func backgroundSelect(jobId: String, points: [SelectionPoint], frameIndex: Int) async throws {
        _ = try await request(
            "api/object-removal/select",
            method: "POST",
            json: [
                "job_id": jobId,
                "points": points.map(\.payload),
                "frame_index": frameIndex
            ]
        )
    }

    /// `modifiedMasks` are hand-corrected masks keyed by object id; the SAM2
    /// worker uses them in place of what it generated.
    func backgroundTrack(jobId: String, modifiedMasks: [(objectId: Int, base64: String)] = []) async throws {
        var payload: [String: Any] = ["job_id": jobId]
        if !modifiedMasks.isEmpty {
            payload["modified_masks"] = modifiedMasks.map {
                ["objectId": $0.objectId, "mask": $0.base64]
            }
        }
        _ = try await request("api/object-removal/track", method: "POST", json: payload)
    }

    func backgroundExport(jobId: String, settings: BackgroundSettings) async throws {
        var payload: [String: Any] = [
            "job_id": jobId,
            "operation": settings.operation.rawValue,
            "background": settings.fill.rawValue,
            "dilation": Int(settings.dilation),
            "format": settings.fill.format
        ]
        if settings.fill == .color { payload["bg_color"] = settings.hex }
        if settings.fill == .blur { payload["blur_amount"] = Int(settings.blurAmount) }

        _ = try await request("api/object-removal/export", method: "POST", json: payload)
    }

    func backgroundStatus(jobId: String) async throws -> JobStatusResponse {
        try decode(
            JobStatusResponse.self,
            from: try await request("api/object-removal/status/\(jobId)")
        )
    }

    nonisolated func backgroundDownloadURL(jobId: String) -> URL {
        baseURL.appendingPathComponent("api/object-removal/download/\(jobId)")
    }
}

private struct BackgroundTicket: Codable {
    let status: String
    let jobId: String
    let uploadURL: String
    let authToken: String
    let remotePath: String
    let cdnURL: String

    enum CodingKeys: String, CodingKey {
        case status
        case jobId = "job_id"
        case uploadURL = "upload_url"
        case authToken = "auth_token"
        case remotePath = "remote_path"
        case cdnURL = "cdn_url"
    }
}

/// Whether the thing you marked is what survives, or what disappears.
enum BackgroundOperation: String, CaseIterable, Identifiable {
    case keepObject = "keep_object"
    case removeObject = "remove_object"

    var id: String { rawValue }
    var title: String { self == .keepObject ? "Keep it" : "Remove it" }
    var detail: String {
        self == .keepObject
            ? "What you marked stays; the rest is replaced."
            : "What you marked goes; the rest stays as it is."
    }
}

/// What fills the background once the subject is isolated.
enum BackgroundFill: String, CaseIterable, Identifiable {
    case transparent, color, blur

    var id: String { rawValue }

    var title: String {
        switch self {
        case .transparent: return "Transparent"
        case .color: return "Solid colour"
        case .blur: return "Blurred"
        }
    }

    var symbol: String {
        switch self {
        case .transparent: return "square.dashed"
        case .color: return "paintpalette"
        case .blur: return "drop.circle"
        }
    }

    /// MP4 has no alpha channel, so a transparent export has to be WebM.
    var format: String { self == .transparent ? "webm" : "mp4" }
}

/// Every knob the website exposes, with its real ranges and defaults.
struct BackgroundSettings {
    var operation: BackgroundOperation = .keepObject
    var fill: BackgroundFill = .transparent
    /// Chroma green by default, same as the site.
    var color: Color = Color(red: 0, green: 1, blue: 0)
    /// 5…50 on the site, default 20.
    var blurAmount: Double = 20
    /// 0…20. Grows the mask outward — a couple of pixels hides a halo of the
    /// old background clinging to the edge.
    var dilation: Double = 0

    var hex: String {
        let ui = UIColor(color)
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        ui.getRed(&r, green: &g, blue: &b, alpha: &a)
        return String(format: "#%02X%02X%02X", Int(r * 255), Int(g * 255), Int(b * 255))
    }
}
