import UIKit

/// Composes a support email with the details that would otherwise take three
/// round trips to establish. It only opens a draft — nothing is sent, and the
/// user sees exactly what's attached before deciding.
enum SupportMail {
    static let address = "support@markremoverai.com"

    static func composeURL(
        email: String?,
        credits: Double,
        host: String,
        recentErrors: [String]
    ) -> URL? {
        let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?"
        let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?"

        var body = """


        ---
        Please leave the details below — they help us find the problem.

        Account: \(email ?? "not signed in")
        Credits: \(Int(credits))
        App: \(version) (\(build))
        Device: \(UIDevice.current.systemName) \(UIDevice.current.systemVersion)
        Server: \(host)
        """

        if !recentErrors.isEmpty {
            body += "\n\nRecent errors:\n" + recentErrors.joined(separator: "\n")
        }

        var components = URLComponents()
        components.scheme = "mailto"
        components.path = address
        components.queryItems = [
            URLQueryItem(name: "subject", value: "ObjectRemoverAI support"),
            URLQueryItem(name: "body", value: body)
        ]
        return components.url
    }
}
