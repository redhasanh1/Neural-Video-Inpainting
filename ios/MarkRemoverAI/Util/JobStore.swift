import Foundation

/// A render survives the app closing — it runs on the GPU either way. Without a
/// record of it, quitting mid-job loses the result entirely: the job finishes,
/// nobody collects it. This keeps that record on disk.
struct TrackedJob: Codable, Identifiable, Equatable {
    enum Kind: String, Codable {
        case removal, background

        var title: String { self == .removal ? "Object removal" : "Background" }
        var symbol: String { self == .removal ? "wand.and.stars" : "person.and.background.dotted" }
    }

    enum State: String, Codable {
        case running, finished, failed
    }

    let id: String
    let kind: Kind
    var state: State = .running
    var submitted: Date = Date()
    var resultURL: String?
    var detail: String?

    var age: String {
        let seconds = Int(Date().timeIntervalSince(submitted))
        if seconds < 60 { return "\(seconds)s ago" }
        if seconds < 3600 { return "\(seconds / 60)m ago" }
        if seconds < 86400 { return "\(seconds / 3600)h ago" }
        return "\(seconds / 86400)d ago"
    }
}

@MainActor
final class JobStore: ObservableObject {
    static let shared = JobStore()

    @Published private(set) var jobs: [TrackedJob] = []

    private let url: URL = {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("jobs.json")
    }()

    private init() { load() }

    var unfinished: [TrackedJob] { jobs.filter { $0.state == .running } }
    var collectable: [TrackedJob] { jobs.filter { $0.state == .finished && $0.resultURL != nil } }

    func record(id: String, kind: TrackedJob.Kind, detail: String? = nil) {
        guard !jobs.contains(where: { $0.id == id }) else { return }
        jobs.insert(TrackedJob(id: id, kind: kind, detail: detail), at: 0)
        // Keep the list honest rather than unbounded.
        if jobs.count > 40 { jobs.removeLast(jobs.count - 40) }
        save()
        AppLog.info(.editor, "Tracking job \(id) (\(kind.rawValue))")
    }

    func finish(id: String, resultURL: String?) {
        update(id) { $0.state = .finished; $0.resultURL = resultURL }
    }

    func fail(id: String, reason: String?) {
        update(id) { $0.state = .failed; $0.detail = reason }
    }

    func remove(id: String) {
        jobs.removeAll { $0.id == id }
        save()
    }

    func clearFinished() {
        jobs.removeAll { $0.state != .running }
        save()
    }

    /// Asks the server about everything still marked running. This is what
    /// turns "app was killed mid-render" into a result you can collect.
    func refreshUnfinished() async {
        for job in unfinished {
            let status: JobStatusResponse?
            switch job.kind {
            case .removal:    status = try? await APIClient.shared.jobStatus(jobId: job.id)
            case .background: status = try? await APIClient.shared.backgroundStatus(jobId: job.id)
            }
            guard let status else { continue }

            switch status.status {
            case "completed", "export_complete":
                let url: String? = job.kind == .background
                    ? APIClient.shared.backgroundDownloadURL(jobId: job.id).absoluteString
                    : status.resultURL
                finish(id: job.id, resultURL: url)
                AppLog.info(.editor, "Recovered finished job \(job.id)")
            case "failed", "error":
                fail(id: job.id, reason: status.error)
            default:
                break
            }
        }
    }

    private func update(_ id: String, _ change: (inout TrackedJob) -> Void) {
        guard let index = jobs.firstIndex(where: { $0.id == id }) else { return }
        change(&jobs[index])
        save()
    }

    private func load() {
        guard let data = try? Data(contentsOf: url),
              let decoded = try? JSONDecoder().decode([TrackedJob].self, from: data) else { return }
        jobs = decoded
    }

    private func save() {
        guard let data = try? JSONEncoder().encode(jobs) else { return }
        try? data.write(to: url, options: .atomic)
    }
}
