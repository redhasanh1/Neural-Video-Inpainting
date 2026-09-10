import Foundation
import OSLog

/// Everything the app does that can fail goes through here. It writes to the
/// unified log (visible in Console.app and `xcrun simctl spawn booted log
/// stream`) and keeps a ring buffer the in-app console can show, because the
/// interesting failures usually happen on a device with no Xcode attached.
enum AppLog {
    enum Category: String, CaseIterable {
        case net, auth, store, editor, app
    }

    static let subsystem = "com.markremoverai.app"

    private static let loggers: [Category: Logger] = Dictionary(
        uniqueKeysWithValues: Category.allCases.map {
            ($0, Logger(subsystem: subsystem, category: $0.rawValue))
        }
    )

    static func debug(_ category: Category, _ message: String) {
        loggers[category]?.debug("\(message, privacy: .public)")
        LogStore.shared.append(.init(level: .debug, category: category, message: message))
    }

    static func info(_ category: Category, _ message: String) {
        loggers[category]?.info("\(message, privacy: .public)")
        LogStore.shared.append(.init(level: .info, category: category, message: message))
    }

    static func error(_ category: Category, _ message: String) {
        loggers[category]?.error("\(message, privacy: .public)")
        LogStore.shared.append(.init(level: .error, category: category, message: message))
    }
}

struct LogEntry: Identifiable, Sendable {
    enum Level: String, Sendable {
        case debug, info, error

        var symbol: String {
            switch self {
            case .debug: return "•"
            case .info: return "›"
            case .error: return "✕"
            }
        }
    }

    let id = UUID()
    let date = Date()
    let level: Level
    let category: AppLog.Category
    let message: String

    var line: String {
        let time = LogStore.timeFormatter.string(from: date)
        return "\(time) \(level.symbol) [\(category.rawValue)] \(message)"
    }
}

/// Bounded on purpose — a long video job can emit a lot of polling noise, and
/// an unbounded buffer would just grow until the app is killed.
@MainActor
final class LogStore: ObservableObject {
    static let shared = LogStore()

    private static let limit = 500

    static let timeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()

    @Published private(set) var entries: [LogEntry] = []

    private init() {}

    nonisolated func append(_ entry: LogEntry) {
        Task { @MainActor in
            entries.append(entry)
            if entries.count > Self.limit {
                entries.removeFirst(entries.count - Self.limit)
            }
        }
    }

    func clear() { entries.removeAll() }

    /// The last few failures, for attaching to a support email.
    func recentErrors(limit: Int) -> [String] {
        entries.filter { $0.level == .error }.suffix(limit).map(\.line)
    }

    var transcript: String {
        entries.map(\.line).joined(separator: "\n")
    }
}
