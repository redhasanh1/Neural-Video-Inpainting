import UIKit

/// Marking a point you can't see land is disorienting, so every tap that
/// registers gets a physical confirmation.
enum Haptics {
    private static let impact = UIImpactFeedbackGenerator(style: .light)
    private static let notice = UINotificationFeedbackGenerator()

    static func tap() {
        impact.prepare()
        impact.impactOccurred()
    }

    static func tick() {
        UISelectionFeedbackGenerator().selectionChanged()
    }

    static func success() {
        notice.notificationOccurred(.success)
    }

    static func failure() {
        notice.notificationOccurred(.error)
    }
}
