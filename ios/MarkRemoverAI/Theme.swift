import SwiftUI

/// Matches the web app's palette so the two don't feel like different products.
enum Theme {
    static let accent = Color(red: 0.42, green: 0.36, blue: 0.90)
    static let accentSoft = Color(red: 0.42, green: 0.36, blue: 0.90).opacity(0.12)
    static let positive = Color(red: 0.18, green: 0.73, blue: 0.44)
    static let warning = Color(red: 0.95, green: 0.62, blue: 0.20)

    /// Background replacement gets its own colour so the two tools stay
    /// distinguishable at a glance — you always know which one you're in.
    static let orange = Color(red: 0.96, green: 0.53, blue: 0.16)
    static let orangeSoft = Color(red: 0.96, green: 0.53, blue: 0.16).opacity(0.14)

    static let orangeGradient = LinearGradient(
        colors: [
            Color(red: 0.98, green: 0.60, blue: 0.15),
            Color(red: 0.93, green: 0.35, blue: 0.22)
        ],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    static let heroGradient = LinearGradient(
        colors: [
            Color(red: 0.42, green: 0.36, blue: 0.90),
            Color(red: 0.72, green: 0.40, blue: 0.86)
        ],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )
}

struct PrimaryButtonStyle: ButtonStyle {
    var enabled: Bool = true
    var gradient: LinearGradient = Theme.heroGradient

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.headline)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 15)
            .background(enabled ? AnyShapeStyle(gradient) : AnyShapeStyle(Color.gray.opacity(0.3)))
            .foregroundStyle(.white)
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
            .opacity(configuration.isPressed ? 0.85 : 1)
            .animation(.easeOut(duration: 0.12), value: configuration.isPressed)
    }
}
