import SwiftUI

/// Duration, resolution and cost, shown once a clip is loaded — the same three
/// facts the website puts above the video. Quoting the price before the button
/// is pressed is the point: nobody should discover the cost by being charged.
struct EstimateBar: View {
    let duration: Double
    let size: CGSize
    let credits: Double
    /// Background replacement runs a lighter pipeline, so the wait differs.
    var isBackground: Bool = false
    var tint: Color = Theme.accent

    var body: some View {
        HStack(spacing: 0) {
            item(String(format: "%.0fs", duration.rounded()), "Duration")
            divider
            item("\(Int(size.width))×\(Int(size.height))", "Resolution")
            divider
            // Saying how long it takes up front is what stops a normal render
            // reading as a hang three minutes in.
            item(
                CreditEstimate.timeLabel(
                    CreditEstimate.seconds(duration: duration, size: size, isBackground: isBackground)
                ),
                "Est. time"
            )
            divider
            item(
                "\(CreditEstimate.label(credits)) credit\(credits == 1 ? "" : "s")",
                "Cost",
                highlighted: true
            )
        }
        .padding(.vertical, 11)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private var divider: some View {
        Rectangle()
            .fill(Color(.separator))
            .frame(width: 1, height: 26)
    }

    private func item(_ value: String, _ caption: String, highlighted: Bool = false) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(highlighted ? tint : .primary)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
            Text(caption)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}
