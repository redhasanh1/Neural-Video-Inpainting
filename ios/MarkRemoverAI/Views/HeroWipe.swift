import SwiftUI

/// Two stills side by side: the same scene, one with the ball and one without.
/// Each is dressed as a video — play badge and a scrub bar — so it reads as
/// footage rather than a photo. No interaction: it is an illustration, not a
/// control.
struct HeroWipe: View {
    var height: CGFloat = 190

    var body: some View {
        HStack(spacing: 10) {
            panel(showSubject: true, label: "Before")
            Image(systemName: "arrow.right")
                .font(.subheadline.bold())
                .foregroundStyle(.secondary)
            panel(showSubject: false, label: "After")
        }
        .frame(height: height)
    }

    private func panel(showSubject: Bool, label: String) -> some View {
        GeometryReader { geo in
            let w = geo.size.width, h = geo.size.height

            ZStack {
                scene(w: w, h: h, showSubject: showSubject)

                // A camera glyph in the corner, not a centred play button —
                // a play button in the middle of an illustration reads as a
                // control and invites tapping something that does nothing.
                Image(systemName: "video.fill")
                    .font(.system(size: h * 0.10))
                    .foregroundStyle(.white)
                    .shadow(color: .black.opacity(0.35), radius: 2)
                    .position(x: w * 0.13, y: h * 0.13)

                VStack(spacing: 0) {
                    Spacer()
                    ZStack(alignment: .leading) {
                        Capsule().fill(.white.opacity(0.35))
                        Capsule().fill(.white).frame(width: w * 0.38)
                    }
                    .frame(height: 3)
                    .padding(.horizontal, 8)
                    .padding(.bottom, 8)
                }

                Text(label)
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 8).padding(.vertical, 3)
                    .background(.black.opacity(0.4), in: Capsule())
                    .position(x: w * 0.68, y: h * 0.13)
            }
            .frame(width: w, height: h)
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .stroke(.black.opacity(0.08), lineWidth: 1)
            )
        }
    }

    /// Each layer is clipped to the frame before the next is stacked: an
    /// oversized child inside a ZStack widens it, which silently shifts every
    /// `.position` placed alongside.
    private func scene(w: CGFloat, h: CGFloat, showSubject: Bool) -> some View {
        ZStack {
            LinearGradient(
                colors: [Color(red: 0.44, green: 0.72, blue: 0.95),
                         Color(red: 0.80, green: 0.90, blue: 0.98)],
                startPoint: .top, endPoint: .bottom
            )

            Circle()
                .fill(Color(red: 1.0, green: 0.87, blue: 0.45))
                .frame(width: h * 0.20, height: h * 0.20)
                .position(x: w * 0.78, y: h * 0.24)

            Capsule()
                .fill(.white.opacity(0.75))
                .frame(width: w * 0.34, height: h * 0.09)
                .position(x: w * 0.28, y: h * 0.24)

            Ellipse()
                .fill(Color(red: 0.45, green: 0.74, blue: 0.52))
                .frame(width: w * 1.7, height: h * 0.70)
                .position(x: w * 0.5, y: h * 1.06)

            Rectangle()
                .fill(Color(red: 0.38, green: 0.67, blue: 0.45))
                .frame(width: w, height: h * 0.16)
                .position(x: w * 0.5, y: h - h * 0.08)

            if showSubject {
                // A plain lit sphere. The rainbow wheel drew the eye to itself;
                // the point of the picture is that it is gone on the right, not
                // what it looked like.
                ZStack {
                    Circle().fill(
                        RadialGradient(
                            colors: [Color(red: 1.0, green: 0.45, blue: 0.38),
                                     Color(red: 0.82, green: 0.18, blue: 0.22)],
                            center: UnitPoint(x: 0.35, y: 0.30),
                            startRadius: 0,
                            endRadius: h * 0.24
                        )
                    )
                    Circle()
                        .fill(.white.opacity(0.5))
                        .frame(width: h * 0.055, height: h * 0.04)
                        .offset(x: -h * 0.045, y: -h * 0.055)
                        .blur(radius: 1.5)
                }
                .frame(width: h * 0.24, height: h * 0.24)
                .shadow(color: .black.opacity(0.28), radius: 5, y: 3)
                .position(x: w * 0.30, y: h * 0.70)
            }
        }
        .frame(width: w, height: h)
        .clipped()
    }
}
