import SwiftUI

struct GuideView: View {
    @State private var expanded: Set<Int> = []

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 26) {
                    hero
                    steps
                    limits
                    faq
                }
                .padding(.bottom, 30)
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("How it works")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var hero: some View {
        VStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 26, style: .continuous)
                    .fill(Theme.heroGradient)
                    .frame(height: 150)

                HStack(spacing: 22) {
                    demoTile(systemName: "person.fill", struck: true, caption: "Before")
                    Image(systemName: "arrow.right")
                        .font(.title3.bold())
                        .foregroundStyle(.white.opacity(0.9))
                    demoTile(systemName: "checkmark", struck: false, caption: "After")
                }
            }
            .padding(.horizontal, 16)

            Text("Erase anything from a video")
                .font(.title3.bold())
            Text("Point at it once. The GPU rebuilds what was behind it, frame by frame, at your original resolution.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 30)
        }
        .padding(.top, 8)
    }

    private func demoTile(systemName: String, struck: Bool, caption: String) -> some View {
        VStack(spacing: 7) {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(.white.opacity(0.22))
                .frame(width: 66, height: 66)
                .overlay(
                    Image(systemName: systemName)
                        .font(.title2)
                        .foregroundStyle(.white)
                        .opacity(struck ? 0.55 : 1)
                )
                .overlay(
                    struck
                        ? Rectangle().fill(.white).frame(height: 2).rotationEffect(.degrees(-38))
                        : nil
                )
            Text(caption)
                .font(.caption2.weight(.medium))
                .foregroundStyle(.white.opacity(0.9))
        }
    }

    private var steps: some View {
        VStack(alignment: .leading, spacing: 14) {
            sectionTitle("Three steps")

            step(1, "Pick a clip", "Anything from your library: phone footage, a screen recording, a download.", "photo.on.rectangle.angled")
            step(2, "Mark what goes", "Tap something that moves to have it followed, or draw a box over anything that stays put.", "hand.tap")
            step(3, "Let it render", "A minute or two later you get the clean version back.", "wand.and.stars")
        }
        .padding(.horizontal, 16)
    }

    private func step(_ number: Int, _ title: String, _ body: String, _ symbol: String) -> some View {
        HStack(alignment: .top, spacing: 14) {
            ZStack {
                Circle()
                    .fill(Theme.accentSoft)
                    .frame(width: 40, height: 40)
                Image(systemName: symbol)
                    .font(.subheadline)
                    .foregroundStyle(Theme.accent)
            }

            VStack(alignment: .leading, spacing: 3) {
                Text("\(number). \(title)")
                    .font(.subheadline.weight(.semibold))
                Text(body)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(14)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private var limits: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionTitle("Good to know")

            HStack(spacing: 10) {
                fact("90s", "removal")
                fact("10m", "background")
                fact("4K+", "quality kept")
            }
        }
        .padding(.horizontal, 16)
    }

    private func fact(_ value: String, _ caption: String) -> some View {
        VStack(spacing: 3) {
            Text(value)
                .font(.title3.bold())
                .foregroundStyle(Theme.accent)
            Text(caption)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 14)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private var faq: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionTitle("Questions")

            ForEach(Array(Self.questions.enumerated()), id: \.offset) { index, item in
                let isOpen = expanded.contains(index)
                VStack(alignment: .leading, spacing: 8) {
                    Button {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            if isOpen { expanded.remove(index) } else { expanded.insert(index) }
                        }
                    } label: {
                        HStack {
                            Text(item.q)
                                .font(.subheadline.weight(.medium))
                                .multilineTextAlignment(.leading)
                            Spacer(minLength: 8)
                            Image(systemName: "chevron.down")
                                .font(.caption.bold())
                                .foregroundStyle(.secondary)
                                .rotationEffect(.degrees(isOpen ? 180 : 0))
                        }
                    }
                    .buttonStyle(.plain)

                    if isOpen {
                        Text(item.a)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .padding(14)
                .background(Color(.secondarySystemGroupedBackground))
                .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
            }
        }
        .padding(.horizontal, 16)
    }

    private func sectionTitle(_ text: String) -> some View {
        Text(text)
            .font(.headline)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// Same answers the website gives, so the two never contradict each other.
    private static let questions: [(q: String, a: String)] = [
        ("What video formats work?",
         "MP4, MOV, WebM, MKV, WMV and most common formats. Phone footage, camera files and screen recordings all work."),
        ("How long can a video be?",
         "Removal takes up to 90 seconds. Background replacement is a lighter job and allows 10 minutes 10 seconds."),
        ("How does the removal actually work?",
         "The object you mark is tracked across every frame, removed, and the background behind it is rebuilt so the result looks untouched."),
        ("Can I use the results commercially?",
         "Yes. Personal and commercial projects both. You own full rights to what comes out."),
        ("What's the difference between the two modes?",
         "Moves around follows something across the frame. Stays in place holds one area for the whole clip, which is faster and cannot drift."),
        ("How many videos can I do?",
         "Two a day. It refills every morning, so there is nothing to keep track of and nothing to buy."),
        ("Do unused ones stack up?",
         "No. It tops back up to two each day rather than adding to what you had, so saving them up isn't possible."),
        ("When does one get used?",
         "Only when a render finishes. A job that fails costs you nothing.")
    ]
}
