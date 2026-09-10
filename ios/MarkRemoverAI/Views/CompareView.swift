import AVKit
import SwiftUI

/// Before/after on one screen. Drag the handle to wipe between the original
/// and the result — far easier to judge than flipping between two players,
/// and it keeps both clips in sync.
struct CompareView: View {
    let beforeURL: URL
    let afterURL: URL

    @State private var split: CGFloat = 0.5
    @State private var beforePlayer: AVPlayer?
    @State private var afterPlayer: AVPlayer?
    @State private var isPlaying = false

    var body: some View {
        VStack(spacing: 12) {
            GeometryReader { geo in
                ZStack(alignment: .topLeading) {
                    if let afterPlayer {
                        VideoPlayer(player: afterPlayer)
                            .disabled(true)
                    }

                    if let beforePlayer {
                        VideoPlayer(player: beforePlayer)
                            .disabled(true)
                            .mask(alignment: .leading) {
                                Rectangle().frame(width: geo.size.width * split)
                            }
                    }

                    // Wipe handle
                    ZStack {
                        Rectangle()
                            .fill(.white)
                            .frame(width: 2)
                            .shadow(radius: 2)
                        Circle()
                            .fill(.white)
                            .frame(width: 34, height: 34)
                            .shadow(radius: 3)
                            .overlay(
                                Image(systemName: "arrow.left.and.right")
                                    .font(.caption.bold())
                                    .foregroundStyle(.black)
                            )
                    }
                    .position(x: geo.size.width * split, y: geo.size.height / 2)

                    label("Before", alignment: .leading)
                        .position(x: 46, y: 22)
                        .opacity(split > 0.14 ? 1 : 0)

                    label("After", alignment: .trailing)
                        .position(x: geo.size.width - 42, y: 22)
                        .opacity(split < 0.86 ? 1 : 0)
                }
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in
                            split = min(max(value.location.x / geo.size.width, 0), 1)
                        }
                )
            }
            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))

            Button {
                togglePlayback()
            } label: {
                Label(isPlaying ? "Pause" : "Play both", systemImage: isPlaying ? "pause.fill" : "play.fill")
                    .font(.subheadline.weight(.medium))
            }
            .buttonStyle(.bordered)
        }
        .task { await prepare() }
        .onDisappear {
            beforePlayer?.pause()
            afterPlayer?.pause()
        }
    }

    private func label(_ text: String, alignment: Alignment) -> some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 9)
            .padding(.vertical, 4)
            .background(.ultraThinMaterial, in: Capsule())
    }

    private func prepare() async {
        let before = AVPlayer(url: beforeURL)
        let after = AVPlayer(url: afterURL)
        before.isMuted = true
        after.isMuted = true
        beforePlayer = before
        afterPlayer = after
    }

    private func togglePlayback() {
        guard let beforePlayer, let afterPlayer else { return }

        if isPlaying {
            beforePlayer.pause()
            afterPlayer.pause()
        } else {
            // Restart together so the wipe compares the same moment on both.
            beforePlayer.seek(to: .zero)
            afterPlayer.seek(to: .zero)
            beforePlayer.play()
            afterPlayer.play()
        }
        isPlaying.toggle()
    }
}
