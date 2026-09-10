import AVKit
import PhotosUI
import SwiftUI

struct HomeView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var model = EditorModel()

    @State private var pickerItem: PhotosPickerItem?
    @State private var markMode = 1          // 1 = erase this, 0 = keep this
    @State private var saveMessage: String?
    @State private var showSignOutConfirm = false
    @State private var showPaywall = false
    @State private var showConsole = false

    var body: some View {
        NavigationStack {
            Group {
                switch model.stage {
                case .picking:            picker
                case .selecting:          selector
                case .uploading:          busy(title: "Uploading", detail: "Sending your video straight to storage…", progress: nil)
                case .processing(let p, let m): busy(title: "Erasing", detail: m, progress: p)
                case .done(let url):      result(url)
                case .failed(let message): failure(message)
                }
            }
            .navigationTitle("MarkRemoverAI")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { toolbarContent }
            .sheet(isPresented: $showPaywall) { PaywallView() }
            .sheet(isPresented: $showConsole) { ConsoleView() }
            .confirmationDialog("Sign out?", isPresented: $showSignOutConfirm, titleVisibility: .visible) {
                Button("Sign out", role: .destructive) { Task { await appState.signOut() } }
                Button("Cancel", role: .cancel) {}
            }
        }
        .task { await appState.refreshHealth() }
        .onChange(of: pickerItem) { _, item in
            guard let item else { return }
            Task { await loadPicked(item) }
        }
    }

    // MARK: - Toolbar

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItem(placement: .topBarLeading) {
            HStack(spacing: 6) {
                Circle()
                    .fill(appState.workerOnline == true ? Theme.positive : Theme.warning)
                    .frame(width: 8, height: 8)
                Text(appState.workerOnline == true ? "Online" : "Checking")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        ToolbarItem(placement: .topBarTrailing) {
            Menu {
                Text(appState.user?.email ?? "")
                Button("Get credits") { showPaywall = true }
                Button("Console") { showConsole = true }
                Button("Sign out", role: .destructive) { showSignOutConfirm = true }
            } label: {
                Label("\(Int(appState.credits))", systemImage: "bolt.fill")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(Theme.accent)
            }
        }
    }

    // MARK: - Stages

    private var picker: some View {
        VStack(spacing: 24) {
            Spacer()

            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .fill(Theme.accentSoft)
                .frame(width: 120, height: 120)
                .overlay(
                    Image(systemName: "video.badge.waveform")
                        .font(.system(size: 46))
                        .foregroundStyle(Theme.accent)
                )

            VStack(spacing: 8) {
                Text("Remove anything from a video")
                    .font(.title2.bold())
                    .multilineTextAlignment(.center)
                Text("Pick a clip, tap what you want gone, and the GPU does the rest — original quality preserved.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            .padding(.horizontal, 28)

            PhotosPicker(selection: $pickerItem, matching: .videos, photoLibrary: .shared()) {
                Label("Choose a video", systemImage: "photo.on.rectangle.angled")
            }
            .buttonStyle(PrimaryButtonStyle())
            .padding(.horizontal, 28)

            Spacer()
        }
    }

    private var selector: some View {
        VStack(spacing: 0) {
            if let frame = model.frame {
                FrameCanvas(
                    frame: frame,
                    points: model.points,
                    maskImage: model.maskImage,
                    isBusy: model.isPreviewingMask,
                    onTap: { model.addPoint(normalized: $0, label: markMode) }
                )
                .padding(.horizontal, 16)
                .padding(.top, 8)
            }

            Picker("Tap mode", selection: $markMode) {
                Text("Erase this").tag(1)
                Text("Keep this").tag(0)
            }
            .pickerStyle(.segmented)
            .padding(.horizontal, 16)
            .padding(.top, 14)

            Text(model.points.isEmpty
                 ? "Tap the object you want removed. Add more taps to refine it."
                 : "\(model.points.count) tap\(model.points.count == 1 ? "" : "s") placed."
            )
            .font(.footnote)
            .foregroundStyle(.secondary)
            .padding(.top, 10)

            if model.maskUnavailable {
                Label("Live preview is unavailable, but processing will still work.",
                      systemImage: "info.circle")
                    .font(.caption)
                    .foregroundStyle(Theme.warning)
                    .padding(.top, 6)
            }

            Spacer(minLength: 12)

            VStack(spacing: 10) {
                HStack(spacing: 12) {
                    Button("Undo") { model.undoLastPoint() }
                        .disabled(model.points.isEmpty)
                    Button("Clear") { model.clearPoints() }
                        .disabled(model.points.isEmpty)
                    Button("New video") { model.reset(); pickerItem = nil }
                }
                .font(.subheadline)
                .buttonStyle(.bordered)

                if appState.credits < 1 {
                    Button("Get credits to continue") { showPaywall = true }
                        .buttonStyle(PrimaryButtonStyle())
                } else {
                    Button("Remove it — 1 credit") {
                        Task { await model.process(appState: appState) }
                    }
                    .buttonStyle(PrimaryButtonStyle(enabled: model.canProcess))
                    .disabled(!model.canProcess)
                }
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 16)
        }
    }

    private func busy(title: String, detail: String, progress: Int?) -> some View {
        VStack(spacing: 18) {
            Spacer()
            ProgressView(value: progress.map { Double($0) / 100 } ?? 0)
                .progressViewStyle(.circular)
                .scaleEffect(1.6)
                .tint(Theme.accent)

            Text(title).font(.title3.bold())
            Text(detail)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)

            if let progress, progress > 0 {
                Text("\(progress)%")
                    .font(.headline.monospacedDigit())
                    .foregroundStyle(Theme.accent)
            }

            Text("Big clips take a few minutes. You can leave this screen open.")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .padding(.top, 6)

            Spacer()
        }
    }

    private func result(_ url: URL) -> some View {
        VStack(spacing: 16) {
            VideoPlayer(player: AVPlayer(url: url))
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                .padding(.horizontal, 16)
                .padding(.top, 8)

            if let saveMessage {
                Text(saveMessage)
                    .font(.footnote)
                    .foregroundStyle(Theme.positive)
            }

            Spacer(minLength: 8)

            VStack(spacing: 10) {
                Button("Save to Photos") {
                    Task {
                        do {
                            try await PhotoSaver.saveVideo(at: url)
                            saveMessage = "Saved to your library."
                        } catch {
                            saveMessage = error.localizedDescription
                        }
                    }
                }
                .buttonStyle(PrimaryButtonStyle())

                ShareLink(item: url) { Text("Share").frame(maxWidth: .infinity) }
                    .buttonStyle(.bordered)

                Button("Do another") {
                    saveMessage = nil
                    model.reset()
                    pickerItem = nil
                }
                .font(.subheadline)
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 16)
        }
    }

    private func failure(_ message: String) -> some View {
        VStack(spacing: 18) {
            Spacer()
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 44))
                .foregroundStyle(Theme.warning)
            Text("That didn't work")
                .font(.title3.bold())
            Text(message)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
            let outOfCredits = message.localizedCaseInsensitiveContains("credit")

            VStack(spacing: 10) {
                if outOfCredits {
                    Button("Get credits") { showPaywall = true }
                        .buttonStyle(PrimaryButtonStyle())
                    Button("Start over") {
                        model.reset()
                        pickerItem = nil
                    }
                    .buttonStyle(.bordered)
                } else {
                    Button("Start over") {
                        model.reset()
                        pickerItem = nil
                    }
                    .buttonStyle(PrimaryButtonStyle())
                }
            }
            .padding(.horizontal, 32)
            Spacer()
        }
    }

    // MARK: - Picking

    private func loadPicked(_ item: PhotosPickerItem) async {
        // Copy out of the picker sandbox first — the transferable URL isn't
        // guaranteed to outlive this call.
        guard let movie = try? await item.loadTransferable(type: VideoFile.self) else {
            model.stage = .failed("Couldn't read that video.")
            return
        }
        await model.load(videoURL: movie.url)
    }
}

/// PhotosPicker hands videos over as a file the receiver must copy.
struct VideoFile: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent("pick-\(UUID().uuidString).\(received.file.pathExtension)")
            try? FileManager.default.removeItem(at: dest)
            try FileManager.default.copyItem(at: received.file, to: dest)
            return VideoFile(url: dest)
        }
    }
}
