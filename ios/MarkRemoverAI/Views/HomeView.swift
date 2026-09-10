import AVKit
import PhotosUI
import SwiftUI

struct HomeView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var model = EditorModel()

    @State private var pickerItem: PhotosPickerItem?
    @State private var markMode = 1          // 1 = erase this, 0 = keep this
    @State private var maskOpacity: Double = 0.55
    @State private var peeking = false
    @State private var tool: StaticMaskBuilder.Tool = .rectangle
    @State private var brushSize: Double = 0.04
    @State private var showCompare = true
    @State private var saveMessage: String?
    @State private var showSignOutConfirm = false
    @State private var showConsole = false
    @State private var showProfile = false

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
            .navigationTitle("ObjectRemoverAI")
            // Large, so the name sits on its own row below the badges rather
            // than being squeezed between them into "ObjectRemo…".
            .navigationBarTitleDisplayMode(.large)
            .toolbar { toolbarContent }
            .sheet(isPresented: $showConsole) { ConsoleView() }
            .sheet(isPresented: $showProfile) { ProfileView() }
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
            HStack(spacing: 10) {
                // A bare dot says nothing and tapping it did nothing. This
                // labels itself and opens the detail behind it, matching the
                // "API: online" badge on the website.
                Menu {
                    Label(
                        appState.workerOnline == true
                            ? "The GPU service is responding"
                            : "The GPU service isn't responding",
                        systemImage: appState.workerOnline == true ? "checkmark.circle" : "exclamationmark.triangle"
                    )
                    Text(APIClient.currentBaseURL.replacingOccurrences(of: "https://", with: ""))
                    Divider()
                    Button {
                        Task { await appState.refreshHealth() }
                    } label: {
                        Label("Check again", systemImage: "arrow.clockwise")
                    }
                } label: {
                    HStack(spacing: 5) {
                        Circle()
                            .fill(appState.workerOnline == true ? Theme.positive : Theme.warning)
                            .frame(width: 7, height: 7)
                        Text("API")
                            .font(.caption.weight(.semibold))
                    }
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Capsule().fill(Color(.tertiarySystemFill)))
                }
                .accessibilityLabel(appState.workerOnline == true ? "Service online" : "Service unreachable")

                // Swap clips without backing out to the start screen.
                if model.frame != nil {
                    PhotosPicker(selection: $pickerItem, matching: .videos, photoLibrary: .shared()) {
                        Image(systemName: "arrow.triangle.2.circlepath")
                            .font(.subheadline)
                    }
                }
            }
        }
        ToolbarItem(placement: .topBarTrailing) {
            HStack(spacing: 10) {
                // Read-only. There is nothing to open from here, so it is
                // a label rather than a button. An HStack, not a Label - the
                // toolbar renders Label as icon-only and the number vanishes.
                HStack(spacing: 3) {
                    Image(systemName: "bolt.fill").font(.caption)
                    Text(CreditEstimate.compact(appState.credits))
                        .font(.subheadline.weight(.semibold))
                        .monospacedDigit()
                }
                .foregroundStyle(Theme.accent)
                .accessibilityLabel("\(CreditEstimate.label(appState.credits)) renders left")

                Button { showProfile = true } label: {
                    Circle()
                        .fill(Theme.heroGradient)
                        .frame(width: 28, height: 28)
                        .overlay(
                            Text(initials)
                                .font(.caption.bold())
                                .foregroundStyle(.white)
                        )
                }
            }
        }
    }

    /// Tools for the stationary mode. Box covers most shapes in one drag;
    /// brush handles anything irregular.
    private var drawTools: some View {
        VStack(spacing: 10) {
            HStack(spacing: 8) {
                ForEach(StaticMaskBuilder.Tool.allCases) { option in
                    Button {
                        tool = option
                        Haptics.tick()
                    } label: {
                        VStack(spacing: 3) {
                            Image(systemName: option.symbol).font(.subheadline)
                            Text(option.title).font(.caption2)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                        .background(
                            RoundedRectangle(cornerRadius: 10, style: .continuous)
                                .fill(tool == option ? Theme.accentSoft : Color(.tertiarySystemFill))
                        )
                        .foregroundStyle(tool == option ? Theme.accent : .primary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 16)

            if tool == .polygon {
                HStack(spacing: 10) {
                    Text("\(model.maskBuilder.pendingPolygon.count) point\(model.maskBuilder.pendingPolygon.count == 1 ? "" : "s")")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Undo point") { model.maskBuilder.undoPolygonVertex() }
                        .disabled(model.maskBuilder.pendingPolygon.isEmpty)
                    Button("Close shape") { model.maskBuilder.closePolygon(); Haptics.tap() }
                        .disabled(model.maskBuilder.pendingPolygon.count < 3)
                }
                .font(.caption)
                .buttonStyle(.bordered)
                .padding(.horizontal, 16)
            }

            if tool == .brush || tool == .eraser {
                HStack(spacing: 10) {
                    Image(systemName: "circle.fill").font(.system(size: 7))
                    Slider(value: $brushSize, in: 0.01...0.25).tint(Theme.accent)
                    Image(systemName: "circle.fill").font(.system(size: 15))
                }
                .foregroundStyle(.secondary)
                .padding(.horizontal, 20)
            }
        }
    }

    /// One chip per object, in its own colour, so it stays obvious which taps
    /// belong together and which one the next tap will extend.
    private var objectChips: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(model.selections) { selection in
                    let isActive = selection.id == model.activeSelectionID
                    Button {
                        model.activeSelectionID = isActive ? nil : selection.id
                    } label: {
                        HStack(spacing: 6) {
                            Circle()
                                .fill(selection.color)
                                .frame(width: 10, height: 10)
                            Text("Object \(selection.id + 1)")
                                .font(.caption.weight(.medium))
                            Text("\(selection.points.count)")
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                        .padding(.horizontal, 11)
                        .padding(.vertical, 7)
                        .background(
                            Capsule().fill(isActive ? selection.color.opacity(0.18) : Color(.tertiarySystemFill))
                        )
                        .overlay(
                            Capsule().stroke(isActive ? selection.color : .clear, lineWidth: 1.5)
                        )
                    }
                    .buttonStyle(.plain)
                    .contextMenu {
                        Button("Delete", role: .destructive) { model.removeSelection(selection.id) }
                    }
                }
            }
            .padding(.horizontal, 20)
        }
    }

    private var initials: String {
        let name = appState.user?.name?.trimmingCharacters(in: .whitespaces)
        let source = (name?.isEmpty == false ? name! : appState.user?.email) ?? "?"
        return String(source.prefix(1)).uppercased()
    }

    // MARK: - Stages

    private var picker: some View {
        ScrollView {
            VStack(spacing: 22) {
                heroCard

                HStack(spacing: 12) {
                    modeCard(
                        title: "Moves around",
                        detail: "A person, a car, anything that travels across the frame.",
                        symbol: "figure.walk.motion",
                        tint: Theme.accent
                    )
                    modeCard(
                        title: "Stays in place",
                        detail: "Anything sitting in the same spot the whole way through.",
                        symbol: "seal",
                        tint: Color(red: 0.85, green: 0.42, blue: 0.68)
                    )
                }
                .padding(.horizontal, 16)

                PhotosPicker(selection: $pickerItem, matching: .videos, photoLibrary: .shared()) {
                    Label("Choose a video", systemImage: "photo.on.rectangle.angled")
                }
                .buttonStyle(PrimaryButtonStyle())
                .padding(.horizontal, 16)

                HStack(spacing: 6) {
                    Image(systemName: "clock").font(.caption2)
                    Text("Up to 90 seconds")
                        .font(.caption)
                }
                .foregroundStyle(.secondary)

                if appState.credits < 1 {
                    // Says when it comes back rather than offering a way out,
                    // because waiting is the only way out.
                    HStack(spacing: 8) {
                        Image(systemName: "bolt.slash.fill")
                        Text("That's today's two videos. Two more tomorrow.")
                            .font(.subheadline.weight(.medium))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 13)
                    .background(Theme.warning.opacity(0.15))
                    .foregroundStyle(Theme.warning)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .padding(.horizontal, 16)
                }
            }
            .padding(.vertical, 14)
        }
        .background(Color(.systemGroupedBackground))
    }

    /// The landing screen carries the whole first impression, so it shows the
    /// before/after idea rather than describing it.
    /// Real frames from a real render, wipeable. The old version was two
    /// SF Symbols on a gradient, which showed nothing about what the app does.
    private var heroCard: some View {
        VStack(spacing: 12) {
            HeroWipe(height: 190)
                .padding(.horizontal, 16)

            VStack(spacing: 3) {
                Text("Remove anything from a video")
                    .font(.title3.bold())
                Text("Original quality preserved")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func modeCard(title: String, detail: String, symbol: String, tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            Image(systemName: symbol)
                .font(.title3)
                .foregroundStyle(tint)
            Text(title)
                .font(.subheadline.weight(.semibold))
            Text(detail)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, minHeight: 128, alignment: .topLeading)
        .padding(13)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    private var selector: some View {
        ScrollView {
            VStack(spacing: 0) {
            Picker("Mode", selection: $model.mode) {
                ForEach(EditorModel.Mode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal, 16)
            .padding(.top, 8)
            .onChange(of: model.mode) { _, _ in
                // Discard the other mode's marks rather than hiding them.
                model.clearAll()
                Haptics.tick()
            }

            if let frame = model.frame {
                EstimateBar(
                    duration: model.duration,
                    size: frame.pixelSize,
                    credits: model.estimatedCredits
                )
                .padding(.horizontal, 16)
                .padding(.top, 10)

                VideoCanvas(
                    frame: frame,
                    selections: model.selections,
                    activeSelectionID: model.activeSelectionID,
                    isBusy: model.isPreviewingMask,
                    maskOpacity: maskOpacity,
                    peeking: peeking,
                    drawnMask: model.mode == .fixed ? model.maskBuilder.preview : nil,
                    isDrawing: model.mode == .fixed && tool != .polygon,
                    polygonVertices: model.mode == .fixed && tool == .polygon
                        ? model.maskBuilder.pendingPolygon : [],
                    onDraw: { from, to in
                        model.maskBuilder.stroke(
                            from: from, to: to, tool: tool, brushFraction: brushSize
                        )
                    },
                    onDrawEnded: { model.maskBuilder.endStroke() },
                    onTap: { location in
                        if model.mode == .fixed {
                            guard tool == .polygon else { return }
                            model.maskBuilder.addPolygonVertex(location)
                        } else {
                            model.addPoint(normalized: location, label: markMode)
                        }
                    }
                )
                .frame(maxHeight: 300)
                .padding(.horizontal, 16)
                .padding(.top, 8)

                if let url = model.sourceURL, model.duration > 0 {
                    FrameScrubber(
                        videoURL: url,
                        duration: model.duration,
                        time: $model.currentTime,
                        onCommit: { model.seek(to: $0) }
                    )
                    .padding(.horizontal, 16)
                    .padding(.top, 10)
                }
            }

            if model.mode == .moving {
                VStack(spacing: 6) {
                    Text("Your next tap")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)

                    HStack(spacing: 10) {
                        ForEach([1, 0], id: \.self) { mode in
                            Button {
                                markMode = mode
                                Haptics.tick()
                            } label: {
                                HStack(spacing: 7) {
                                    Image(systemName: mode == 1 ? "plus.circle.fill" : "minus.circle.fill")
                                        .font(.title3)
                                    Text(mode == 1 ? "this" : "not this")
                                        .font(.subheadline.weight(.medium))
                                }
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 11)
                                .background(
                                    RoundedRectangle(cornerRadius: 11, style: .continuous)
                                        .fill(markMode == mode
                                              ? (mode == 1 ? Theme.positive.opacity(0.18) : Color.red.opacity(0.15))
                                              : Color(.tertiarySystemFill))
                                )
                                .foregroundStyle(markMode == mode
                                                 ? (mode == 1 ? Theme.positive : .red)
                                                 : .primary)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
                .padding(.horizontal, 16)
                .padding(.top, 14)
            } else {
                drawTools
                    .padding(.top, 12)
            }

            if model.mode == .fixed {
                Text(EditorModel.Mode.fixed.hint)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 24)
                    .padding(.top, 10)
            } else if model.selections.isEmpty {
                Text("Tap what you want gone. Pinch to zoom in first if it's small.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 24)
                    .padding(.top, 10)
            } else if model.mode == .moving {
                objectChips
                    .padding(.top, 10)

                HStack(spacing: 10) {
                    Image(systemName: "circle.lefthalf.filled").font(.caption)
                    Slider(value: $maskOpacity, in: 0...1)
                        .tint(Theme.accent)
                    Text("\(Int(maskOpacity * 100))%")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .frame(width: 34, alignment: .trailing)

                    // Hold to drop the overlays and check what is underneath.
                    Image(systemName: peeking ? "eye.fill" : "eye")
                        .font(.subheadline)
                        .foregroundStyle(peeking ? Theme.accent : .secondary)
                        .frame(width: 30, height: 30)
                        .contentShape(Rectangle())
                        .gesture(
                            DragGesture(minimumDistance: 0)
                                .onChanged { _ in
                                    if !peeking { Haptics.tick(); peeking = true }
                                }
                                .onEnded { _ in peeking = false }
                        )
                        .accessibilityLabel("Hold to hide the mask")
                }
                .padding(.horizontal, 20)
                .padding(.top, 8)
            }

            if model.maskUnavailable {
                Label("Live preview is unavailable, but processing will still work.",
                      systemImage: "info.circle")
                    .font(.caption)
                    .foregroundStyle(Theme.warning)
                    .padding(.top, 6)
            }

            VStack(spacing: 10) {
                HStack(spacing: 10) {
                    if model.mode == .moving {
                        Button("Undo") { model.undoLastPoint() }
                            .disabled(model.selections.isEmpty)
                        Button("Clear") { model.clearAll() }
                            .disabled(model.selections.isEmpty)
                        Button {
                            model.startNewObject()
                        } label: {
                            Label("Next object", systemImage: "plus.circle")
                        }
                        .disabled(model.activeSelection?.points.isEmpty != false)
                    } else {
                        Button("Clear mask") { model.clearAll() }
                            .disabled(model.maskBuilder.isEmpty)
                    }
                    Button("New video") { model.reset(); pickerItem = nil }
                }
                .font(.subheadline)
                .buttonStyle(.bordered)

            }
            .padding(.horizontal, 16)
            .padding(.bottom, 12)
            }
        }
        .safeAreaInset(edge: .bottom) {
            // Pinned so the primary action is never scrolled off screen.
            Group {
                if appState.credits < model.estimatedCredits {
                    Button("Nothing left for today") {}
                        .buttonStyle(PrimaryButtonStyle(enabled: false))
                        .disabled(true)
                } else {
                    Button("Remove it · \(CreditEstimate.label(model.estimatedCredits)) credit\(model.estimatedCredits == 1 ? "" : "s")") {
                        Task { await model.process(appState: appState) }
                    }
                    .buttonStyle(PrimaryButtonStyle(enabled: model.canProcess))
                    .disabled(!model.canProcess)
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(.bar)
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

            // The same estimate shown before the button was pressed, repeated
            // here so a long render never looks like a stalled one.
            Text(waitText)
                .font(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
                .padding(.top, 6)

            Spacer()
        }
    }

    private func result(_ url: URL) -> some View {
        VStack(spacing: 16) {
            Group {
                if showCompare, let original = model.sourceURL {
                    CompareView(beforeURL: original, afterURL: url)
                } else {
                    VideoPlayer(player: AVPlayer(url: url))
                        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                }
            }
            .padding(.horizontal, 16)
            .padding(.top, 8)

            if model.sourceURL != nil {
                Picker("View", selection: $showCompare) {
                    Text("Compare").tag(true)
                    Text("Result").tag(false)
                }
                .pickerStyle(.segmented)
                .padding(.horizontal, 16)
            }

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
                            Haptics.success()
                            saveMessage = "Saved to your library."
                        } catch {
                            Haptics.failure()
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
            // Every failure leads to the same place now: there is no
            // "top up" branch to offer, and running out is phrased as a
            // wait rather than an error.
            Button("Start over") {
                model.reset()
                pickerItem = nil
            }
            .buttonStyle(PrimaryButtonStyle())
            .padding(.horizontal, 32)
            Spacer()
        }
    }

    /// What to tell someone while they wait. Falls back to the vague line only
    /// when the clip's dimensions were never read.
    private var waitText: String {
        guard let frame = model.frame else {
            return "Big clips take a few minutes. You can leave this screen open."
        }
        return CreditEstimate.waitLabel(
            duration: model.duration,
            size: frame.pixelSize,
            isBackground: false
        ) + " You can leave this screen open."
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
