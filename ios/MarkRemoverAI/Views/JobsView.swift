import AVKit
import SwiftUI

/// Renders you left behind. A job keeps going on the GPU whether the app is
/// open or not, so this is where they come back.
struct JobsView: View {
    @ObservedObject private var store = JobStore.shared
    @Environment(\.dismiss) private var dismiss

    @State private var playing: URL?
    @State private var busyID: String?
    @State private var message: String?

    var body: some View {
        NavigationStack {
            Group {
                if store.jobs.isEmpty {
                    ContentUnavailableView(
                        "Nothing here yet",
                        systemImage: "tray",
                        description: Text("Renders you start will show up here, even if you close the app.")
                    )
                } else {
                    List {
                        if !store.unfinished.isEmpty {
                            Section("Still running") {
                                ForEach(store.unfinished) { row($0) }
                            }
                        }
                        let done = store.jobs.filter { $0.state != .running }
                        if !done.isEmpty {
                            Section("Finished") {
                                ForEach(done) { row($0) }
                            }
                        }
                    }
                }
            }
            .navigationTitle("Renders")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Refresh") { Task { await store.refreshUnfinished() } }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
            .sheet(item: $playing) { url in
                NavigationStack {
                    VideoPlayer(player: AVPlayer(url: url))
                        .navigationTitle("Preview")
                        .navigationBarTitleDisplayMode(.inline)
                        .toolbar {
                            ToolbarItem(placement: .topBarTrailing) {
                                Button("Close") { playing = nil }
                            }
                        }
                }
            }
            .overlay(alignment: .bottom) {
                if let message {
                    Text(message)
                        .font(.footnote)
                        .padding(.horizontal, 14).padding(.vertical, 9)
                        .background(.ultraThinMaterial, in: Capsule())
                        .padding(.bottom, 16)
                }
            }
            .task { await store.refreshUnfinished() }
        }
    }

    private func row(_ job: TrackedJob) -> some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack(spacing: 10) {
                Image(systemName: job.kind.symbol)
                    .foregroundStyle(job.kind == .background ? Theme.orange : Theme.accent)
                VStack(alignment: .leading, spacing: 2) {
                    Text(job.kind.title).font(.subheadline.weight(.medium))
                    Text(job.age).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                switch job.state {
                case .running:  ProgressView()
                case .finished: Image(systemName: "checkmark.circle.fill").foregroundStyle(Theme.positive)
                case .failed:   Image(systemName: "xmark.circle.fill").foregroundStyle(.red)
                }
            }

            if job.state == .failed, let detail = job.detail {
                Text(detail).font(.caption).foregroundStyle(.red)
            }

            if job.state == .finished, let raw = job.resultURL {
                HStack(spacing: 10) {
                    Button {
                        Task { await open(job, raw, save: false) }
                    } label: {
                        Label("Preview", systemImage: "play.circle")
                    }
                    Button {
                        Task { await open(job, raw, save: true) }
                    } label: {
                        Label("Save", systemImage: "square.and.arrow.down")
                    }
                    if busyID == job.id { ProgressView().padding(.leading, 4) }
                }
                .font(.caption)
                .buttonStyle(.bordered)
            }
        }
        .padding(.vertical, 4)
        .swipeActions {
            Button("Remove", role: .destructive) { store.remove(id: job.id) }
        }
    }

    private func open(_ job: TrackedJob, _ raw: String, save: Bool) async {
        guard let url = APIClient.shared.absoluteResultURL(raw) else { return }
        busyID = job.id
        defer { busyID = nil }

        guard let local = try? await APIClient.shared.download(url) else {
            message = "Couldn't fetch that file."
            return
        }
        if save {
            do {
                try await PhotoSaver.saveVideo(at: local)
                Haptics.success()
                message = "Saved to your library."
            } catch {
                message = error.localizedDescription
            }
        } else {
            playing = local
        }
    }
}

extension URL: @retroactive Identifiable {
    public var id: String { absoluteString }
}
