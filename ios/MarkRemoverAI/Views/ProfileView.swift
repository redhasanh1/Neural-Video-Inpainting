import SwiftUI

struct ProfileView: View {
    /// As a tab there's nothing to dismiss, so the Done button is dropped.
    var embedded = false

    @EnvironmentObject private var appState: AppState
    @Environment(\.dismiss) private var dismiss

    @State private var showConsole = false
    @State private var showJobs = false
    @State private var showDeleteAccount = false
    @State private var deleteConfirmation = ""
    @State private var deleting = false
    @State private var deleteError: String?
    @State private var showSignOutConfirm = false
    @State private var showHostPicker = false
    @State private var host = APIClient.currentBaseURL

    var body: some View {
        NavigationStack {
            List {
                Section {
                    HStack(spacing: 14) {
                        Circle()
                            .fill(Theme.heroGradient)
                            .frame(width: 54, height: 54)
                            .overlay(
                                Text(initials)
                                    .font(.title3.bold())
                                    .foregroundStyle(.white)
                            )

                        VStack(alignment: .leading, spacing: 3) {
                            Text(appState.user?.name ?? "Signed in")
                                .font(.headline)
                            Text(Self.displayEmail(appState.user?.email))
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                    }
                    .padding(.vertical, 6)
                }

                Section("Renders") {
                    Button { showJobs = true } label: {
                        HStack {
                            Label("Your renders", systemImage: "tray.full")
                            Spacer()
                            if !JobStore.shared.unfinished.isEmpty {
                                Text("\(JobStore.shared.unfinished.count) running")
                                    .font(.caption)
                                    .foregroundStyle(Theme.warning)
                            } else if !JobStore.shared.collectable.isEmpty {
                                Text("\(JobStore.shared.collectable.count) ready")
                                    .font(.caption)
                                    .foregroundStyle(Theme.positive)
                            }
                        }
                    }
                    Text("Renders keep going if you close the app. Collect them here.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("Free videos") {
                    HStack {
                        Label("Left to use", systemImage: "bolt.fill")
                        Spacer()
                        Text("\(Int(appState.credits))")
                            .font(.headline.monospacedDigit())
                            .foregroundStyle(Theme.accent)
                    }
                    Text("Two free videos a day, topped back up every morning. One video is one removal or one background swap.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("Service") {
                    HStack {
                        Label("GPU workers", systemImage: "cpu")
                        Spacer()
                        HStack(spacing: 6) {
                            Circle()
                                .fill(appState.workerOnline == true ? Theme.positive : Theme.warning)
                                .frame(width: 8, height: 8)
                            Text(appState.workerOnline == true ? "Online" : "Unreachable")
                                .foregroundStyle(.secondary)
                        }
                        .font(.subheadline)
                    }

                    Button { showHostPicker = true } label: {
                        VStack(alignment: .leading, spacing: 2) {
                            Label("Server", systemImage: "network")
                            Text(host.replacingOccurrences(of: "https://", with: ""))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                    }

                    Button { showConsole = true } label: {
                        Label("Console", systemImage: "text.alignleft")
                    }
                }

                Section("Help") {
                    Button {
                        openSupportMail()
                    } label: {
                        Label("Email support", systemImage: "envelope")
                    }
                    Text(SupportMail.address)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }

                Section {
                    Link(destination: URL(string: "https://markremoverai.com/terms")!) {
                        Label("Terms of service", systemImage: "doc.text")
                    }
                    Link(destination: URL(string: "https://markremoverai.com/privacy")!) {
                        Label("Privacy policy", systemImage: "hand.raised")
                    }
                }

                Section {
                    Button(role: .destructive) {
                        showSignOutConfirm = true
                    } label: {
                        Label("Sign out", systemImage: "rectangle.portrait.and.arrow.right")
                    }
                }

                Section {
                    Button(role: .destructive) {
                        deleteConfirmation = ""
                        deleteError = nil
                        showDeleteAccount = true
                    } label: {
                        Label("Delete account", systemImage: "trash")
                    }
                } footer: {
                    Text("Removes your account, your uploads and your results. This cannot be undone.")
                }
            }
            .safeAreaInset(edge: .bottom) {
                // The floating tab bar sits over the final row otherwise.
                Color.clear.frame(height: 8)
            }
            .navigationTitle("Profile")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                if !embedded {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("Done") { dismiss() }
                    }
                }
            }
            .sheet(isPresented: $showConsole) { ConsoleView() }
            .sheet(isPresented: $showJobs) { JobsView() }
            .alert("Delete account?", isPresented: $showDeleteAccount) {
                // Typed confirmation rather than a single tap: this destroys
                // the account, its files and any remaining balance.
                TextField("Type DELETE", text: $deleteConfirmation)
                    .textInputAutocapitalization(.characters)
                Button("Cancel", role: .cancel) {}
                Button("Delete", role: .destructive) {
                    guard deleteConfirmation.uppercased() == "DELETE" else { return }
                    Task {
                        deleting = true
                        defer { deleting = false }
                        do {
                            try await appState.deleteAccount()
                            if !embedded { dismiss() }
                        } catch {
                            deleteError = error.localizedDescription
                        }
                    }
                }
            } message: {
                Text("This removes your account, uploads and results permanently. Type DELETE to confirm.")
            }
            .alert("Couldn't delete the account", isPresented: .constant(deleteError != nil)) {
                Button("OK") { deleteError = nil }
            } message: {
                Text(deleteError ?? "")
            }
            .confirmationDialog("Sign out?", isPresented: $showSignOutConfirm, titleVisibility: .visible) {
                Button("Sign out", role: .destructive) {
                    Task {
                        await appState.signOut()
                        if !embedded { dismiss() }
                    }
                }
                Button("Cancel", role: .cancel) {}
            }
            .confirmationDialog("Server", isPresented: $showHostPicker, titleVisibility: .visible) {
                Button("Railway (recommended)") { select(APIClient.railwayBase) }
                Button("markremoverai.com") { select(APIClient.productionBase) }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("The app talks to Railway directly so a blocked domain can't take it offline.")
            }
            .task { await appState.refreshUser() }
        }
    }


    /// What to print under someone's name.
    ///
    /// Sign in with Apple's Hide My Email hands back an address like
    /// 2kgpsd59vw@privaterelay.appleid.com. That is the feature working, not a
    /// fault, but printing the random half of it under a person's name reads
    /// like the app got something wrong. Say what it actually is instead.
    static func displayEmail(_ email: String?) -> String {
        guard let email, !email.isEmpty else { return "" }
        return email.lowercased().hasSuffix("@privaterelay.appleid.com")
            ? "Hidden by Apple"
            : email
    }

    private var initials: String {
        let source = appState.user?.name?.trimmingCharacters(in: .whitespaces).isEmpty == false
            ? appState.user!.name!
            : (appState.user?.email ?? "?")
        return String(source.prefix(1)).uppercased()
    }

    /// Pre-fills the details we'd otherwise have to ask for. Nothing is sent
    /// until the user hits send in Mail — this only composes a draft.
    private func openSupportMail() {
        let url = SupportMail.composeURL(
            email: appState.user?.email,
            credits: appState.credits,
            host: host,
            recentErrors: LogStore.shared.recentErrors(limit: 6)
        )
        guard let url else { return }
        UIApplication.shared.open(url)
    }

    private func select(_ value: String) {
        APIClient.setBaseURL(value)
        host = value
        Task { await appState.refreshHealth() }
    }
}
