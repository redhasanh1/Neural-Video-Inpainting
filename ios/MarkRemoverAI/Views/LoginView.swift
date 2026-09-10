import AuthenticationServices
import SwiftUI

struct LoginView: View {
    @EnvironmentObject private var appState: AppState

    @State private var isRegistering = false
    @State private var name = ""
    @State private var email = ""
    @State private var password = ""
    @State private var error: String?
    @State private var notice: String?
    @State private var busy = false
    @State private var showHostPicker = false
    @State private var host = APIClient.currentBaseURL
    @State private var showConsole = false

    private var canSubmit: Bool {
        !busy && email.contains("@") && password.count >= 6
            && (!isRegistering || !name.trimmingCharacters(in: .whitespaces).isEmpty)
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 22) {
                header

                socialButtons

                HStack(spacing: 10) {
                    line
                    Text("or")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    line
                }
                .padding(.vertical, 2)

                VStack(spacing: 12) {
                    if isRegistering {
                        field("Name", text: $name, content: .name)
                    }
                    field("Email", text: $email, content: .emailAddress, keyboard: .emailAddress)
                    secureField("Password")
                }

                if let notice {
                    banner(notice, color: Theme.positive)
                }
                if let error {
                    banner(error, color: .red)
                }

                Button {
                    Task { await submit() }
                } label: {
                    if busy {
                        ProgressView().tint(.white)
                    } else {
                        Text(isRegistering ? "Create account" : "Sign in")
                    }
                }
                .buttonStyle(PrimaryButtonStyle(enabled: canSubmit))
                .disabled(!canSubmit)

                Button {
                    withAnimation {
                        isRegistering.toggle()
                        error = nil
                        notice = nil
                    }
                } label: {
                    Text(isRegistering
                         ? "Already have an account? Sign in"
                         : "New here? Create an account")
                    .font(.subheadline)
                    .foregroundStyle(Theme.accent)
                }

                Text("New accounts start with free credits. One credit erases one video.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)

                HStack(spacing: 18) {
                    Button("Can't connect?") { showHostPicker = true }
                    Button("Console") { showConsole = true }
                }
                .font(.caption)
                .foregroundStyle(.tertiary)
            }
            .padding(24)
        }
        .background(Color(.systemGroupedBackground))
        .sheet(isPresented: $showConsole) { ConsoleView() }
        .confirmationDialog(
            "Server",
            isPresented: $showHostPicker,
            titleVisibility: .visible
        ) {
            Button("markremoverai.com (normal)") { select(APIClient.productionBase) }
            Button("Railway host (bypasses DNS filters)") { select(APIClient.railwayBase) }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Currently using \(host). Some school and library networks block the main domain.")
        }
    }

    private func select(_ value: String) {
        APIClient.setBaseURL(value)
        host = value
        error = nil
        notice = "Now using \(value)."
    }

    private var socialButtons: some View {
        VStack(spacing: 10) {
            SignInWithAppleButton(.signIn) { request in
                request.requestedScopes = [.fullName, .email]
            } onCompletion: { result in
                Task { await handleApple(result) }
            }
            .signInWithAppleButtonStyle(.black)
            .frame(height: 50)
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))

            Button {
                Task { await handleGoogle() }
            } label: {
                HStack(spacing: 10) {
                    Image(systemName: "globe")
                        .font(.headline)
                    Text("Continue with Google")
                        .font(.headline)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 15)
                .background(Color(.secondarySystemGroupedBackground))
                .foregroundStyle(.primary)
                .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .stroke(Color(.separator), lineWidth: 1)
                )
            }
            .disabled(busy)
        }
    }

    private var line: some View {
        Rectangle()
            .fill(Color(.separator))
            .frame(height: 1)
    }

    private func handleGoogle() async {
        busy = true
        error = nil
        notice = nil
        defer { busy = false }

        do {
            try await appState.signInWithGoogle()
        } catch {
            guard !SocialAuthError.isCancellation(error) else { return }
            self.error = error.localizedDescription
        }
    }

    private func handleApple(_ result: Result<ASAuthorization, Error>) async {
        error = nil
        notice = nil

        switch result {
        case .failure(let failure):
            guard !SocialAuthError.isCancellation(failure) else { return }
            let nsError = failure as NSError
            // A user tapping Cancel on the Apple sheet is not a failure.
            guard nsError.code != ASAuthorizationError.canceled.rawValue else { return }
            error = failure.localizedDescription

        case .success(let authorization):
            guard
                let credential = authorization.credential as? ASAuthorizationAppleIDCredential,
                let tokenData = credential.identityToken,
                let token = String(data: tokenData, encoding: .utf8)
            else {
                error = SocialAuthError.missingToken.localizedDescription
                return
            }

            let fullName = [credential.fullName?.givenName, credential.fullName?.familyName]
                .compactMap { $0 }
                .joined(separator: " ")

            busy = true
            defer { busy = false }

            do {
                try await appState.signInWithApple(
                    identityToken: token,
                    email: credential.email,
                    name: fullName
                )
            } catch {
                self.error = error.localizedDescription
            }
        }
    }

    private var header: some View {
        VStack(spacing: 10) {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(Theme.heroGradient)
                .frame(width: 68, height: 68)
                .overlay(Image(systemName: "wand.and.stars").font(.system(size: 30)).foregroundStyle(.white))

            Text("MarkRemoverAI")
                .font(.largeTitle.bold())
            Text("Erase anything from your videos.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .padding(.top, 30)
        .padding(.bottom, 8)
    }

    private func field(
        _ label: String,
        text: Binding<String>,
        content: UITextContentType,
        keyboard: UIKeyboardType = .default
    ) -> some View {
        TextField(label, text: text)
            .textContentType(content)
            .keyboardType(keyboard)
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
            .padding(14)
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    private func secureField(_ label: String) -> some View {
        SecureField(label, text: $password)
            .textContentType(isRegistering ? .newPassword : .password)
            .padding(14)
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    private func banner(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.footnote)
            .foregroundStyle(color)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(12)
            .background(color.opacity(0.1))
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    private func submit() async {
        busy = true
        error = nil
        notice = nil
        defer { busy = false }

        do {
            if isRegistering {
                try await appState.register(email: email, password: password, name: name)
                notice = "Account created. Check \(email) for the verification link, then sign in."
                isRegistering = false
            } else {
                try await appState.signIn(email: email, password: password)
            }
        } catch {
            self.error = error.localizedDescription
        }
    }
}
