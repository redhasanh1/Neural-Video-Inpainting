import SwiftUI

@main
struct MarkRemoverAIApp: App {
    @StateObject private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(appState)
                .tint(Theme.accent)
                .task {
                    APIClient.bootstrap()
                    // A render outlives the app; pick up anything left behind.
                    await JobStore.shared.refreshUnfinished()
                }
        }
    }
}

struct RootView: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        Group {
            content
        }
        .animation(.easeInOut(duration: 0.2), value: appState.phase)
        .task { await appState.restoreSession() }
    }

    @ViewBuilder
    private var content: some View {
        Group {
            switch appState.phase {
            case .launching:
                VStack(spacing: 16) {
                    ProgressView().tint(Theme.accent)
                    Text("ObjectRemoverAI").font(.headline).foregroundStyle(.secondary)
                }
            case .signedOut:
                LoginView()
            case .signedIn:
                MainTabs()
            }
        }
    }
}


/// Bottom tabs so the app isn't a single dead-end screen. Create is the job;
/// the other two are the things people reach for when it doesn't go smoothly.
struct MainTabs: View {
    @EnvironmentObject private var appState: AppState
    @State private var selection = 0

    var body: some View {
        TabView(selection: $selection) {
            HomeView()
                .tabItem { Label("Objects", systemImage: "wand.and.stars") }
                .tag(0)

            BackgroundView()
                .tabItem { Label("Background", systemImage: "person.and.background.dotted") }
                .tag(1)

            GuideView()
                .tabItem { Label("Guide", systemImage: "questionmark.circle") }
                .tag(2)

            ProfileView(embedded: true)
                .tabItem { Label("Profile", systemImage: "person.crop.circle") }
                .tag(3)
                .badge(appState.credits < 1 ? "!" : nil)  // nothing left to render today
        }
        // The selected tab's label follows the tool's colour, so the accent
        // matches whichever screen you're actually on.
        .tint(selection == 1 ? Theme.orange : Theme.accent)
    }
}
