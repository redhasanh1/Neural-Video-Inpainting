import SwiftUI

/// One object being removed. Several can be marked before processing, each
/// with its own colour, so it stays obvious which taps belong to which object.
struct Selection: Identifiable, Equatable {
    let id: Int
    var points: [SelectionPoint] = []
    /// Tinted, ready to draw. Nil until SAM2 answers (or if it never does).
    var mask: UIImage?

    var colorIndex: Int { id }
    var color: Color { SelectionPalette.color(colorIndex) }
    var uiColor: UIColor { SelectionPalette.uiColor(colorIndex) }

    static func == (lhs: Selection, rhs: Selection) -> Bool {
        lhs.id == rhs.id && lhs.points == rhs.points && lhs.mask === rhs.mask
    }
}

/// Same palette the website uses, so a mask looks the same in both places.
/// High-saturation on purpose — these sit on top of real footage and have to
/// stay readable over any background.
enum SelectionPalette {
    static let colors: [(r: Double, g: Double, b: Double)] = [
        (0, 255, 255),     // cyan
        (255, 0, 255),     // pink
        (0, 255, 0),       // lime
        (255, 255, 0),     // yellow
        (0, 128, 255),     // arc blue
        (255, 128, 0),     // orange
        (128, 255, 0),     // chartreuse
        (255, 0, 0),       // red
        (0, 255, 128),     // mint
        (255, 0, 128),     // hot pink
        (128, 0, 255)      // violet
    ]

    static func color(_ index: Int) -> Color {
        let c = colors[index % colors.count]
        return Color(red: c.r / 255, green: c.g / 255, blue: c.b / 255)
    }

    static func uiColor(_ index: Int) -> UIColor {
        let c = colors[index % colors.count]
        return UIColor(red: c.r / 255, green: c.g / 255, blue: c.b / 255, alpha: 1)
    }
}
