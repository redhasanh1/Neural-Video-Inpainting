import CoreImage
import UIKit

/// Turns SAM2's black-and-white mask PNG into a tinted, alpha-masked image
/// ready to draw straight over the frame.
///
/// The website rebuilds this per-pixel in JavaScript on every redraw — a loop
/// over every canvas pixel for every mask. At 4K that is ~8M iterations a
/// frame, at 8K ~33M. Doing it once per mask on the GPU keeps the canvas
/// responsive no matter the resolution.
enum MaskRenderer {
    private static let context: CIContext = {
        // Software renderer off; this runs on Metal where available.
        CIContext(options: [.useSoftwareRenderer: false])
    }()

    /// Decodes the base64 PNG and recolours it so white becomes `color` at full
    /// alpha and black becomes fully transparent.
    static func tinted(base64 mask: String, color: UIColor) -> UIImage? {
        guard let data = Data(base64Encoded: mask),
              let source = CIImage(data: data) else { return nil }
        return tinted(source, color: color)
    }

    static func tinted(_ source: CIImage, color: UIColor) -> UIImage? {
        var red: CGFloat = 0, green: CGFloat = 0, blue: CGFloat = 0, alpha: CGFloat = 0
        color.getRed(&red, green: &green, blue: &blue, alpha: &alpha)

        // Flatten RGB to a constant colour via the bias, and lift the mask's
        // luminance into the alpha channel. Rec. 709 coefficients, so a grey
        // anti-aliased mask edge fades instead of hard-clipping.
        guard let filter = CIFilter(name: "CIColorMatrix") else { return nil }
        filter.setValue(source, forKey: kCIInputImageKey)
        filter.setValue(CIVector(x: 0, y: 0, z: 0, w: 0), forKey: "inputRVector")
        filter.setValue(CIVector(x: 0, y: 0, z: 0, w: 0), forKey: "inputGVector")
        filter.setValue(CIVector(x: 0, y: 0, z: 0, w: 0), forKey: "inputBVector")
        filter.setValue(CIVector(x: 0.2126, y: 0.7152, z: 0.0722, w: 0), forKey: "inputAVector")
        filter.setValue(CIVector(x: red, y: green, z: blue, w: 0), forKey: "inputBiasVector")

        guard let output = filter.outputImage,
              let cgImage = context.createCGImage(output, from: output.extent) else { return nil }
        return UIImage(cgImage: cgImage)
    }
}
