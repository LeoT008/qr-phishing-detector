from pathlib import Path
import sys
import cv2


def decode_image(image):
    """Decode locally, using ZXing when available and OpenCV as a fallback."""
    try:
        import zxingcpp
    except ImportError:
        zxingcpp = None

    if zxingcpp is not None:
        result = zxingcpp.read_barcode(image, formats=zxingcpp.BarcodeFormat.QRCode,
                                        try_rotate=True, try_downscale=True)
        if result and result.text:
            return result.text

    detector = cv2.QRCodeDetector()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # A white border and enlarged image help with cropped or small QR codes.
    bordered = cv2.copyMakeBorder(gray, 32, 32, 32, 32, cv2.BORDER_CONSTANT, value=255)
    variants = [image, bordered]
    if max(gray.shape) < 1800:
        variants.append(cv2.resize(bordered, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
    for candidate in variants:
        try:
            payload, _, _ = detector.detectAndDecode(candidate)
        except cv2.error:
            continue
        if payload:
            return payload
    raise ValueError("No readable QR code found. Try a clear, uncropped image with a white border.")


def decode_qr(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Cannot read this image file.")
    return decode_image(image)


if __name__ == "__main__":
    # Accept the image filename from the Terminal command.
    if len(sys.argv) != 2:
        print("Usage: python decode_qr.py image.png")
        sys.exit(1)

    image_path = Path(sys.argv[1]).expanduser()

    if not image_path.is_file():
        print("File not found:", image_path)
        sys.exit(1)

    try:
        payload = decode_qr(image_path)
        print("Decoded content:")
        print(payload)
        print("\nNo website was opened.")
    except ValueError as error:
        print("Error:", error)
        sys.exit(1)