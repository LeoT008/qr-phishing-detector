import qrcode

# Create a plain QR code with a clear white border.
qr = qrcode.QRCode(box_size=10, border=4)
qr.add_data("https://www.google.com")
qr.make(fit=True)

image = qr.make_image(fill_color="black", back_color="white")
image.save("plain_test_qr.png")

print("Saved plain_test_qr.png")