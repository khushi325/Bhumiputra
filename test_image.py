from PIL import Image

# Since bricks.jpg is in the same folder as this file, we just use the name
path = "bricks.jpg"

try:
    img = Image.open(path).convert("RGB")
    print("✅ Image opened successfully!")
    print("📷 Format:", img.format)
    print("📏 Size:", img.size)
    print("🎨 Mode:", img.mode)
except Exception as e:
    print("❌ Error loading image:", e)
