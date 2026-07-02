from PIL import Image

# 👇 paste your exact path here between quotes
path = r"C:\Users\khush\Downloads\bricks.jpg"   # Windows example
# path = "/home/khushi/Downloads/bricks.jpg"     # Mac/Linux example

try:
    img = Image.open(path).convert("RGB")
    print("✅ Image opened successfully!")
    print("📷 Format:", img.format)
    print("📏 Size:", img.size)
    print("🎨 Mode:", img.mode)
except Exception as e:
    print("❌ Error loading image:", e)
