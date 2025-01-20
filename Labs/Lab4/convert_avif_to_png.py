from PIL import Image
import os
import glob

input_folder = "images"  # Schimbă dacă ai alt folder
output_folder = "converted_images"

os.makedirs(output_folder, exist_ok=True)

# Găsește toate fișierele AVIF în folder
for avif_file in glob.glob(os.path.join(input_folder, "*.avif")):
    img = Image.open(avif_file)
    jpg_filename = os.path.join(output_folder, os.path.basename(avif_file).replace(".avif", ".jpg"))
    img.convert("RGB").save(jpg_filename, "JPEG")
    print(f"Converted: {avif_file} -> {jpg_filename}")

print("Conversie completă! Imaginile sunt salvate în:", output_folder)