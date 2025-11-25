import os
import easyocr

# Path to your dataset root
dataset_root = r'C:\Users\satis\OneDrive\Documents\ANPR_YOLO_EasyOCR\dataset'
output_file = 'dataset.txt'

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Valid image extensions
valid_exts = ['.jpg', '.jpeg', '.png']

# Open output file for writing
with open(output_file, 'w') as f:
    for split in ['train', 'val', 'test']:
        folder = os.path.join(dataset_root, split)
        if not os.path.exists(folder):
            print(f"Skipping missing folder: {folder}")
            continue

        for img_name in os.listdir(folder):
            ext = os.path.splitext(img_name)[1].lower()
            if ext in valid_exts:
                img_path = os.path.join(folder, img_name)
                try:
                    results = reader.readtext(img_path)
                    if results:
                        # Sort by OCR confidence and take the top result
                        results.sort(key=lambda x: x[2], reverse=True)
                        plate_number = results[0][1].strip()
                        print(f"{img_name}: {plate_number}")
                        f.write(plate_number + '\n')
                    else:
                        print(f"{img_name}: No text found.")
                except Exception as e:
                    print(f"Error reading {img_path}: {e}")

print(f"\n✅ All plate numbers saved to `{output_file}`.")
