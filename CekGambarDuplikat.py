import os
import hashlib
from PIL import Image

def file_hash(filepath):
    """Menghitung MD5 hash dari isi file."""
    with open(filepath, 'rb') as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()

def check_duplicate_images(folder_path):
    # Dictionary untuk menyimpan hash dan nama file
    hashes = {}
    duplicates = []

    # Iterasi melalui semua file di folder
    for filename in os.listdir(folder_path):
        # Dapatkan path penuh file
        file_path = os.path.join(folder_path, filename)
        
        # Buka dan hitung hash gambar
        if os.path.isfile(file_path): # Pastikan itu file, bukan folder
            try:
                # Menghitung hash
                img_hash = file_hash(file_path)
                
                # Cek apakah hash sudah ada di dictionary
                if img_hash in hashes:
                    duplicates.append((filename, hashes[img_hash]))
                else:
                    hashes[img_hash] = filename
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    
    # Cetak file yang duplicate
    if duplicates:
        print("Found duplicates:")
        for dup in duplicates:
            print(f"{dup[0]} is a duplicate of {dup[1]}")
    else:
        print("No duplicate files found.")

# Path folder yang berisi gambar
folder_path = "Gym Equipment Output\Stair Climber"

# Jalankan fungsi
check_duplicate_images(folder_path)
