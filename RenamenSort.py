import os

def auto_rename_to_jpeg(folder_path):
    # Daftar ekstensi gambar yang akan diubah
    image_extensions = ['.avif', '.png', '.jpg', '.bmp', '.gif', '.tiff', '.jpeg', '.webp']
    
    # Mengumpulkan file dan ukuran mereka
    files_with_sizes = []
    for filename in os.listdir(folder_path):
        # Dapatkan nama file dan ekstensi
        name, ext = os.path.splitext(filename)
        
        if ext.lower() in image_extensions:
            # Buat path penuh untuk file
            full_path = os.path.join(folder_path, filename)
            # Dapatkan ukuran file
            size = os.path.getsize(full_path)
            # Tambahkan tuple (size, filename) ke list
            files_with_sizes.append((size, filename))
    
    # Urutkan berdasarkan size
    files_with_sizes.sort()

    counter = 1
    # Iterasi berdasarkan list file yang sudah diurutkan
    for _, filename in files_with_sizes:
        # Tentukan nama baru dengan ekstensi .jpeg
        new_name = f"Stair-Climber_{counter}.jpeg"
        counter += 1
        
        # Buat path penuh untuk file lama dan baru
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        
        # Ganti nama file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")

# Path folder yang berisi gambar
folder_path = "Gym Equipment Output\Stair Climber"

# Jalankan fungsi
auto_rename_to_jpeg(folder_path)
