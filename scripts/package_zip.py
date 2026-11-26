"""
Creates a clean, portable ZIP archive of the Softmax_Kernel_Fusion project.
Strictly excludes nested zip files, temporary build artifacts, and cache files.
Output: C:\\Users\\ariha\\Desktop\\PGC LAB\\Softmax_Kernel_Fusion.zip
"""
import os
import zipfile

def main():
    proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parent_dir = os.path.abspath(os.path.join(proj_dir, ".."))
    out_zip = os.path.join(parent_dir, "Softmax_Kernel_Fusion.zip")

    # If previous zip exists at destination, delete it
    if os.path.exists(out_zip):
        try:
            os.remove(out_zip)
            print(f"Removed old archive: {out_zip}")
        except Exception as e:
            print(f"Warning: could not delete old zip: {e}")

    exclude_dirs = {".git", "__pycache__", ".vs", "temp_html"}
    exclude_extensions = {".zip", ".pyc", ".tmp"}

    print(f"Packaging {proj_dir} into {out_zip}...")
    file_count = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, files in os.walk(proj_dir):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in exclude_extensions:
                    print(f"Skipping excluded file: {file}")
                    continue

                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, parent_dir)

                zf.write(full_path, rel_path)
                file_count += 1
                total_uncompressed_bytes += os.path.getsize(full_path)

    zip_size_mb = os.path.getsize(out_zip) / (1024 * 1024)
    orig_size_mb = total_uncompressed_bytes / (1024 * 1024)
    print(f"\n[SUCCESS] Archive generated successfully!")
    print(f"Archive Path : {out_zip}")
    print(f"Files Stored : {file_count}")
    print(f"Original Size: {orig_size_mb:.2f} MB")
    print(f"Archive Size : {zip_size_mb:.2f} MB")

if __name__ == "__main__":
    main()
