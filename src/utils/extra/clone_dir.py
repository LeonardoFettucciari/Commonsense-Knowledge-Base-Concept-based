import os

def clone_empty_folder_structure(src_dir, dst_dir):
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    
    for root, dirs, files in os.walk(src_dir):
        # Construct destination path
        relative_path = os.path.relpath(root, src_dir)
        target_root = os.path.join(dst_dir, relative_path)

        # Create directories in destination
        os.makedirs(target_root, exist_ok=True)

        # Create empty files in destination
        for file in files:
            empty_file_path = os.path.join(target_root, file)
            with open(empty_file_path, 'w') as f:
                pass  # create an empty file

    print(f"Cloned folder structure from '{src_dir}' to '{dst_dir}' with empty files.")

# Example usage:
if __name__ == "__main__":
    source = "outputs"      
    destination = "download/outputs"
    clone_empty_folder_structure(source, destination)
