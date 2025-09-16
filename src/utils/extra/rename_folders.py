import os

def rename_folders(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            if dirname == "untrained_retriever":
                old_path = os.path.join(dirpath, dirname)
                new_path = os.path.join(dirpath, "untrained_retriever_zscotk5_retriever")
                
                print(f"\nFound folder: {old_path}")
                response = input("Rename to '{}'? (yes/no): ".format(new_path)).strip().lower()
                
                if response == "yes":
                    if not os.path.exists(new_path):
                        os.rename(old_path, new_path)
                        print(f"Renamed: {old_path} -> {new_path}")
                    else:
                        print(f"Skipped: target folder already exists: {new_path}")
                else:
                    print(f"Skipped: {old_path}")

if __name__ == "__main__":
    input_directory = input("Enter the input directory path: ").strip()
    if os.path.isdir(input_directory):
        rename_folders(input_directory)
    else:
        print("Invalid directory path.")
