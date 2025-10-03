import os
import shutil

def delete_folders_by_name(parent_folder, target_folder_name):
    for dirpath, dirnames, filenames in os.walk(parent_folder, topdown=False):
        for dirname in dirnames:
            if dirname == target_folder_name:
                folder_to_delete = os.path.join(dirpath, dirname)
                try:
                    shutil.rmtree(folder_to_delete)
                    print(f"Deleted: {folder_to_delete}")
                except Exception as e:
                    print(f"Failed to delete {folder_to_delete}: {e}")

if __name__ == "__main__":
    parent = "outputs/inference" 
    folder_name_to_delete = ""

    delete_folders_by_name(parent, folder_name_to_delete)
