import os
from tkinter import filedialog

def delete_files_to_count(folder_path, target_count):
    """
    Deletes files in the specified folder until the number of files matches the target count.

    :param folder_path: Path to the folder where files will be deleted.
    :param target_count: The target number of files to keep in the folder.
    """
    try:
        # Get the list of files in the folder
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        
        # Check if the target count is greater than the current number of files
        if len(files) <= target_count:
            print(f"Folder already has {len(files)} files or less. No files to delete.")
            return
        
        # Calculate how many files need to be deleted
        files_to_delete = len(files) - target_count
        
        # Delete the necessary number of files
        for i in range(files_to_delete):
            file_to_delete = os.path.join(folder_path, files[i])
            os.remove(file_to_delete)
            print(f"Deleted: {file_to_delete}")
        
        print(f"Deleted {files_to_delete} files. {target_count} files remain in the folder.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

folder_path = filedialog.askdirectory(initialdir=os.path.join("", ""))
if folder_path:
    target_count = int(input("Target count: "))
    delete_files_to_count(folder_path, target_count)