import os
import shutil
import re

def organize_files(folder_path):
    files = os.listdir(folder_path)
    base_names = {}
    
    # Regular expression to detect names ending with 'dshow'
    pattern = re.compile(r'^(.*dshow).*\.(mp4|csv|h5|pickle)$')
    
    for file in files:
        match = pattern.match(file)
        if match:
            base_name = match.group(1)
            if base_name not in base_names:
                base_names[base_name] = []
            base_names[base_name].append(file)
    
    # Create folders and move files
    for name, group_files in base_names.items():
        destination_folder = os.path.join(folder_path, name)
        os.makedirs(destination_folder, exist_ok=True)
        
        for file in group_files:
            file_path = os.path.join(folder_path, file)
            if os.path.exists(file_path):
                shutil.move(file_path, os.path.join(destination_folder, file))

if __name__ == "__main__":
    folder_path = "C:/TFG_CISC/Videos/NOL_P150"  
    organize_files(folder_path)