"""
Python Glacier Evolution Model (PyGEM)

copyright © 2024 Brandon Tober <btober@cmu.edu> David Rounce <drounce@cmu.edu>

Distrubted under the MIT lisence
"""
import os
import shutil

def print_file_tree(start_path, indent=""):
    print(os.path.abspath(start_path))
    # Loop through all files and directories in the current directory
    for item in os.listdir(start_path):
        path = os.path.join(start_path, item)
        
        # Print the current item with indentation
        print(indent + "|-- " + item)
        
        # Recursively call this function if the item is a directory
        if os.path.isdir(path):
            print_file_tree(path, indent + "    ")

def copy_analyses(dest_dir,src_dir):
    """Check if the config file exists, and copy it if not."""
    os.makedirs(src_dir, exist_ok=True)  # Ensure the base directory exists
    try:
        shutil.copytree(src_dir, dest_dir)  # Copy the file
        print(f"Copied example PyGEM notebooks:")
        print_file_tree(dest_dir)
    except FileExistsError:
        print(f'Failed to copy PyGEM example notebooks, directory already exists: {dest_dir}')
    return

def main():
    # Define the base directory and the path to the analyses
    basedir = os.path.join(os.path.expanduser('~'), 'PyGEM')
    dest_analyses_dir = os.path.abspath(os.path.join(basedir, 'example_notebooks/'))

    package_dir = os.path.dirname(__file__)  # Get the directory of the current script
    source_analyses_dir = os.path.abspath(os.path.join(package_dir, '../../../../example_notebooks'))  # Path to copy notebooks
    copy_analyses(dest_analyses_dir,source_analyses_dir)

if __name__ == "__main__":
    main()