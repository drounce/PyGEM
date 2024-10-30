"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distrubted under the MIT lisence
"""
import os
import shutil


def copy_analyses(dest_dir,src_dir):
    """Check if the config file exists, and copy it if not."""
    os.makedirs(src_dir, exist_ok=True)  # Ensure the base directory exists
    shutil.copytree(src_dir, dest_dir)  # Copy the file
    print(f"Copied analysis notebooks: {src_dir}\n{os.listdir(src_dir)}")
    return

def main():
    # Define the base directory and the path to the analyses
    basedir = os.path.join(os.path.expanduser('~'), 'PyGEM')
    dest_analyses_dir = os.path.join(basedir, '/analyses/')

    package_dir = os.path.dirname(__file__)  # Get the directory of the current script
    source_analyses_dir = os.path.join(package_dir, '/analyses/')  # Path to copy notebooks
    copy_analyses(dest_analyses_dir,source_analyses_dir)

if __name__ == "__main__":
    main()