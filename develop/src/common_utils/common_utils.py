import os

def make_dirs(dirs):
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)