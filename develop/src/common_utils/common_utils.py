import os


def make_dirs(dirs):
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)


def load_text(path):
    with open(path, "r") as f:
        text = f.read().splitlines()

    return text
