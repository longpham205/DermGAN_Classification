import os

# Các thư mục / file loại trừ theo TÊN
EXCLUDE_NAMES = {
    "venv",
    ".git",
    "__pycache__",
    ".idea",
    ".vscode",
    "node_modules",
    # "data",
    #"Tests",
    "tempCodeRunnerFile.py",
}

# Các đuôi file cần ẩn
EXCLUDE_EXTENSIONS = {
    ".jpg", 
    ".png"
}

def print_tree(root_path, prefix=""):
    try:
        items = sorted(os.listdir(root_path))
    except PermissionError:
        return

    filtered_items = []

    for item in items:
        path = os.path.join(root_path, item)

        # 1. Loại theo tên
        if item in EXCLUDE_NAMES:
            continue

        # 2. Loại theo đuôi file
        if os.path.isfile(path):
            _, ext = os.path.splitext(item)
            if ext.lower() in EXCLUDE_EXTENSIONS:
                continue

        filtered_items.append(item)

    for index, item in enumerate(filtered_items):
        path = os.path.join(root_path, item)
        connector = "└── " if index == len(filtered_items) - 1 else "├── "
        print(prefix + connector + item)

        if os.path.isdir(path):
            extension = "    " if index == len(filtered_items) - 1 else "│   "
            print_tree(path, prefix + extension)


if __name__ == "__main__":
    PROJECT_ROOT = r"g:\VsCode\Python\NCKH\DermGAN_Classification"
    print_tree(PROJECT_ROOT)
