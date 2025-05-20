import os


class FolderTreeGenerator:
    def __init__(self, root_path):
        self.tree = []
        self.file_paths = []
        self.root_path = os.path.abspath(root_path)
        self._generate(self.root_path)

    def _generate(self, directory, prefix='', is_root=True):
        if is_root:
            root_name = os.path.basename(self.root_path) or os.path.basename(os.path.dirname(self.root_path))
            self.tree.append(f"{root_name}/")
            is_root = False

        try:
            entries = os.listdir(directory)
        except PermissionError:
            self.tree.append(f"{prefix}└── [Permission denied]")
            return
        except Exception as e:
            self.tree.append(f"{prefix}└── [Error: {e}]")
            return

        # Separate directories and files, ignoring symlinks for directories, .git, and __pycache__
        dirs, files = [], []
        for entry in entries:
            if entry in [".git", "__pycache__", "models", "haarcascade_frontalface_default.xml", ".mp4"]:  # Skip the .git and __pycache__ folders
                continue
            full_path = os.path.join(directory, entry)
            if os.path.isdir(full_path) and not os.path.islink(full_path):
                dirs.append(entry)
            else:
                files.append(entry)

        # Sort directories and files case-insensitively
        sorted_entries = sorted(dirs, key=lambda s: s.lower()) + sorted(files, key=lambda s: s.lower())
        entries_count = len(sorted_entries)

        for index, entry in enumerate(sorted_entries):
            full_path = os.path.join(directory, entry)
            is_last = index == entries_count - 1
            connector = "└── " if is_last else "├── "
            next_prefix = "    " if is_last else "│   "

            if os.path.isdir(full_path) and not os.path.islink(full_path):
                self.tree.append(f"{prefix}{connector}{entry}/")
                self._generate(full_path, prefix + next_prefix, is_root=False)
            else:
                self.tree.append(f"{prefix}{connector}{entry}")
                if not os.path.islink(full_path):
                    rel_path = os.path.relpath(full_path, self.root_path)
                    self.file_paths.append(rel_path)


def read_file_contents(root_path, file_paths):
    contents = []
    for rel_path in file_paths:
        full_path = os.path.join(root_path, rel_path)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            contents.append(f"\n\n=== {rel_path} ===\n{file_content}")
        except UnicodeDecodeError:
            contents.append(f"\n\n=== {rel_path} ===\n[Binary file content omitted]")
        except Exception as e:
            contents.append(f"\n\n=== {rel_path} ===\n[Error reading file: {str(e)}]")
    return ''.join(contents)


def generate_folder_report(start_path, output_file):
    generator = FolderTreeGenerator(start_path)

    # Generate folder structure
    tree = "\n".join(generator.tree)

    # Generate file contents
    contents = read_file_contents(generator.root_path, generator.file_paths)

    # Combine both parts
    report = f"""FOLDER STRUCTURE:
{tree}

FILE CONTENTS:
{contents}"""

    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)


if __name__ == "__main__":
    target_dir = None
    if not target_dir:
        target_dir = os.getcwd()

    output_path = "pai-prompt-codebase.txt"
    generate_folder_report(target_dir, output_path)
    print(f"Report generated successfully: {output_path}")