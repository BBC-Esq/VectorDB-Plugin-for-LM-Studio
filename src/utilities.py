import os

def validate_symbolic_links(source_directory):
    # Checks symbolic links in the docs for db folder to make sure they're valid, and removes the ones that aren't
    symbolic_links = [entry for entry in os.listdir(source_directory) if os.path.islink(os.path.join(source_directory, entry))]

    for symlink in symbolic_links:
        symlink_path = os.path.join(source_directory, symlink)
        target_path = os.readlink(symlink_path)

        if not os.path.exists(target_path):
            print(f"Warning: Symbolic link {symlink} points to a missing file. It will be skipped.")
            os.remove(symlink_path)

if __name__ == '__main__':
    source_directory = "Docs_for_DB"
    validate_symbolic_links(source_directory)