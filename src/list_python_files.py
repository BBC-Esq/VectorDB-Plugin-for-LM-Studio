import os

def list_python_files_in_markdown():
    # Get list of all files in the current directory
    all_files = os.listdir()
    
    # Filter out only Python files
    python_files = [file for file in all_files if file.endswith('.py')]
    
    # Open a Markdown file to write the list
    with open("PythonFilesList.md", "w") as md_file:
        md_file.write("# List of Python Files\n")
        
        # Write each Python file name as a list item in Markdown
        for file in python_files:
            md_file.write(f"- {file}\n")

if __name__ == "__main__":
    list_python_files_in_markdown()
