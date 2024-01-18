import os
import re

BASE_DOCS_DIR = "docs"

def get_all_docs_files():
    all_docs_files = []
    for root, dirs, files in os.walk(BASE_DOCS_DIR):
        for file in files:
            if file.endswith(".md"):
                all_docs_files.append(os.path.join(root, file))
    return all_docs_files

def markdown_to_html(md_string):
    # This regex pattern matches only absolute URLs not preceded by an exclamation mark
    pattern = r'(?<!\!)\[([^\]]+)\]\((https?://[^)]+|ftp://[^)]+)\)'
    html_string = re.sub(pattern, r'<a href="\2" target="_blank">\1</a>', md_string)
    return html_string

def main():
    all_docs_files = get_all_docs_files()
    for doc in all_docs_files:
        with open(doc, "r") as f:
            lines = f.readlines()
        with open(doc, "w") as f:
            for line in lines:
                line = markdown_to_html(line)
                f.write(line)

if __name__ == "__main__":
    main()