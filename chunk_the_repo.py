import os
from pathlib import Path

def load_repo(path):
    files = []
    for p in Path(path).rglob("*.py"):   # change *.py to *.go, *.js, etc.
        with open(p, "r", encoding="utf-8") as f:
            files.append((str(p), f.read()))
    return files

def chunk_code(filename, text, max_lines=150):
    lines = text.split("\n")
    for i in range(0, len(lines), max_lines):
        yield {
            "file": filename,
            "content": "\n".join(lines[i:i+max_lines])
        }
