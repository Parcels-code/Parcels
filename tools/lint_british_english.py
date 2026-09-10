"""Lint tutorial Markdown and notebook prose with Vale."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def extract_markdown(notebook_path: Path, output_path: Path) -> None:
    notebook = json.loads(notebook_path.read_text())
    markdown_cells = ("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
    output_path.write_text("\n\n".join(markdown_cells))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    arguments = parser.parse_args()

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        vale_paths = []
        for path in arguments.paths:
            if path.suffix == ".ipynb":
                extracted_path = temporary_path / path.with_suffix(".md").name
                extract_markdown(path, extracted_path)
                vale_paths.append(extracted_path)
            else:
                vale_paths.append(path)

        result = subprocess.run(["vale", *map(str, vale_paths)], check=False)

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
