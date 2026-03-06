#!/usr/bin/env python
"""
Script to generate diagrams from source files using Kroki.

This script reads configuration from config.toml and processes diagram files
using the kroki CLI tool to generate output images.
"""

import subprocess
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

try:
    import tomllib
except ImportError:
    import tomli as tomllib


# Supported diagram types in Kroki
DiagramType = Literal[ "actdiag", "blockdiag", "bpmn", "bytefield", "c4plantuml", "d2", "diagramsnet", "ditaa", "erd", "excalidraw", "graphviz", "mermaid", "nomnoml", "nwdiag", "packetdiag", "pikchr", "plantuml", "rackdiag", "seqdiag", "structurizr", "svgbob", "umlet", "vega", "vegalite", "wavedrom"]  # fmt: skip

# Supported output formats
OutputFormat = Literal["base64", "jpeg", "pdf", "png", "svg"]


class DiagramConfig(BaseModel):
    """Configuration for a single diagram."""

    file_src: str = Field(
        ..., description="Path to the source diagram file, relative to the script"
    )
    output_name: str | None = Field(
        None, description="Output filename without extension (defaults to input stem)"
    )
    format: OutputFormat = Field("svg", description="Output format")
    type: DiagramType | None = Field(
        None, description="Diagram type (defaults to infer from file extension)"
    )

    @field_validator("file_src")
    @classmethod
    def validate_file_src(cls, v: str) -> str:
        """Ensure file_src is not empty."""
        if not v or not v.strip():
            raise ValueError("file_src cannot be empty")
        return v


class Config(BaseModel):
    """Root configuration model."""

    diagram: list[DiagramConfig] = Field(
        default_factory=list, description="List of diagrams to generate"
    )


def main():
    """Main function to process diagrams based on config.toml."""
    script_dir = Path(__file__).parent
    config_file = script_dir / "config.toml"
    build_dir = script_dir / "build"

    # Ensure build directory exists
    build_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    try:
        with open(config_file, "rb") as f:
            config_data = tomllib.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading configuration file: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate configuration with Pydantic
    try:
        config = Config(**config_data)
    except Exception as e:
        print(f"Error: Invalid configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Check if there are any diagrams configured
    if not config.diagram:
        print("No diagrams configured in config.toml")
        return

    print(f"Processing {len(config.diagram)} diagram(s)...")

    # Process each diagram
    for i, diagram in enumerate(config.diagram, 1):
        # Get input file path (relative to script directory)
        input_file = script_dir / diagram.file_src
        if not input_file.exists():
            print(
                f"Warning: Source file not found: {input_file}, skipping",
                file=sys.stderr,
            )
            continue

        # Get output configuration
        output_name = diagram.output_name or input_file.stem
        output_format = diagram.format
        output_file = build_dir / f"{output_name}.{output_format}"

        # Detect diagram type from file extension or use explicit type
        if diagram.type:
            diagram_type = diagram.type
        else:
            # Infer from file extension
            diagram_type = input_file.suffix.lstrip(".")
            if diagram_type == "excalidraw":
                diagram_type = "excalidraw"

        # Build kroki command
        cmd = [
            "kroki",
            "convert",
            str(input_file),
            "--type",
            diagram_type,
            "--format",
            output_format,
            "--out-file",
            str(output_file),
        ]

        # Execute kroki command
        print(
            f"  [{i}/{len(config.diagram)}] Processing {diagram.file_src} -> {output_file.name}"
        )
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if result.stdout:
                print(f"    {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(
                f"    Error processing {diagram.file_src}: {e.stderr.strip()}",
                file=sys.stderr,
            )
            continue
        except FileNotFoundError:
            print(
                "Error: 'kroki' command not found. Please ensure kroki is installed.",
                file=sys.stderr,
            )
            sys.exit(1)

    print("Done!")


if __name__ == "__main__":
    main()
