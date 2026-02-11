"""Get a Parcels v4 alpha version string based on commits since the last tag.

Sets the PARCELS_ALPHA_VERSION environment variable (via GITHUB_ENV in CI,
or prints the version to stdout for local use).
"""

import subprocess


def get_alpha_version():
    result = subprocess.run(
        ["git", "describe", "--tags", "--long"],
        capture_output=True,
        text=True,
        check=True,
    )
    # Format: <tag>-<n>-g<hash>
    # e.g. v3.1.2-1967-g1857d5219
    parts = result.stdout.strip().rsplit("-", 2)
    n_commits = parts[1]
    return f"4.0.0alpha{n_commits}"


def main():
    version = get_alpha_version()
    print(f"PARCELS_ALPHA_VERSION={version}")


if __name__ == "__main__":
    main()
