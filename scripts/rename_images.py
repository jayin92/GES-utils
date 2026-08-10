import argparse
import os
import re
import sys
from pathlib import Path

# GES exports frames as <project_name>_<index>.jpeg
SOURCE_SUFFIXES = (".jpg", ".jpeg")
FRAME_PATTERN = re.compile(r"^.*_(\d+)$")


def plan_renames(folder, max_index=None):
    """
    Build the list of (source, destination) renames for a folder of GES exports.

    Returns (planned, conflicts). `planned` is a list of (Path, Path) pairs;
    `conflicts` is a list of human-readable strings describing collisions.
    """
    candidates = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SOURCE_SUFFIXES
    )

    planned = []
    for path in candidates:
        match = FRAME_PATTERN.match(path.stem)
        if not match:
            print(f"Skipped (no frame index): {path.name}")
            continue

        frame_number = int(match.group(1))

        if max_index is not None and frame_number > max_index:
            print(f"Skipped (index > {max_index}): {path.name}")
            continue

        # ges2colmap.py writes file_path as images/frame_{i:04}.jpg
        destination = folder / f"frame_{frame_number:04d}.jpg"

        if destination == path:
            print(f"Skipped (already named): {path.name}")
            continue

        planned.append((path, destination))

    # Detect collisions before touching anything: os.rename overwrites silently.
    conflicts = []
    claimed = {}
    sources = {src for src, _ in planned}
    for source, destination in planned:
        if destination in claimed:
            conflicts.append(
                f"{claimed[destination].name} and {source.name} both map to {destination.name}"
            )
        claimed[destination] = source
        if destination.exists() and destination not in sources:
            conflicts.append(
                f"{source.name} would overwrite existing {destination.name}"
            )

    return planned, conflicts


def rename_images(folder_path, dry_run=False, max_index=None):
    """
    Rename GES exports from <name>_00XX.jpeg to frame_00XX.jpg.

    Args:
        folder_path: Folder containing the images.
        dry_run: Print the planned renames without touching the filesystem.
        max_index: Skip frames with an index above this value (default: no limit).
    """
    folder = Path(folder_path).expanduser().resolve()
    if not folder.is_dir():
        print(f"Error: not a directory: {folder}")
        return 1

    planned, conflicts = plan_renames(folder, max_index)

    if conflicts:
        print("\nAborting, no files were renamed. Conflicts:")
        for conflict in conflicts:
            print(f"  - {conflict}")
        return 1

    if not planned:
        print("\nNothing to rename.")
        return 0

    for source, destination in planned:
        if dry_run:
            print(f"Would rename: {source.name} -> {destination.name}")
        else:
            os.rename(source, destination)
            print(f"Renamed: {source.name} -> {destination.name}")

    verb = "would be renamed" if dry_run else "renamed"
    print(f"\nTotal files {verb}: {len(planned)}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rename GES exports from <name>_00XX.jpeg to frame_00XX.jpg"
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="Folder containing the images (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the planned renames without modifying any files",
    )
    parser.add_argument(
        "--max-index",
        type=int,
        default=None,
        help="Skip frames with an index above this value (default: no limit)",
    )

    args = parser.parse_args()
    sys.exit(rename_images(args.folder, args.dry_run, args.max_index))
