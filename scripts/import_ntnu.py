"""Import a locally downloaded DataverseNO ZIP into ignored raw storage."""

import argparse
from pathlib import Path

from research_graph.sources.ntnu_loader import import_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--destination", type=Path, default=Path("data/raw/ntnu"))
    args = parser.parse_args()
    records = import_dataset(args.archive, args.destination)
    print(f"Imported {len(records)} files into {args.destination}")


if __name__ == "__main__":
    main()
