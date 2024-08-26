"""
This script reads a .gbx file, and creates a list of Virtual CheckPoints (VCP) based on the best ghost found in that file.

The .gbx file may either be a .challenge.gbx, or a .replay.gbx.

The VCP file is saved in base_dir/maps/map.npy. It should typically be renamed manually after running this script.

The distance between virtual checkpoints is currently 50cm (hardcoded).
"""

import argparse
from pathlib import Path

from trackmania_rl.geometry import extract_cp_distance_interval
from trackmania_rl.map_loader import gbx_to_raw_pos_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gbx_path", type=Path)
    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parents[2]

    raw_positions_list = gbx_to_raw_pos_list(args.gbx_path)
    _ = extract_cp_distance_interval(raw_positions_list, 0.5, base_dir)


if __name__ == "__main__":
    main()
