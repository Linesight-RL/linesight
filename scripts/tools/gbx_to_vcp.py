"""
This script reads a .gbx file, and creates a list of Virtual CheckPoints (VCP) based on the best ghost found in that file.

The .gbx file may either be a .challenge.gbx, or a .replay.gbx.

The VCP file is saved in base_dir/maps/map.npy. It should typically be renamed manually after running this script.

The distance between virtual checkpoints is currently 50cm (hardcoded).
"""
import argparse
from pathlib import Path

import numpy as np
from pygbx import Gbx, GbxType

from trackmania_rl.geometry import extract_cp_distance_interval


def gbx_to_raw_pos_list(gbx_path: Path):
    """
    Read a .gbx file, extract the raw positions of the best ghost included in that file.
    """
    gbx = Gbx(str(gbx_path))
    ghosts = gbx.get_classes_by_ids([GbxType.CTN_GHOST])
    assert len(ghosts) > 0, "The file does not contain any ghost."
    ghost = min(ghosts, key=lambda g: g.cp_times[-1])
    assert ghost.num_respawns == 0, "The ghost contains respawns"
    records_to_keep = round(ghost.race_time / 100)

    print(ghost.race_time, f"ghost has {len(ghost.records)} records and {len(ghost.control_entries)} control entries")
    print("Keeping", records_to_keep, "out of", len(ghost.records), "records for a race time of", ghost.race_time / 1000)

    raw_positions_list = []
    for r in ghost.records[:records_to_keep]:
        raw_positions_list.append(np.array([r.position.x, r.position.y, r.position.z]))

    return raw_positions_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gbx_path", type=Path)
    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parents[2]

    raw_positions_list = gbx_to_raw_pos_list(args.gbx_path)
    _ = extract_cp_distance_interval(raw_positions_list, 0.5, base_dir)


if __name__ == "__main__":
    main()
