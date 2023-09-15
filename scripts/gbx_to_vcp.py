import shutil
from pathlib import Path

import numpy as np
from pygbx import Gbx, GbxType

from trackmania_rl.geometry import extract_cp_distance_interval


def gbx_to_vcp(gbx_path: str, base_dir):
    gbx = Gbx(gbx_path)
    ghosts = gbx.get_classes_by_ids([GbxType.CTN_GHOST])
    assert len(ghosts) > 0, "The file does not contain any ghost."
    ghost = min(ghosts, key=lambda ghost: ghost.cp_times[-1])
    assert ghost.num_respawns == 0, "The ghost contains respawns"
    print(ghost.race_time, f"ghost has {len(ghost.records)} records and {len(ghost.control_entries)} control entries")
    records_to_keep = round(ghost.race_time / 100)
    print("Keeping", records_to_keep, "out of", len(ghost.records), "records for a race time of", ghost.race_time / 1000)
    raw_positions_list = []
    for r in ghost.records[:records_to_keep]:
        raw_positions_list.append(np.array([r.position.x, r.position.y, r.position.z]))
    _ = extract_cp_distance_interval(raw_positions_list, 0.5, base_dir)


base_dir = Path(__file__).resolve().parents[1]

"""
EXTRACT VCP FROM A SINGLE GBX
"""
gbx_to_vcp("Map5_pb4608(02'04''91).Replay.Gbx", base_dir)


"""
EXTRACT VCP FROM MULTIPLE GBX
"""
# replay_dir = Path(r"C:\Users\chopi\Documents\TrackMania\Tracks\Challenges\Official Maps\Green\replays")

# for replay_file in replay_dir.iterdir():
#     map_name = replay_file.name.split(".")[0]
#     # gbx_to_vcp(str(replay_file), base_dir)
#     # shutil.copyfile(base_dir / "maps" / "map.npy", base_dir / "maps" / f"{map_name}_0.5m_author.npy")

"""
LIST MAP NAMES
"""
# [replay_file.name.split('.')[0] for replay_file in replay_dir.iterdir()]


