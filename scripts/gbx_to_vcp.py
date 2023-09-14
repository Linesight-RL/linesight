from pathlib import Path

import numpy as np
from pygbx import Gbx, GbxType

from trackmania_rl.geometry import extract_cp_distance_interval

gbx = Gbx("A01-Race_eddieman194(00'23''79).Replay.gbx")
ghost = gbx.get_class_by_id(GbxType.CTN_GHOST)
if not ghost:
    exit()
if ghost.num_respawns > 0:
    print("WARNING: GHOST HAS RESPAWNS")

print(ghost.race_time,f"ghost has {len(ghost.records)} records and {len(ghost.control_entries)} control entries")
records_to_keep = round(ghost.race_time/100)
print("Keeping",records_to_keep,"out of",len(ghost.records),"records for a race time of",ghost.race_time/1000)
raw_positions_list = []
for r in ghost.records[:records_to_keep]:
    raw_positions_list.append(np.array([r.position.x, r.position.y, r.position.z]))
base_dir = Path(__file__).resolve().parents[1]
_ = extract_cp_distance_interval(raw_positions_list, 0.5, base_dir)
