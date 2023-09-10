import numpy as np
from pathlib import Path
from scipy.interpolate import make_interp_spline
from pygbx import Gbx, GbxType
from trackmania_rl.geometry import extract_cp_distance_interval

gbx = Gbx('CoolerMaster2_$o$i$a0aK$a06a$a30ck$a60iest Kack$a00$a0ay $f40#–35.Replay.gbx')
ghost = gbx.get_class_by_id(GbxType.CTN_GHOST)
if not ghost:
    exit()
if ghost.num_respawns>0:
	print("WARNING: GHOST HAS RESPAWNS")

print("Ghost has",len(ghost.records),"records and",len(ghost.control_entries),"control entries")
raw_positions_list = []
for r in ghost.records:
	raw_positions_list.append(np.array([r.position.x, r.position.y, r.position.z]))
interpolation_function = make_interp_spline(x=range(len(raw_positions_list)),y=raw_positions_list,k=1)
raw_positions_list_interpolated = interpolation_function(np.arange(0,len(raw_positions_list),0.01))
base_dir = Path(__file__).resolve().parents[1]
_ = extract_cp_distance_interval(raw_positions_list_interpolated, 0.5, base_dir)
