"""
From a list of replays, create a list of "medal_type": "time" pairs.
Used to fill in trackmania_rl/map_reference_times.py
"""

from pathlib import Path

from pygbx import Gbx, GbxType


def gbx_to_vcp(gbx_path: str, base_dir):
    gbx = Gbx(gbx_path)
    ghosts = gbx.get_classes_by_ids([GbxType.CTN_GHOST])
    assert len(ghosts) > 0, "The file does not contain any ghost."
    for medal_name, ghost in zip(["author", "gold", "silver", "bronze"], ghosts):
        print(f'"{medal_name}": {ghost.cp_times[-1] / 1000 : .2f}, ')


base_dir = Path(__file__).resolve().parents[2]


"""
EXTRACT TIMES FROM MULTIPLE GBX
"""
replay_dir = Path(r"C:\Users\chopi\Documents\TrackMania\Tracks\Challenges\Official Maps\White\replays")
for replay_file in replay_dir.iterdir():
    map_name = replay_file.name.split(".")[0][:3]
    print(f'"{map_name}": {{')
    gbx_to_vcp(str(replay_file), base_dir)
    print("},")
replay_dir = Path(r"C:\Users\chopi\Documents\TrackMania\Tracks\Challenges\Official Maps\Green\replays")
for replay_file in replay_dir.iterdir():
    map_name = replay_file.name.split(".")[0][:3]
    print(f'"{map_name}": {{')
    gbx_to_vcp(str(replay_file), base_dir)
    print("},")
