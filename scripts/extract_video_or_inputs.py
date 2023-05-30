from pathlib import Path

from trackmania_rl import run_to_video

base_dir = Path(__file__).resolve().parents[1]
tmi_dir = Path("C:\\Users\\chopi\\Documents\\TMInterface\\Scripts")

run_dir = base_dir / "save" / "58" / "best_runs"
out_dir = tmi_dir / "map5_b_58"

out_dir.mkdir(parents=True, exist_ok=True)

for a in run_dir.iterdir():
    time_ms = a.name
    run_to_video.write_actions_from_disk_in_TMI_format(infile_path=run_dir / time_ms / "actions.joblib", outfile_path=out_dir / f"{time_ms}.inputs")


# run_to_video.make_wdiget_video_from_q_values_on_disk(
#     q_values_path=run_dir / "127000" / "q_values.joblib", video_path=base_dir / "127000.mp4"
# )
