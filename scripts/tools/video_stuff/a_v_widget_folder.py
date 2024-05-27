"""
Make the input widget overlaid on the map5 and Hockolicious videos.
"""
import subprocess
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from trackmania_rl import run_to_video

tmi_dir = Path("C:\\Users\\chopi\\Documents\\TMInterface\\Scripts")
base_dir = Path(__file__).resolve().parents[3]
temp_path_str = str(base_dir / "temp")

inputs = [
    {  # 0 Forward
        "left": False,
        "right": False,
        "accelerate": True,
        "brake": False,
    },
    {  # 1 Forward left
        "left": True,
        "right": False,
        "accelerate": True,
        "brake": False,
    },
    {  # 2 Forward right
        "left": False,
        "right": True,
        "accelerate": True,
        "brake": False,
    },
    {  # 3 Nothing
        "left": False,
        "right": False,
        "accelerate": False,
        "brake": False,
    },
    {  # 4 Nothing left
        "left": True,
        "right": False,
        "accelerate": False,
        "brake": False,
    },
    {  # 5 Nothing right
        "left": False,
        "right": True,
        "accelerate": False,
        "brake": False,
    },
    {  # 6 Brake
        "left": False,
        "right": False,
        "accelerate": False,
        "brake": True,
    },
    {  # 7 Brake left
        "left": True,
        "right": False,
        "accelerate": False,
        "brake": True,
    },
    {  # 8 Brake right
        "left": False,
        "right": True,
        "accelerate": False,
        "brake": True,
    },
    {  # 9 Brake and accelerate
        "left": False,
        "right": False,
        "accelerate": True,
        "brake": True,
    },
    {  # 10 Brake and accelerate left
        "left": True,
        "right": False,
        "accelerate": True,
        "brake": True,
    },
    {  # 11 Brake and accelerate right
        "left": False,
        "right": True,
        "accelerate": True,
        "brake": True,
    },
]


def rm_tree(pth: Path):
    for child in pth.iterdir():
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()


q_value_gap = 0.03
# with tempfile.TemporaryDirectory() as zou_dir:
rm_tree(Path(temp_path_str))
Path(temp_path_str).mkdir(parents=True, exist_ok=True)
temp_dir = Path(temp_path_str)
# Place the keys where they should be
key_reorder = [1, 0, 2, 4, 3, 5, 10, 9, 11, 7, 6, 8]


def plot_q_and_key(ax, q_values_one_frame, key_number_one_frame):
    ax.axis("off")
    ax.set_ylim([-0.05, 2.1])
    ax.set_xlim([-0.55, 2.55])

    alpha = np.clip(1 + (q_values_one_frame[key_number_one_frame] - q_values_one_frame.max()) / q_value_gap, 0, 1)

    if key_number_one_frame == 3:
        # Brake
        ax.bar(x=1, height=1, width=1.9, bottom=0, align="center", hatch="////", color=(1, 1, 1, 0.5), linewidth=0.5, edgecolor="black")

        # Brake
        ax.bar(
            x=1,
            height=alpha * ((key_number_one_frame == 3) | inputs[key_number_one_frame]["brake"]),
            width=1.9,
            bottom=0,
            align="center",
            color=(
                1 * (alpha < 1.0),
                1 * (alpha >= 1.0),
                0,
                alpha * ((key_number_one_frame == 3) | inputs[key_number_one_frame]["brake"]),
            ),
            linewidth=0.5,
            edgecolor="black",
        )

        ax.text(
            x=1,
            y=0.5,
            s="No\nKeypress",
            ha="center",
            va="center",
            # backgroundcolor='w',
            bbox=dict(boxstyle="square,pad=0.1", fc=(1, 1, 1, 0.7), ec="none"),
        )

    else:
        # Left
        ax.bar(
            x=0,
            height=1,
            width=0.9,
            bottom=0,
            align="center",
            hatch="////" if not inputs[key_number_one_frame]["left"] else "",
            color=(1, 1, 1, 0.5),
            linewidth=0.5,
            edgecolor="black",
        )

        # Brake
        ax.bar(
            x=1,
            height=1,
            width=0.9,
            bottom=0,
            align="center",
            hatch="////" if not inputs[key_number_one_frame]["brake"] else "",
            color=(1, 1, 1, 0.5),
            linewidth=0.5,
            edgecolor="black",
        )

        # Right
        ax.bar(
            x=2,
            height=1,
            width=0.9,
            bottom=0,
            align="center",
            hatch="////" if not inputs[key_number_one_frame]["right"] else "",
            color=(1, 1, 1, 0.5),
            linewidth=0.5,
            edgecolor="black",
        )

        # Accelerate
        ax.bar(
            x=1,
            height=1,
            width=0.9,
            bottom=1.1,
            align="center",
            hatch="////" if not inputs[key_number_one_frame]["accelerate"] else "",
            color=(1, 1, 1, 0.5),
            linewidth=0.5,
            edgecolor="black",
        )

        # Left
        ax.bar(
            x=0,
            height=alpha * ((key_number_one_frame == 3) | inputs[key_number_one_frame]["left"]),
            width=0.9,
            bottom=0,
            align="center",
            color=(
                1 * (alpha < 1.0),
                1 * (alpha >= 1.0),
                0,
                alpha * ((key_number_one_frame == 3) | inputs[key_number_one_frame]["left"]),
            ),
            linewidth=0.5,
            edgecolor="black",
        )

        # Brake
        ax.bar(
            x=1,
            height=alpha * ((key_number_one_frame == 3) | inputs[key_number_one_frame]["brake"]),
            width=0.9,
            bottom=0,
            align="center",
            color=(
                1 * (alpha < 1.0),
                1 * (alpha >= 1.0),
                0,
                alpha * ((key_number_one_frame == 3) | inputs[key_number_one_frame]["brake"]),
            ),
            linewidth=0.5,
            edgecolor="black",
        )

        # Right
        ax.bar(
            x=2,
            height=alpha * ((key_number_one_frame == 3) | inputs[key_number_one_frame]["right"]),
            width=0.9,
            bottom=0,
            align="center",
            color=(
                1 * (alpha < 1.0),
                1 * (alpha >= 1.0),
                0,
                alpha * ((key_number_one_frame == 3) | inputs[key_number_one_frame]["right"]),
            ),
            linewidth=0.5,
            edgecolor="black",
        )

        # Accelerate
        ax.bar(
            x=1,
            height=alpha * ((key_number_one_frame == 3) | inputs[key_number_one_frame]["accelerate"]),
            width=0.9,
            bottom=1.1,
            align="center",
            color=(
                1 * (alpha < 1.0),
                1 * (alpha >= 1.0),
                0,
                alpha * ((key_number_one_frame == 3) | inputs[key_number_one_frame]["accelerate"]),
            ),
            linewidth=0.5,
            edgecolor="black",
        )


def plot_v(ax, q_values_one_frame):
    ax.axis("off")
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlim([-0.55, 2.55])

    value = np.max(q_values_one_frame)
    vmin = -6.2  # -4.0765357
    vmax = -3.0  # 0.009419574

    # final run hocko
    vmin = -4.5  # -4.0765357
    vmax = -2.5  # 0.00941957

    # alpha = np.clip(1 + (q_values_one_frame[key_number_one_frame] - q_values_one_frame.max()) / q_value_gap, 0, 1)
    alpha = 0.5

    # Brake
    ax.bar(x=1, height=1, width=1, bottom=0, align="center", hatch="", color=(1, 1, 1, 0.5), linewidth=0.5, edgecolor="black")

    # Brake
    ax.bar(
        x=1,
        height=(value - vmin) / (vmax - vmin),
        width=1,
        bottom=0,
        align="center",
        color=(252 / 255, 186 / 255, 3 / 255, 0.4),
        # color=(
        #     1 * (alpha < 1.0),
        #     1 * (alpha >= 1.0),
        #     0,
        #     alpha * ((key_number_one_frame == 3) | inputs[key_number_one_frame]["brake"]),
        # ),
        linewidth=0.5,
        edgecolor="black",
    )


pc = time.perf_counter()

mosaic = [
    ["v", 0, 1, 2],
    ["v", 3, 4, 5],
    ["v", 6, 7, 8],
    ["v", 9, 10, 11],
]


run_dir = base_dir / "save" / "video2" / "runs"


for a in run_dir.iterdir():
    out_dir = tmi_dir / "video2"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = a.name

    if run_name != "1024":
        continue

    print(f"{run_name=}")
    run_to_video.write_actions_from_disk_in_tmi_format(
        infile_path=run_dir / run_name / "actions.joblib", outfile_path=out_dir / f"{run_name}.inputs"
    )

    q_values = joblib.load(run_dir / run_name / "q_values.joblib")
    video_path = Path(r"D:\video2\widgets") / f"{run_name}.mov"

    def foo(frame_id):
        print(f"{(1+frame_id) / len(q_values):.1%}", end="\r")
        # fig, axes = plt.subplots(nrows=len(q_values[0]) // 3, ncols=3)
        fig, axes = plt.subplot_mosaic(mosaic=mosaic, figsize=(9.5, 6))
        for key_number in range(len(q_values[0])):
            plot_q_and_key(axes[key_reorder[key_number]], q_values[frame_id], key_number)
        plot_v(axes["v"], q_values[frame_id])
        fig.suptitle(frame_id)
        plt.savefig(Path(temp_dir) / f"frame_{frame_id:09}.png", transparent=True, dpi=400)
        plt.close(fig)

    joblib.Parallel(n_jobs=6, verbose=0)(joblib.delayed(foo)(i) for i in range(len(q_values)))

    print(f"{(time.perf_counter() - pc):.2f} seconds for 100 frames.")

    print("")
    print("Individual frames done.")
    print("Begin encoding video.")

    subprocess.call(
        [
            "ffmpeg",
            "-framerate",
            "20",
            "-pattern_type",
            "sequence",
            "-start_number",
            "000000000",
            "-i",
            temp_path_str + "\\frame_%09d.png",
            "-pix_fmt",
            "yuva420p",
            "-vcodec",
            "png",
            str(video_path),
            "-y",
        ]
    )
    print("Encoding done, script finished.")
