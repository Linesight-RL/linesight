import argparse
import math
import os
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas
from matplotlib.ticker import StrMethodFormatter

# Parameters
target_times = []  # Racing times at which to draw a horizontal line
target_times_labels = []  # Leave empty for default labels (time)
duration = 20  # seconds
y_axis_percentile = 80  # Only show up to this percentile of values
# Axis_Hour_Range = 0.2 #Percentage of total training time
DPI = 200
FPS = 60
scatter_size = 60
scatter_opacity = 0.25
target_resolution = np.array([2560, 1440])
y_axis_margin = 0.05  # Percentage
label_fontsize = 30
ticks_fontsize = 18
annotate_fontsize = 18
draw_points = True
draw_min = True
draw_references = True

# Calculated from parameters
total_intervals = math.ceil(duration * FPS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", "-i", type=str, required=True)
    args = parser.parse_args()
    df = pandas.read_csv(args.inputs)
    df["Wall time"] -= df["Wall time"].iloc[0]
    df["Wall time"] /= 60 * 60  # seconds to hours
    df["Race time min"] = np.minimum.accumulate(df["Value"])

    plt.rcParams.update(
        {
            "figure.facecolor": (1.0, 1.0, 1.0, 0.0),
            "axes.facecolor": (1.0, 1.0, 1.0, 0.0),
            "savefig.facecolor": (1.0, 1.0, 1.0, 0.0),
        }
    )
    x_min = df["Wall time"].min()
    x_max = df["Wall time"].max()
    fig, ax = plt.subplots(figsize=target_resolution / DPI)
    twin_ax = ax.twinx()
    fig.patch.set_alpha(0.0)
    # ax.patch.set_alpha(0.0)
    ax.set_xlabel("Training hours", fontsize=label_fontsize)
    ax.set_ylabel("Race time", fontsize=label_fontsize)
    ax.tick_params(labelsize=ticks_fontsize)
    twin_ax.tick_params(labelsize=ticks_fontsize)

    if draw_references:
        reference_lines = ax.hlines(target_times, 0, df["Wall time"].max(), linestyle=((0, (1, 4))), color="green")
    if draw_points:
        points_scatter = ax.scatter([], [], s=scatter_size, color="blue", alpha=scatter_opacity)  # , linewidth=0
    if draw_min:
        min_line = ax.plot([], [], color="red")[0]
        min_text = ax.annotate("", xy=(df["Wall time"].max() / 2, df["Value"].mean()), color="red", fontsize=annotate_fontsize)

    def animate(i):
        points_to_show = max(1, round((i + 1) * len(df["Wall time"]) / total_intervals))
        print(f"Plotting {points_to_show}/{len(df['Value'])}")
        # ax.clear()
        # ax.patch.set_visible(False)
        # axis_relevant_values = df['Value'].iloc[max(0,round(points_to_show-len(df['Value'])*Axis_Hour_Range)):points_to_show]
        # y_max = axis_relevant_values.max()
        # y_min = axis_relevant_values.min()
        axis_relevant_values = df["Value"].iloc[:points_to_show]
        y_min = axis_relevant_values.min()
        y_max = np.percentile(axis_relevant_values, y_axis_percentile)
        twin_ax.set_yticks(target_times, target_times_labels if len(target_times_labels) > 0 else None)
        ax.set_ylim(y_min - (y_max - y_min) * y_axis_margin, y_max)
        twin_ax.set_ylim(ax.get_ylim())
        ax.set_xlim(x_min, x_max)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))
        if draw_references:
            # reference_lines.set_offsets((x_min,x_max))
            pass
        if draw_points:
            points_scatter.set_offsets(df[["Wall time", "Value"]].iloc[:points_to_show])
        if draw_min:
            min_line.set_data(df["Wall time"].iloc[:points_to_show], df["Race time min"].iloc[:points_to_show])
            min_text.set_text("{:.2f}".format(round(df["Race time min"].iloc[points_to_show - 1], 2)))
            min_text.set_position(
                (
                    df["Wall time"].iloc[points_to_show - 1] - 0.05 * (x_max - x_min),
                    df["Race time min"].iloc[points_to_show - 1] - (y_max - y_min) * y_axis_margin * 0.75,
                )
            )

    ani = animation.FuncAnimation(fig, animate, frames=total_intervals)
    FFwriter = animation.FFMpegWriter(fps=FPS, codec="png")  # , extra_args=['-pix_fmt', 'yuva444p']
    # FFwriter = animation.PillowWriter(fps=FPS, codec="gif") #,extra_args=['-pix_fmt', 'yuva444p']
    # FFwriter = animation.ImageMagickWriter(fps=FPS, codec="gif")
    Start_Time = time.perf_counter()
    ani.save("animated_race_time.mov", writer=FFwriter, dpi=DPI)  # ,savefig_kwargs={"transparent": True}
    print("Took", time.perf_counter() - Start_Time, "s to make mp4")
