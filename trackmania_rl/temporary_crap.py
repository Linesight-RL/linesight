from collections import defaultdict

import matplotlib.pyplot as plt
import torch

from . import misc


def race_time_left_curves(rollout_results, trainer):
    rollout_results_copy = rollout_results.copy()
    for frame_number in range(0, 10):
        q = defaultdict(list)
        a = defaultdict(list)
        std = defaultdict(list)
        x_axis = range(110, 140)

        tau = torch.linspace(0.05, 0.95, misc.iqn_k)[:, None].to("cuda")
        for j in x_axis:
            # print(j)
            rollout_results_copy["state_float"][frame_number][0] = j
            per_quantile_output = trainer.infer_model(
                rollout_results_copy["frames"][frame_number], rollout_results_copy["state_float"][frame_number], tau
            )
            for i, q_val in enumerate(list(per_quantile_output.mean(axis=0))):
                # print(i, q_val)
                q[i].append(q_val)
                a[i].append(q_val - per_quantile_output.mean(axis=0).max())
            for i, s in enumerate(list(per_quantile_output.std(axis=0))):
                # print(i, q_val)
                std[i].append(s)
            print(j, per_quantile_output.mean(axis=0).argmax())

        for i in range(12):
            plt.plot(x_axis, a[i], label=str(i))
        plt.gcf().legend()
        plt.savefig(f"temporary_crap_figure_a_{frame_number}.png")
        plt.close()

        for i in range(12):
            plt.plot(x_axis, q[i], label=str(i))
        plt.gcf().legend()
        plt.savefig(f"temporary_crap_figure_q_{frame_number}.png")
        plt.close()
