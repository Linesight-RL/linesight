from collections import defaultdict

import matplotlib.pyplot as plt
import torch

from . import misc


def race_time_left_curves(rollout_results, trainer, save_dir, map_name):
    color_cycle = [
        "red",
        "forestgreen",
        "blue",
        "darkred",
        "darkgreen",
        "darkblue",
        "salmon",
        "limegreen",
        "cornflowerblue",
        "orange",
        "lime",
        "cyan",
    ]

    (save_dir / "figures_A").mkdir(parents=True, exist_ok=True)
    (save_dir / "figures_Q").mkdir(parents=True, exist_ok=True)

    rollout_results_copy = rollout_results.copy()
    for frame_number in [0, 5, 10, 20]:
        for x_axis in [
            range(0, misc.temporal_mini_race_duration_actions),
            range(int(0.7 * misc.temporal_mini_race_duration_actions), misc.temporal_mini_race_duration_actions),
        ]:
            q = defaultdict(list)
            a = defaultdict(list)
            q_l = defaultdict(list)
            a_l = defaultdict(list)
            q_h = defaultdict(list)
            a_h = defaultdict(list)

            tau = torch.linspace(0.05, 0.95, misc.iqn_k)[:, None].to("cuda")
            for j in x_axis:
                # print(j)
                rollout_results_copy["state_float"][frame_number][0] = j
                per_quantile_output = trainer.infer_model(
                    rollout_results_copy["frames"][frame_number], rollout_results_copy["state_float"][frame_number], tau
                )  # (iqn_k, n_actions)
                for i, q_val in enumerate(list(per_quantile_output.mean(axis=0))):
                    # print(i, q_val)
                    q[i].append(q_val)
                    a[i].append(q_val - per_quantile_output.mean(axis=0).max())

                for i, q_val in enumerate(list(per_quantile_output[: misc.iqn_k // 2, :].mean(axis=0))):
                    # print(i, q_val)
                    q_l[i].append(q_val)
                    a_l[i].append(q_val - per_quantile_output[: misc.iqn_k // 2, :].mean(axis=0).max())

                for i, q_val in enumerate(list(per_quantile_output[misc.iqn_k // 2 :, :].mean(axis=0))):
                    # print(i, q_val)
                    q_h[i].append(q_val)
                    a_h[i].append(q_val - per_quantile_output[misc.iqn_k // 2 :, :].mean(axis=0).max())

            for i in reversed(range(12)):
                plt.plot(x_axis, a_l[i], label=str(i) + "_l", c=color_cycle[i], linestyle="dotted")
                plt.plot(x_axis, a_h[i], label=str(i) + "_h", c=color_cycle[i], linestyle="dashed")
            for i in reversed(range(12)):
                plt.plot(x_axis, a[i], label=str(i), c=color_cycle[i])
            plt.gcf().legend()
            plt.gcf().suptitle(f"crap_A_{str(x_axis)[5:]}_{frame_number}_{map_name}.png")
            plt.savefig(save_dir / "figures_A" / f"crap_A_{str(x_axis)[5:]}_{frame_number}_{map_name}.png")
            plt.close()

            for i in reversed(range(12)):
                plt.plot(x_axis, q[i], label=str(i), c=color_cycle[i])
            plt.gcf().legend()
            plt.gcf().suptitle(f"crap_Q_{str(x_axis)[5:]}_{frame_number}_{map_name}.png")
            plt.savefig(save_dir / "figures_Q" / f"crap_Q_{str(x_axis)[5:]}_{frame_number}_{map_name}.png")
            plt.close()


def tau_curves(rollout_results, trainer, save_dir, map_name):
    if "race_time" not in rollout_results:
        return

    rollout_results_copy = rollout_results.copy()

    tau = torch.linspace(0.05, 0.95, misc.iqn_k)[:, None].to("cuda")

    n_best_actions_to_plot = 12

    figs, axes = zip(*[plt.subplots() for _ in range(n_best_actions_to_plot)])

    for frame_number in range(100, 500, 5):
        if frame_number > len(rollout_results["frames"]) - 140:
            break

        per_quantile_output = trainer.infer_model(
            rollout_results_copy["frames"][frame_number], rollout_results_copy["state_float"][frame_number], tau
        )  # (iqn_k, n_actions)

        for i in range(n_best_actions_to_plot):
            action_idx = per_quantile_output.mean(axis=0).argmax()
            axes[i].plot(
                tau.to(device="cpu"), per_quantile_output[:, action_idx] - per_quantile_output[:, action_idx].mean(), c="gray", alpha=0.2
            )

            per_quantile_output[:, action_idx] -= 10000

    (save_dir / "figures_tau").mkdir(parents=True, exist_ok=True)

    for i in range(n_best_actions_to_plot):
        figs[i].suptitle(f"tau_{i}_{map_name}.png")
        figs[i].savefig(save_dir / "figures_tau" / f"tau_{i}_{map_name}.png")
        plt.close(figs[i])
