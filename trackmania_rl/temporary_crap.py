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

    (save_dir / "figures_A" / map_name).mkdir(parents=True, exist_ok=True)
    (save_dir / "figures_Q" / map_name).mkdir(parents=True, exist_ok=True)

    rollout_results_copy = rollout_results.copy()
    for frame_number in [0, 5, 10, 20, -180, -140, -100, -60, -20]:
        if frame_number < 0 and "race_time" not in rollout_results:
            return

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
                per_quantile_output = trainer.infer_online_network(
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
            plt.savefig(save_dir / "figures_A" / map_name / f"crap_A_{str(x_axis)[5:]}_{frame_number}_{map_name}.png")
            plt.close()

            for i in reversed(range(12)):
                plt.plot(x_axis, q[i], label=str(i), c=color_cycle[i])
            plt.gcf().legend()
            plt.gcf().suptitle(f"crap_Q_{str(x_axis)[5:]}_{frame_number}_{map_name}.png")
            plt.savefig(save_dir / "figures_Q" / map_name / f"crap_Q_{str(x_axis)[5:]}_{frame_number}_{map_name}.png")
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

        per_quantile_output = trainer.infer_online_network(
            rollout_results_copy["frames"][frame_number], rollout_results_copy["state_float"][frame_number], tau
        )  # (iqn_k, n_actions)

        for i in range(n_best_actions_to_plot):
            action_idx = per_quantile_output.mean(axis=0).argmax()
            axes[i].plot(
                tau.to(device="cpu"), per_quantile_output[:, action_idx] - per_quantile_output[:, action_idx].mean(), c="gray", alpha=0.2
            )

            per_quantile_output[:, action_idx] -= 10000

    (save_dir / "figures_tau" / map_name).mkdir(parents=True, exist_ok=True)

    for i in range(n_best_actions_to_plot):
        figs[i].suptitle(f"tau_{i}_{map_name}.png")
        figs[i].savefig(save_dir / "figures_tau" / map_name / f"tau_{i}_{map_name}.png")
        plt.close(figs[i])


def patrick_curves(rollout_results, trainer, save_dir, map_name):
    if "race_time" not in rollout_results:
        return

    rollout_results_copy = rollout_results.copy()

    tau = torch.linspace(0.05, 0.95, misc.iqn_k)[:, None].to("cuda")

    horizons_to_plot = [140, 120, 100, 80, 60, 40, 20, 10]

    figs, axes = zip(*[plt.subplots() for _ in range(len(horizons_to_plot))])

    values_predicted = [[] for _ in horizons_to_plot]
    values_observed = [[] for _ in horizons_to_plot]

    for frame_number in range(0, len(rollout_results_copy["frames"]) - 200, 5):
        for ihorz, horizon in enumerate(horizons_to_plot):
            rollout_results_copy["state_float"][frame_number][0] = 140 - horizon

            per_quantile_output = trainer.infer_online_network(
                rollout_results_copy["frames"][frame_number], rollout_results_copy["state_float"][frame_number], tau
            )  # (iqn_k, n_actions)

            action_idx = per_quantile_output.mean(axis=0).argmax()
            values_predicted[ihorz].append(per_quantile_output[:, action_idx].mean())

            values_observed[ihorz].append(
                horizon * misc.ms_per_action * misc.constant_reward_per_ms
                + misc.reward_per_m_advanced_along_centerline
                * (
                    rollout_results_copy["meters_advanced_along_centerline"][frame_number + horizon]
                    - rollout_results_copy["meters_advanced_along_centerline"][frame_number]
                )
            )

    (save_dir / "figures_patrick" / map_name).mkdir(parents=True, exist_ok=True)

    for ihorz, horizon in enumerate(horizons_to_plot):
        axes[ihorz].plot(range(len(values_predicted[ihorz])), values_predicted[ihorz], label="Value predicted")
        axes[ihorz].plot(range(len(values_observed[ihorz])), values_observed[ihorz], label="Value observed")

        figs[ihorz].suptitle(f"patrick_{horizon}_{map_name}.png")
        figs[ihorz].legend()
        figs[ihorz].savefig(save_dir / "figures_patrick" / map_name / f"patrick_{horizon}_{map_name}.png")
        plt.close(figs[ihorz])
