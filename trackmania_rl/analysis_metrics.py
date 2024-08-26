"""
This file contains functions to plot figures to help diagnose the agent's learning progress.
Plotting can be enabled/disabled within config.py
"""

import random
import shutil
from collections import defaultdict
from itertools import islice

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import torch
from PIL import Image

from config_files import config_copy
from trackmania_rl.agents.iqn import iqn_loss


def batched(iterable, n):  # Can be included from itertools with python >=3.12
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be >= 1")
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


def race_time_left_curves(rollout_results, inferer, save_dir, map_name):
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
            range(0, config_copy.temporal_mini_race_duration_actions),
            range(int(0.7 * config_copy.temporal_mini_race_duration_actions), config_copy.temporal_mini_race_duration_actions),
        ]:
            q = defaultdict(list)
            a = defaultdict(list)
            q_l = defaultdict(list)
            a_l = defaultdict(list)
            q_h = defaultdict(list)
            a_h = defaultdict(list)

            tau = torch.linspace(0.05, 0.95, config_copy.iqn_k)[:, None].to("cuda")
            for j in x_axis:
                # print(j)
                rollout_results_copy["state_float"][frame_number][0] = j
                per_quantile_output = inferer.infer_network(
                    rollout_results_copy["frames"][frame_number], rollout_results_copy["state_float"][frame_number], tau
                )  # (iqn_k, n_actions)
                for i, q_val in enumerate(list(per_quantile_output.mean(axis=0))):
                    # print(i, q_val)
                    q[i].append(q_val)
                    a[i].append(q_val - per_quantile_output.mean(axis=0).max())

                for i, q_val in enumerate(list(per_quantile_output[: config_copy.iqn_k // 2, :].mean(axis=0))):
                    # print(i, q_val)
                    q_l[i].append(q_val)
                    a_l[i].append(q_val - per_quantile_output[: config_copy.iqn_k // 2, :].mean(axis=0).max())

                for i, q_val in enumerate(list(per_quantile_output[config_copy.iqn_k // 2 :, :].mean(axis=0))):
                    # print(i, q_val)
                    q_h[i].append(q_val)
                    a_h[i].append(q_val - per_quantile_output[config_copy.iqn_k // 2 :, :].mean(axis=0).max())

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


def tau_curves(rollout_results, inferer, save_dir, map_name):
    if "race_time" not in rollout_results:
        return

    rollout_results_copy = rollout_results.copy()

    tau = torch.linspace(0.05, 0.95, config_copy.iqn_k)[:, None].to("cuda")

    n_best_actions_to_plot = 12

    figs, axes = zip(*[plt.subplots() for _ in range(n_best_actions_to_plot)])

    for frame_number in range(100, 500, 5):
        if frame_number > len(rollout_results["frames"]) - 140:
            break

        per_quantile_output = inferer.infer_network(
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


def patrick_curves(rollout_results, inferer, save_dir, map_name):
    if "race_time" not in rollout_results:
        return

    rollout_results_copy = rollout_results.copy()

    tau = torch.linspace(0.05, 0.95, config_copy.iqn_k)[:, None].to("cuda")

    horizons_to_plot = [140, 120, 100, 80, 60, 40, 20, 10]

    figs, axes = zip(*[plt.subplots() for _ in range(len(horizons_to_plot))])

    values_predicted = [[] for _ in horizons_to_plot]
    values_observed = [[] for _ in horizons_to_plot]

    for frame_number in range(0, len(rollout_results_copy["frames"]) - 200, 5):
        for ihorz, horizon in enumerate(horizons_to_plot):
            rollout_results_copy["state_float"][frame_number][0] = 140 - horizon

            per_quantile_output = inferer.infer_network(
                rollout_results_copy["frames"][frame_number], rollout_results_copy["state_float"][frame_number], tau
            )  # (iqn_k, n_actions)

            action_idx = per_quantile_output.mean(axis=0).argmax()
            values_predicted[ihorz].append(per_quantile_output[:, action_idx].mean())

            values_observed[ihorz].append(
                horizon * config_copy.ms_per_action * config_copy.constant_reward_per_ms
                + config_copy.reward_per_m_advanced_along_centerline
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


def highest_prio_transitions(buffer, save_dir):
    shutil.rmtree(save_dir / "high_prio_figures", ignore_errors=True)
    (save_dir / "high_prio_figures").mkdir(parents=True, exist_ok=True)

    prios = [buffer._sampler._sum_tree.at(i) for i in range(len(buffer))]

    for high_error_idx in np.argsort(prios)[-20:]:
        for idx in range(max(0, high_error_idx - 4), min(len(buffer) - 1, high_error_idx + 5)):
            Image.fromarray(
                np.hstack((buffer._storage[idx].state_img.squeeze(), buffer._storage[idx].next_state_img.squeeze()))
                .repeat(4, 0)
                .repeat(4, 1)
            ).save(save_dir / "high_prio_figures" / f"{high_error_idx}_{idx}_{buffer._storage[idx].n_steps}_{prios[idx]:.2f}.png")


def get_output_and_target_for_batch(batch, online_network, target_network, num_quantiles):
    (
        state_img_tensor,
        state_float_tensor,
        actions,
        rewards,
        next_state_img_tensor,
        next_state_float_tensor,
        gammas_terminal,
    ) = batch
    batch_size = len(state_img_tensor)

    is_terminal = gammas_terminal > 0

    delta = next_state_float_tensor[:, 0] - state_float_tensor[:, 0]
    state_float_tensor[:, 0] = (0 - config_copy.float_inputs_mean[0]) / config_copy.float_inputs_std[0]
    next_state_float_tensor[:, 0] = state_float_tensor[:, 0] + delta

    tau = torch.linspace(0, 1, num_quantiles, device="cuda").repeat_interleave(batch_size).unsqueeze(1)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        with torch.no_grad():
            rewards = rewards.unsqueeze(-1).repeat(
                [num_quantiles, 1]
            )  # (batch_size*iqn_n, 1)     a,b,c,d devient a,b,c,d,a,b,c,d,a,b,c,d,...
            gammas_terminal = gammas_terminal.unsqueeze(-1).repeat([num_quantiles, 1])  # (batch_size*iqn_n, 1)
            actions = actions.unsqueeze(-1).repeat([num_quantiles, 1])  # (batch_size*iqn_n, 1)
            #
            #   Use target_network to evaluate the action chosen, per quantile.
            #
            q__stpo__target__quantiles_tau2, _ = target_network(
                next_state_img_tensor, next_state_float_tensor, num_quantiles, tau=tau
            )  # (batch_size*iqn_n,n_actions)
            #
            #   Use online network to choose an action for next state.
            #   This action is chosen AFTER reduction to the mean, and repeated to all quantiles
            #
            outputs_target_tau2 = (
                rewards + gammas_terminal * q__stpo__target__quantiles_tau2.max(dim=1, keepdim=True)[0]
            )  # (batch_size*iqn_n, 1)
            #
            #   This is our target
            #
            outputs_target_tau2 = outputs_target_tau2.reshape([num_quantiles, batch_size, 1]).transpose(0, 1)  # (batch_size, iqn_n, 1)
            q__st__online__quantiles_tau3, tau3 = online_network(
                state_img_tensor, state_float_tensor, num_quantiles, tau=tau
            )  # (batch_size*iqn_n,n_actions)
            outputs_tau3 = (
                q__st__online__quantiles_tau3.gather(1, actions).reshape([num_quantiles, batch_size, 1]).transpose(0, 1)
            )  # (batch_size, iqn_n, 1)

    losses = {
        "target_self_loss": iqn_loss(outputs_target_tau2, outputs_target_tau2, tau, num_quantiles, batch_size).cpu().numpy(),
        "output_self_loss": iqn_loss(outputs_tau3, outputs_tau3, tau, num_quantiles, batch_size).cpu().numpy(),
        "real_loss": iqn_loss(outputs_target_tau2, outputs_tau3, tau, num_quantiles, batch_size).cpu().numpy(),
    }

    return (
        (is_terminal * outputs_tau3).cpu().numpy().astype(np.float32),
        (is_terminal * outputs_target_tau2).cpu().numpy().astype(np.float32),
        losses,
    )


def loss_distribution(buffer, save_dir, online_network, target_network):
    shutil.rmtree(save_dir / "loss_distribution", ignore_errors=True)
    (save_dir / "loss_distribution").mkdir(parents=True, exist_ok=True)
    buffer_loss = []
    for batch in batched(range(len(buffer)), config_copy.batch_size):
        quantiles_output, quantiles_target, losses = get_output_and_target_for_batch(
            buffer[batch], online_network, target_network, config_copy.iqn_n
        )
        buffer_loss.extend(losses["real_loss"])
    buffer_loss = np.array(buffer_loss)
    buffer_loss_mean = buffer_loss.mean()
    plt.figure()
    plt.hist(buffer_loss, bins=50, density=True)
    plt.vlines(buffer_loss_mean, 0, 1, color="red")
    plt.yscale("log")
    plt.title("Buffer Size:" + str(len(buffer)))
    plt.savefig(save_dir / "loss_distribution" / "loss_distribution.png")

    plt.figure()
    plt.hist(buffer_loss / buffer_loss_mean, bins=50, density=True)
    plt.yscale("log")
    plt.title("Buffer Size:" + str(len(buffer)))
    plt.savefig(save_dir / "loss_distribution" / "loss_distribution_mean_units.png")

    plt.figure()
    plt.hist(buffer_loss_mean / buffer_loss, bins=50, density=True)
    plt.yscale("log")
    plt.title("Buffer Size:" + str(len(buffer)))
    plt.savefig(save_dir / "loss_distribution" / "loss_distribution_inverse_mean_units.png")


def distribution_curves(buffer, save_dir, online_network, target_network):
    if config_copy.n_transitions_to_plot_in_distribution_curves <= 0:
        return

    shutil.rmtree(save_dir / "distribution_curves", ignore_errors=True)
    (save_dir / "distribution_curves").mkdir(parents=True, exist_ok=True)

    first_transition_to_plot = random.randrange(4000, len(buffer) - config_copy.n_transitions_to_plot_in_distribution_curves)

    num_quantiles = 16
    my_dpi = 100
    max_height = 60

    for i in range(first_transition_to_plot, first_transition_to_plot + config_copy.n_transitions_to_plot_in_distribution_curves):
        fig, ax = plt.subplots(figsize=(640 / my_dpi, 480 / my_dpi), dpi=my_dpi)

        quantiles_output, quantiles_target, losses = get_output_and_target_for_batch(
            buffer[[i]], online_network, target_network, num_quantiles
        )

        quantiles_output = np.sort(quantiles_output.ravel())
        quantiles_target = np.sort(quantiles_target.ravel())

        if (np.min(quantiles_output) == np.max(quantiles_output)) and (np.min(quantiles_output) == 0.0):
            # terminal transition, can't be interpreted as long term
            continue

        x_output = 0.5 * (quantiles_output[1:] + quantiles_output[:-1])
        height_output = np.clip(
            1 / ((num_quantiles - 1) * (quantiles_output[1:] - quantiles_output[:-1]) + 5e-4), a_min=None, a_max=max_height
        )
        width_output = 1 / ((num_quantiles - 1) * height_output)

        ax.bar(x=x_output, height=height_output, width=width_output)
        ax.vlines(
            0.5 * (quantiles_output[1:] + quantiles_output[:-1])[[1, 3, 7, -4, -2]],
            0,
            np.max(height_output),
            linestyles="dotted",
            color="lightblue",
        )
        ax.vlines(quantiles_output[[0, 15]], 0, np.max(height_output), linestyles="dotted", color="lightblue")

        x_target = 0.5 * (quantiles_target[1:] + quantiles_target[:-1])
        height_target = np.clip(
            1 / ((num_quantiles - 1) * (quantiles_target[1:] - quantiles_target[:-1]) + 5e-4), a_min=None, a_max=max_height
        )
        width_target = 1 / ((num_quantiles - 1) * height_target)

        ax.bar(x=x_target, height=-height_target, width=width_target)
        ax.vlines(
            0.5 * (quantiles_target[1:] + quantiles_target[:-1])[[1, 3, 7, -4, -2]],
            0,
            -np.max(height_target),
            linestyles="dotted",
            color="orange",
        )
        ax.vlines(quantiles_target[[0, 15]], 0, -np.max(height_target), linestyles="dotted", color="orange")

        ax.set_axisbelow(True)
        ax.grid(color="#E0E0E0")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        loc = plticker.MultipleLocator(base=0.025)  # this locator puts ticks at regular intervals
        ax.xaxis.set_major_locator(loc)

        ax.set_title("    ".join([f"{1000*v[0]:.2f}" for k, v in losses.items()]))
        for k, v in losses.items():
            print(k)

        print("")
        print("")
        print("")
        print("")

        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        Image.fromarray(
            (
                np.hstack(
                    (
                        np.expand_dims(
                            np.hstack((buffer._storage[i].state_img.squeeze(), buffer._storage[i].next_state_img.squeeze()))
                            .repeat(4, 0)
                            .repeat(4, 1),
                            axis=-1,
                        ).repeat(3, axis=-1),
                        image_from_plot,
                    )
                )
            )
        ).save(save_dir / "distribution_curves" / f"{i}_{buffer._storage[i].n_steps}.png")
        plt.close()
