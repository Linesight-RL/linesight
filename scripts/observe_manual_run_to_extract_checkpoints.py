import sys
from pathlib import Path

import numpy as np
from tminterface.client import Client, run_client
from tminterface.interface import TMInterface

Path("maps").mkdir(exist_ok=True)

class MainClient(Client):
    def __init__(self) -> None:
        super(MainClient, self).__init__()
        self.race_finished = False
        self.raw_position_list = []
        self.period_save_pos_ms = 10
        # self.target_time_gap_between_cp_ms = 500
        self.target_distance_between_cp_m = 10
        self.zone_centers = None

    def on_registered(self, iface: TMInterface) -> None:
        print(f"Registered to {iface.server_name}")

    def on_run_step(self, iface: TMInterface, _time: int):
        state = iface.get_simulation_state()

        if _time == 0:
            self.raw_position_list = []

        if _time >= 0 and _time % self.period_save_pos_ms == 0:
            if not self.race_finished:
                self.raw_position_list.append(np.array(state.position))

            # print(f'\x1b[1K\rPosition: {state.position[0]:>8.1f}, {state.position[1]:>8.1f}, {state.position[2]:>8.1f}', end='', flush=True)
            # print(f'\x1b[1K\rVelocity: {state.velocity[0]:>8.1f}, {state.velocity[1]:>8.1f}, {state.velocity[2]:>8.1f}', end='', flush=True)
            # print(f'\x1b[1K\rYPR: {state.yaw_pitch_roll[0]:>8.2f}, {state.yaw_pitch_roll[1]:>8.2f}, {state.yaw_pitch_roll[2]:>8.2f}, ', end='', flush=True)

            # print(
            #     f'Time: {_time}\n'
            #     f'Display Speed: {state.display_speed}\n'
            #     f'Position: {state.position}\n'
            #     f'Velocity: {state.velocity}\n'
            #     f'YPW: {state.yaw_pitch_roll}\n'
            # )

    def on_checkpoint_count_changed(self, iface, current: int, target: int):
        """
        Called when the current checkpoint count changed (a new checkpoint has been passed by the vehicle).
        The `current` and `target` parameters account for the total amount of checkpoints to be collected,
        taking lap count into consideration.

        Args:
            iface (TMInterface): the TMInterface object
            current (int): the current amount of checkpoints passed
            target (int): the total amount of checkpoints on the map (including finish)
        """

        if current == target:
            self.raw_position_list.append(np.array(iface.get_simulation_state().position))
            self.race_finished = True
            self.extract_cp_distance_interval()

    # def extract_zone_centers_time_interval(self):
    #     number_checkpoints = round(len(self.raw_position_list) * self.period_save_pos_ms / self.target_time_gap_between_cp_ms)
    #     self.checkpoints = [
    #         self.raw_position_list[int(i)] for i in np.linspace(0, len(self.raw_position_list) - 1, number_checkpoints).round()
    #     ]
    #     self.checkpoints.append(2 * self.checkpoints[-1] - self.checkpoints[-2])  # Add a virtual checkpoint after the finish line
    #     np.save(base_dir / "maps" / "map.npy", np.array(self.checkpoints).round(1))

    def extract_cp_distance_interval(self):
        a = np.array(self.raw_position_list)
        b = np.linalg.norm(a[:-1] - a[1:], axis=1)  # b[i] : distance traveled between point i and point i+1, for i > 0
        c = np.pad(b.cumsum(), (1, 0))  # c[i] : distance traveled between point 0 and point i
        number_zones = round(c[-1] / self.target_distance_between_cp_m - 0.5) + 0.5  # half a zone for the end
        zone_length = c[-1] / number_zones
        index_first_pos_in_new_zone = np.unique(c // zone_length, return_index=True)[1][1:]
        index_last_pos_in_current_zone = index_first_pos_in_new_zone - 1
        w1 = 1 - (c[index_last_pos_in_current_zone] % zone_length) / zone_length
        w2 = (c[index_first_pos_in_new_zone] % zone_length) / zone_length
        self.zone_centers = a[index_last_pos_in_current_zone] + (a[index_first_pos_in_new_zone] - a[index_last_pos_in_current_zone]) * (
            w1 / (1e-4 + w1 + w2)
        ).reshape((-1, 1))
        self.zone_centers = np.vstack(
            (
                client.raw_position_list[0][None, :],
                self.zone_centers,
                (2 * client.raw_position_list[-1] - self.zone_centers[-1])[None, :],
            )
        )
        np.save(base_dir / "maps" / "map.npy", np.array(self.zone_centers).round(1))

        # Print success message
        save_path = base_dir / "maps" / "map.npy"
        print(f"map.npy was saved successfully to {save_path}")

base_dir = Path(__file__).resolve().parents[1]
server_name = f"TMInterface{sys.argv[1]}" if len(sys.argv) > 1 else "TMInterface0"
print(f"Connecting to {server_name}...")
client = MainClient()
run_client(client, server_name)
# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(26, 15))
plt.scatter(client.zone_centers[:, 0], client.zone_centers[:, 2], s=0.5)
plt.scatter(
    -np.array(client.raw_position_list)[:, 0],
    np.array(client.raw_position_list)[:, 2],
    s=0.5,
)
