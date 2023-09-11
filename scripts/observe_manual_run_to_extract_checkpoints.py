import sys
from pathlib import Path

import numpy as np
from tminterface.client import Client, run_client
from tminterface.interface import TMInterface

from trackmania_rl.geometry import extract_cp_distance_interval


class MainClient(Client):
    def __init__(self) -> None:
        super(MainClient, self).__init__()
        self.race_finished = False
        self.raw_position_list = []
        self.period_save_pos_ms = 10
        self.target_distance_between_cp_m = 0.5
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
            self.zone_centers = extract_cp_distance_interval(self.raw_position_list, self.target_distance_between_cp_m, base_dir)


base_dir = Path(__file__).resolve().parents[1]
server_name = f"TMInterface{sys.argv[1]}" if len(sys.argv) > 1 else "TMInterface0"
print(f"Connecting to {server_name}...")
client = MainClient()
run_client(client, server_name)
