import math
import os
import socket
import subprocess
import time

import cv2
import numba
import numpy as np
import numpy.typing as npt
import psutil

from trackmania_rl import contact_materials, map_loader, misc
from trackmania_rl.tmi_interaction.tminterface2 import MessageType, TMInterface

if misc.is_linux:
    from xdo import Xdo
else:
    import win32.lib.win32con as win32con
    import win32com.client
    import win32gui
    from ReadWriteMemory import ReadWriteMemory


def _set_window_focus(
    trackmania_window,
):  # https://stackoverflow.com/questions/14295337/win32gui-setactivewindow-error-the-specified-procedure-could-not-be-found
    if misc.is_linux:
        Xdo.focus_window(trackmania_window)
    else:
        shell = win32com.client.Dispatch("WScript.Shell")
        shell.SendKeys("%")
        win32gui.SetForegroundWindow(trackmania_window)


def is_fullscreen(trackmania_window):
    if misc.is_linux:
        return False  # shape = Xdo().get_window_size()
    else:
        rect = win32gui.GetWindowPlacement(trackmania_window)[4]
        return rect[0] == 0 and rect[1] == 0 and rect[2] == misc.W_screen and rect[3] == misc.H_screen


def ensure_not_minimized(trackmania_window):
    if misc.is_linux:
        Xdo().map_window(trackmania_window)
    else:
        if win32gui.IsIconic(
            trackmania_window
        ):  # https://stackoverflow.com/questions/54560987/restore-window-without-setting-to-foreground
            win32gui.ShowWindow(trackmania_window, win32con.SW_SHOWNORMAL)  # Unminimize window
        if is_fullscreen(trackmania_window):
            _set_window_focus(trackmania_window)


@numba.njit
def update_current_zone_idx(current_zone_idx, zone_centers, sim_state_position):
    d1 = np.linalg.norm(zone_centers[current_zone_idx + 1] - sim_state_position)
    d2 = np.linalg.norm(zone_centers[current_zone_idx] - sim_state_position)
    d3 = np.linalg.norm(zone_centers[current_zone_idx - 1] - sim_state_position)
    while (
        d1 <= d2
        and d1 <= misc.max_allowable_distance_to_checkpoint
        and current_zone_idx < len(zone_centers) - 1 - misc.n_zone_centers_extrapolate_after_end_of_map
        # We can never enter the final virtual zone
    ):
        # Move from one virtual zone to another
        current_zone_idx += 1
        d2, d3 = d1, d2
        d1 = np.linalg.norm(zone_centers[current_zone_idx + 1] - sim_state_position)
    while current_zone_idx >= 2 and d3 < d2 and d3 <= misc.max_allowable_distance_to_checkpoint:
        current_zone_idx -= 1
        d1, d2 = d2, d3
        d3 = np.linalg.norm(zone_centers[current_zone_idx - 1] - sim_state_position)
    return current_zone_idx


class TMInterfaceManager:
    def __init__(
        self,
        base_dir,
        running_speed=1,
        run_steps_per_action=10,
        max_overall_duration_ms=2000,
        max_minirace_duration_ms=2000,
        tmi_port=None,
    ):
        # Create TMInterface we will be using to interact with the game client
        self.iface = None
        self.latest_tm_engine_speed_requested = 1
        self.running_speed = running_speed
        self.run_steps_per_action = run_steps_per_action
        self.max_overall_duration_ms = max_overall_duration_ms
        self.max_minirace_duration_ms = max_minirace_duration_ms
        self.timeout_has_been_set = False
        self.msgtype_response_to_wakeup_TMI = None
        self.latest_map_path_requested = -2
        self.last_rollout_crashed = False
        self.last_game_reboot = time.perf_counter()
        self.UI_disabled = False
        self.tmi_port = tmi_port
        self.tm_process_id = None
        self.tm_window_id = None
        self.start_states = {}

    def get_tm_window_id(self):
        assert self.tm_process_id is not None

        if misc.is_linux:
            self.tm_window_id = Xdo().search_windows(winname=b"Track", pid=self.tm_process_id)
        else:

            def get_hwnds_for_pid(pid):
                def callback(hwnd, hwnds):
                    _, found_pid = win32process.GetWindowThreadProcessId(hwnd)

                    if found_pid == pid:
                        hwnds.append(hwnd)
                    return True

                hwnds = []
                win32gui.EnumWindows(callback, hwnds)
                return hwnds

            while True:
                for hwnd in get_hwnds_for_pid(self.tm_process_id):
                    if win32gui.GetWindowText(hwnd).startswith("Track"):
                        self.tm_window_id = hwnd
                        return
                # else:
                #     raise Exception("Could not find TmForever window id.")

    def launch_game(self):
        self.tm_process_id = None

        if misc.is_linux:
            pid_before = [proc.pid for proc in psutil.process_iter() if proc.name().startswith("TmForever")]
            os.system("./launch_game.sh " + str(self.tmi_port))
            pid_after = [proc.pid for proc in psutil.process_iter() if proc.name().startswith("TmForever")]
            tmi_pid_candidates = set(pid_after) - set(pid_before)
            assert len(tmi_pid_candidates) == 1
            self.tm_process_id = list(tmi_pid_candidates)[0]
        else:
            tmi_process_id = int(
                subprocess.check_output(
                    'powershell -executionPolicy bypass -command "& {$process = start-process $args[0] -passthru -argumentList \'/configstring=\\"set custom_port '
                    + str(self.tmi_port)
                    + '\\"\'; echo exit $process.id}" TMInterface.lnk'
                )
                .decode()
                .split("\r\n")[1]
            )

            print(f"Found {tmi_process_id=}")

            tm_processes = list(
                filter(
                    lambda s: s.startswith("TmForever"),
                    subprocess.check_output("wmic process get Caption,ParentProcessId,ProcessId").decode().split("\r\n"),
                )
            )
            for process in tm_processes:
                name, parent_id, process_id = process.split()
                parent_id = int(parent_id)
                process_id = int(process_id)
                if parent_id == tmi_process_id:
                    self.tm_process_id = process_id

        assert self.tm_process_id is not None
        print(f"Found Trackmania process id: {self.tm_process_id=}")
        self.last_game_reboot = time.perf_counter()
        self.latest_map_path_requested = -1
        self.msgtype_response_to_wakeup_TMI = None
        while not self.is_game_running():
            time.sleep(0)

        self.get_tm_window_id()

    def is_game_running(self):
        return (self.tm_process_id is not None) and (self.tm_process_id in (p.pid for p in psutil.process_iter()))

    def close_game(self):
        assert self.tm_process_id is not None
        if misc.is_linux:
            os.system("kill -9 " + str(self.tm_process_id))
        else:
            os.system(f"taskkill /PID {self.tm_process_id} /f")
        while self.is_game_running():
            time.sleep(0)

    def game_shortcut_exists(self):
        return os.path.exists("./launch_game.sh") if misc.is_linux else os.path.exists(".\\TMInterface.lnk")

    def ensure_game_launched(self):
        if not self.is_game_running():
            if self.game_shortcut_exists():
                print("Game not found. Restarting TMInterface.")
                self.launch_game()
            else:
                print(
                    """Game needs to be restarted but cannot be.
                    Add TMInterface shortcut inside /scripts directory (TMInterface.lnk for windows, launch_game.sh for linux).
                    """
                )

    def grab_screen(self):
        return self.iface.get_frame(misc.W_downsized, misc.H_downsized)

    def request_speed(self, requested_speed):
        self.iface.set_speed(requested_speed)
        self.latest_tm_engine_speed_requested = requested_speed

    def request_inputs(self, action_idx, rollout_results):
        if (
            len(rollout_results["actions"]) == 0 or rollout_results["actions"][-1] != action_idx
        ):  # Small performance trick, don't update input_state if it doesn't need to be updated
            self.iface.set_input_state(**misc.inputs[action_idx])

    def request_map(self, map_path):
        self.iface.execute_command(f"map {map_path}")
        # self.iface.execute_command("press delete")
        self.latest_map_path_requested = map_path
        self.UI_disabled = False

    def rollout(self, exploration_policy, map_path: str, zone_centers: npt.NDArray):
        (
            zone_transitions,
            distance_between_zone_transitions,
            distance_from_start_track_to_prev_zone_transition,
            normalized_vector_along_track_axis,
        ) = map_loader.precalculate_virtual_checkpoints_information(zone_centers)

        self.ensure_game_launched()
        if time.perf_counter() - self.last_game_reboot > misc.game_reboot_interval:
            self.close_game()
            self.iface = None
            self.launch_game()

        end_race_stats = {
            "cp_time_ms": [0],
        }

        time_to_answer_normal_step = 0
        time_to_answer_action_step = 0
        time_between_normal_on_run_steps = 0
        time_between_action_on_run_steps = 0
        time_to_grab_frame = 0
        time_between_grab_frame = 0
        time_to_iface_set_set = 0
        time_after_iface_set_set = 0
        time_exploration_policy = 0
        time_A_rgb2gray = 0
        time_A_geometry = 0
        time_A_stack = 0

        rollout_results = {
            "current_zone_idx": [],
            "frames": [],
            "input_w": [],
            "actions": [],
            "action_was_greedy": [],
            "car_gear_and_wheels": [],
            "q_values": [],
            "meters_advanced_along_centerline": [],
            "state_float": [],
            "furthest_zone_idx": 0,
        }

        last_progress_improvement_ms = 0

        if (self.iface is None) or (not self.iface.registered):
            assert self.msgtype_response_to_wakeup_TMI is None
            print("Initialize connection to TMInterface ")
            self.iface = TMInterface(self.tmi_port)

            if not self.iface.registered:
                while True:
                    try:
                        self.iface.register(misc.tmi_protection_timeout_s)
                        break
                    except ConnectionRefusedError as e:
                        print(e)
        else:
            assert self.msgtype_response_to_wakeup_TMI is not None or self.last_rollout_crashed

            self.request_speed(self.running_speed)
            if self.msgtype_response_to_wakeup_TMI is not None:
                self.iface._respond_to_call(self.msgtype_response_to_wakeup_TMI)
                self.msgtype_response_to_wakeup_TMI = None

        self.last_rollout_crashed = False

        _time = -3000
        current_zone_idx = 1

        give_up_signal_has_been_sent = False
        this_rollout_has_seen_t_negative = False
        this_rollout_is_finished = False
        n_th_action_we_compute = 0
        compute_action_asap = False
        compute_action_asap_floats = False
        frame_expected = False
        map_change_requested_time = math.inf

        last_known_simulation_state = None

        time_last_on_run_step = time.perf_counter()

        try:
            while not this_rollout_is_finished:
                if compute_action_asap_floats:
                    pc2 = time.perf_counter_ns()

                    sim_state_race_time = last_known_simulation_state.race_time
                    sim_state_dyna_current = last_known_simulation_state.dyna.current_state
                    sim_state_mobil = last_known_simulation_state.scene_mobil
                    sim_state_mobil_engine = sim_state_mobil.engine
                    simulation_wheels = last_known_simulation_state.simulation_wheels
                    wheel_state = [simulation_wheels[i].real_time_state for i in range(4)]
                    sim_state_position = np.array(
                        sim_state_dyna_current.position,
                        dtype=np.float32,
                    )  # (3,)
                    sim_state_orientation = sim_state_dyna_current.rotation.to_numpy().T  # (3, 3)
                    sim_state_velocity = np.array(
                        sim_state_dyna_current.linear_speed,
                        dtype=np.float32,
                    )  # (3,)
                    sim_state_angular_speed = np.array(
                        sim_state_dyna_current.angular_speed,
                        dtype=np.float32,
                    )  # (3,)

                    gearbox_state = sim_state_mobil.gearbox_state
                    counter_gearbox_state = 0
                    if gearbox_state != 0 and len(rollout_results["car_gear_and_wheels"]) > 0:
                        counter_gearbox_state = 1 + rollout_results["car_gear_and_wheels"][-1][15]

                    sim_state_car_gear_and_wheels = np.array(
                        [
                            *(ws.is_sliding for ws in wheel_state),  # Bool
                            *(ws.has_ground_contact for ws in wheel_state),  # Bool
                            *(ws.damper_absorb for ws in wheel_state),  # 0.005 min, 0.15 max, 0.01 typically
                            gearbox_state,  # Bool, except 2 at startup
                            sim_state_mobil_engine.gear,  # 0 -> 5 approx
                            sim_state_mobil_engine.actual_rpm,  # 0-10000 approx
                            counter_gearbox_state,  # Up to typically 28 when changing gears
                            *(
                                i == contact_materials.physics_behavior_fromint[ws.contact_material_id & 0xFFFF]
                                for ws in wheel_state
                                for i in range(misc.n_contact_material_physics_behavior_types)
                            ),
                        ],
                        dtype=np.float32,
                    )

                    current_zone_idx = update_current_zone_idx(current_zone_idx, zone_centers, sim_state_position)

                    if current_zone_idx > rollout_results["furthest_zone_idx"]:
                        last_progress_improvement_ms = sim_state_race_time
                        rollout_results["furthest_zone_idx"] = current_zone_idx

                    rollout_results["current_zone_idx"].append(current_zone_idx)

                    meters_in_current_zone = np.clip(
                        (sim_state_position - zone_transitions[current_zone_idx - 1]).dot(
                            normalized_vector_along_track_axis[current_zone_idx - 1]
                        ),
                        0,
                        distance_between_zone_transitions[current_zone_idx - 1],
                    )

                    distance_since_track_begin = (
                        distance_from_start_track_to_prev_zone_transition[current_zone_idx - 1] + meters_in_current_zone
                    )

                    # ===================================================================================================

                    pc3 = time.perf_counter_ns()
                    time_between_grab_frame += pc3 - pc2

                    # ==== Construct features
                    state_zone_center_coordinates_in_car_reference_system = sim_state_orientation.dot(
                        (
                            zone_centers[
                                current_zone_idx : current_zone_idx
                                + misc.one_every_n_zone_centers_in_inputs
                                * misc.n_zone_centers_in_inputs : misc.one_every_n_zone_centers_in_inputs,
                                :,
                            ]
                            - sim_state_position
                        ).T
                    ).T  # (n_zone_centers_in_inputs, 3)
                    state_y_map_vector_in_car_reference_system = sim_state_orientation.dot(np.array([0, 1, 0]))
                    state_car_velocity_in_car_reference_system = sim_state_orientation.dot(sim_state_velocity)
                    state_car_angular_velocity_in_car_reference_system = sim_state_orientation.dot(sim_state_angular_speed)

                    previous_actions = [
                        misc.inputs[rollout_results["actions"][k] if k >= 0 else misc.action_forward_idx]
                        for k in range(len(rollout_results["actions"]) - misc.n_prev_actions_in_inputs, len(rollout_results["actions"]))
                    ]

                    pc4 = time.perf_counter_ns()
                    time_A_geometry += pc4 - pc3

                    floats = np.hstack(
                        (
                            0,
                            np.array(
                                [
                                    previous_action[input_str]
                                    for previous_action in previous_actions
                                    for input_str in ["accelerate", "brake", "left", "right"]
                                ]
                            ),  # NEW
                            sim_state_car_gear_and_wheels.ravel(),  # NEW
                            state_car_angular_velocity_in_car_reference_system.ravel(),  # NEW
                            state_car_velocity_in_car_reference_system.ravel(),
                            state_y_map_vector_in_car_reference_system.ravel(),
                            state_zone_center_coordinates_in_car_reference_system.ravel(),
                            min(
                                misc.margin_to_announce_finish_meters,
                                distance_from_start_track_to_prev_zone_transition[
                                    len(zone_centers) - misc.n_zone_centers_extrapolate_after_end_of_map
                                ]
                                - distance_since_track_begin,
                            ),
                        )
                    ).astype(np.float32)

                    pc5 = time.perf_counter_ns()
                    time_A_stack += pc5 - pc4

                    compute_action_asap_floats = False

                msgtype = self.iface._read_int32()

                # =============================================
                #        READ INCOMING MESSAGES
                # =============================================
                if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                    _time = self.iface._read_int32()

                    if _time > 0 and this_rollout_has_seen_t_negative:
                        if _time % 50 == 0:
                            time_between_normal_on_run_steps += time.perf_counter_ns() - pc
                        elif _time % 60 == 0:
                            time_between_action_on_run_steps += time.perf_counter_ns() - pc
                    pc = time.perf_counter_ns()

                    # ============================
                    # BEGIN ON RUN STEP
                    # ============================

                    if not self.timeout_has_been_set:
                        self.iface.set_timeout(misc.timeout_during_run_ms)
                        self.timeout_has_been_set = True

                    if not self.UI_disabled and _time < map_change_requested_time:
                        self.iface.toggle_interface(False)
                        self.UI_disabled = True

                    if _time == 0 and (map_path not in self.start_states):
                        self.start_states[map_path] = self.iface.get_simulation_state()

                    if (not give_up_signal_has_been_sent) and (map_path != self.latest_map_path_requested):
                        self.request_map(map_path)
                        map_change_requested_time = _time
                        give_up_signal_has_been_sent = True
                    elif (not give_up_signal_has_been_sent) and (map_path not in self.start_states):
                        self.iface.give_up()
                        give_up_signal_has_been_sent = True
                    else:
                        if not give_up_signal_has_been_sent:
                            self.iface.rewind_to_state(self.start_states[map_path])
                            _time = 0
                            give_up_signal_has_been_sent = True
                            this_rollout_has_seen_t_negative = True
                        if (
                            (_time > self.max_overall_duration_ms or _time > last_progress_improvement_ms + self.max_minirace_duration_ms)
                            and this_rollout_has_seen_t_negative
                            and not this_rollout_is_finished
                        ):
                            # FAILED TO FINISH IN TIME
                            simulation_state = self.iface.get_simulation_state()
                            race_time = max([simulation_state.race_time, 1e-12])  # Epsilon trick to avoid division by zero

                            end_race_stats["race_finished"] = False
                            end_race_stats["race_time"] = misc.cutoff_rollout_if_race_not_finished_within_duration_ms
                            end_race_stats["race_time_for_ratio"] = race_time
                            end_race_stats["time_to_answer_normal_step"] = time_to_answer_normal_step / race_time * 50
                            end_race_stats["time_to_answer_action_step"] = time_to_answer_action_step / race_time * 50
                            end_race_stats["time_between_normal_on_run_steps"] = time_between_normal_on_run_steps / race_time * 50
                            end_race_stats["time_between_action_on_run_steps"] = time_between_action_on_run_steps / race_time * 50
                            end_race_stats["time_to_grab_frame"] = time_to_grab_frame / race_time * 50
                            end_race_stats["time_between_grab_frame"] = time_between_grab_frame / race_time * 50
                            end_race_stats["time_A_rgb2gray"] = time_A_rgb2gray / race_time * 50
                            end_race_stats["time_A_geometry"] = time_A_geometry / race_time * 50
                            end_race_stats["time_A_stack"] = time_A_stack / race_time * 50
                            end_race_stats["time_exploration_policy"] = time_exploration_policy / race_time * 50
                            end_race_stats["time_to_iface_set_set"] = time_to_iface_set_set / race_time * 50
                            end_race_stats["time_after_iface_set_set"] = time_after_iface_set_set / race_time * 50
                            end_race_stats["tmi_protection_cutoff"] = False

                            self.iface.rewind_to_current_state()

                            self.msgtype_response_to_wakeup_TMI = msgtype
                            self.iface.set_timeout(misc.timeout_between_runs_ms)
                            if frame_expected:
                                self.iface.unrequest_frame()
                                frame_expected = False
                            this_rollout_is_finished = True

                        if not this_rollout_is_finished:
                            this_rollout_has_seen_t_negative |= _time < 0

                            if _time >= 0 and _time % (10 * self.run_steps_per_action) == 0 and this_rollout_has_seen_t_negative:
                                last_known_simulation_state = self.iface.get_simulation_state()
                                self.iface.rewind_to_current_state()
                                self.request_speed(0)
                                compute_action_asap = True  # not self.iface.race_finished() #Paranoid check that the race is not finished, which I think could happen because on_step comes before on_cp_count
                                if compute_action_asap:
                                    compute_action_asap_floats = True
                                    frame_expected = True
                                    self.iface.request_frame(misc.W_downsized, misc.H_downsized)
                    # ============================
                    # END ON RUN STEP
                    # ============================
                    if self.msgtype_response_to_wakeup_TMI is None:
                        self.iface._respond_to_call(msgtype)

                    if _time > 0 and this_rollout_has_seen_t_negative:
                        if _time % 40 == 0:
                            time_to_answer_normal_step += time.perf_counter_ns() - pc
                            pc = time.perf_counter_ns()
                        elif _time % 50 == 0:
                            time_to_answer_action_step += time.perf_counter_ns() - pc
                            pc = time.perf_counter_ns()
                elif msgtype == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
                    current = self.iface._read_int32()
                    target = self.iface._read_int32()

                    simulation_state = self.iface.get_simulation_state()
                    end_race_stats["cp_time_ms"].append(simulation_state.race_time)
                    # ============================
                    # BEGIN ON CP COUNT
                    # ============================

                    if current == target:  # Finished the race !!
                        cp_times_bug_handling_attempts = 0
                        while len(simulation_state.cp_data.cp_times) == 0 and cp_times_bug_handling_attempts < 5:
                            cp_times_bug_handling_attempts += 1
                        if len(simulation_state.cp_data.cp_times) != 0:
                            simulation_state.cp_data.cp_times[-1].time = -1  # Equivalent to prevent_simulation_finish()
                            self.iface.rewind_to_state(simulation_state)
                        else:
                            self.iface.prevent_simulation_finish()

                        if (
                            this_rollout_has_seen_t_negative and not this_rollout_is_finished
                        ):  # We shouldn't take into account a race finished after we ended the rollout
                            if (
                                len(rollout_results["current_zone_idx"]) == len(rollout_results["frames"]) + 1
                            ):  # Handle the case where the floats have been computed but the race ended so we don't actually compute an action
                                rollout_results["current_zone_idx"].pop(-1)

                            end_race_stats["race_finished"] = True
                            end_race_stats["race_time"] = simulation_state.race_time
                            rollout_results["race_time"] = simulation_state.race_time
                            end_race_stats["race_time_for_ratio"] = simulation_state.race_time
                            end_race_stats["time_to_answer_normal_step"] = time_to_answer_normal_step / simulation_state.race_time * 50
                            end_race_stats["time_to_answer_action_step"] = time_to_answer_action_step / simulation_state.race_time * 50
                            end_race_stats["time_between_normal_on_run_steps"] = (
                                time_between_normal_on_run_steps / simulation_state.race_time * 50
                            )
                            end_race_stats["time_between_action_on_run_steps"] = (
                                time_between_action_on_run_steps / simulation_state.race_time * 50
                            )
                            end_race_stats["time_to_grab_frame"] = time_to_grab_frame / simulation_state.race_time * 50
                            end_race_stats["time_between_grab_frame"] = time_between_grab_frame / simulation_state.race_time * 50
                            end_race_stats["time_A_rgb2gray"] = time_A_rgb2gray / simulation_state.race_time * 50
                            end_race_stats["time_A_geometry"] = time_A_geometry / simulation_state.race_time * 50
                            end_race_stats["time_A_stack"] = time_A_stack / simulation_state.race_time * 50
                            end_race_stats["time_exploration_policy"] = time_exploration_policy / simulation_state.race_time * 50
                            end_race_stats["time_to_iface_set_set"] = time_to_iface_set_set / simulation_state.race_time * 50
                            end_race_stats["time_after_iface_set_set"] = time_after_iface_set_set / simulation_state.race_time * 50
                            end_race_stats["tmi_protection_cutoff"] = False

                            this_rollout_is_finished = True  # SUCCESSFULLY FINISHED THE RACE
                            self.msgtype_response_to_wakeup_TMI = msgtype
                            self.iface.set_timeout(misc.timeout_between_runs_ms)
                            if frame_expected:
                                self.iface.unrequest_frame()
                                frame_expected = False

                            rollout_results["current_zone_idx"].append(len(zone_centers) - misc.n_zone_centers_extrapolate_after_end_of_map)
                            rollout_results["frames"].append(np.nan)
                            rollout_results["input_w"].append(np.nan)
                            rollout_results["actions"].append(np.nan)
                            rollout_results["action_was_greedy"].append(np.nan)
                            rollout_results["car_gear_and_wheels"].append(np.nan)
                            rollout_results["meters_advanced_along_centerline"].append(
                                distance_from_start_track_to_prev_zone_transition[
                                    len(zone_centers) - misc.n_zone_centers_extrapolate_after_end_of_map
                                ]
                            )

                    # ============================
                    # END ON CP COUNT
                    # ============================
                    if self.msgtype_response_to_wakeup_TMI is None:
                        self.iface._respond_to_call(msgtype)
                elif msgtype == int(MessageType.SC_LAP_COUNT_CHANGED_SYNC):
                    self.iface._read_int32()
                    self.iface._read_int32()
                    self.iface._respond_to_call(msgtype)
                elif msgtype == int(MessageType.SC_REQUESTED_FRAME_SYNC):
                    frame = self.grab_screen()
                    frame_expected = False
                    if (
                        give_up_signal_has_been_sent
                        and this_rollout_has_seen_t_negative
                        and not this_rollout_is_finished
                        and compute_action_asap
                    ):
                        pc6 = time.perf_counter_ns()
                        time_to_grab_frame += pc6 - pc5
                        assert self.latest_tm_engine_speed_requested == 0
                        assert not compute_action_asap_floats
                        frame = np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY), 0)
                        rollout_results["frames"].append(frame)
                        pc7 = time.perf_counter_ns()
                        time_A_rgb2gray += pc7 - pc6

                        (
                            action_idx,
                            action_was_greedy,
                            q_value,
                            q_values,
                        ) = exploration_policy(rollout_results["frames"][-1], floats)

                        pc8 = time.perf_counter_ns()
                        time_exploration_policy += pc8 - pc7

                        self.request_inputs(action_idx, rollout_results)
                        self.request_speed(self.running_speed)

                        pc9 = time.perf_counter_ns()
                        time_to_iface_set_set += pc9 - pc8

                        if n_th_action_we_compute == 0:
                            end_race_stats["value_starting_frame"] = q_value
                            for i, val in enumerate(np.nditer(q_values)):
                                end_race_stats[f"q_value_{i}_starting_frame"] = val
                        rollout_results["meters_advanced_along_centerline"].append(distance_since_track_begin)
                        rollout_results["input_w"].append(misc.inputs[action_idx]["accelerate"])
                        rollout_results["actions"].append(action_idx)
                        rollout_results["action_was_greedy"].append(action_was_greedy)
                        rollout_results["car_gear_and_wheels"].append(sim_state_car_gear_and_wheels)
                        rollout_results["q_values"].append(q_values)
                        rollout_results["state_float"].append(floats)

                        compute_action_asap = False
                        n_th_action_we_compute += 1

                        time_after_iface_set_set += time.perf_counter_ns() - pc9
                    self.iface._respond_to_call(msgtype)
                elif msgtype == int(MessageType.C_SHUTDOWN):
                    self.iface.close()
                elif msgtype == int(MessageType.SC_ON_CONNECT_SYNC):
                    self.process_prepare()
                    if self.latest_map_path_requested == -1:  # Game was relaunched and must have console open
                        self.iface.execute_command("toggle_console")
                    self.request_speed(1)
                    self.iface.set_on_step_period(self.run_steps_per_action * 10)
                    self.iface.execute_command(f"set countdown_speed {self.running_speed}")
                    self.iface.execute_command(f"set autologin {'pb4608' if misc.is_pb_desktop else 'agade09'}")
                    self.iface.execute_command(f"set auto_reload_plugins false")
                    self.iface.execute_command(f"set skip_map_load_screens true")
                    self.iface.execute_command(f"cam 1")
                    self.iface.execute_command(f"set temp_save_states_collect false")
                    if self.iface.is_in_menus() and map_path != self.latest_map_path_requested:
                        print("Requested map load")
                        self.request_map(map_path)
                    self.iface._respond_to_call(msgtype)
                else:
                    pass
        except socket.timeout as err:
            print("Cutoff rollout due to TMI timeout", err)
            self.iface.close()
            end_race_stats["tmi_protection_cutoff"] = True
            self.last_rollout_crashed = True
            ensure_not_minimized(self.tm_window_id)

        return rollout_results, end_race_stats

    def process_prepare(self):
        if not misc.is_linux:
            remove_fps_cap()
            remove_map_begin_camera_zoom_in()
            # custom_resolution(misc.W_screen, misc.H_screen)
            _set_window_focus(self.tm_window_id)


if not misc.is_linux:

    def remove_fps_cap():
        # from @Kim on TrackMania Tool Assisted Discord server
        process = filter(lambda pr: pr.name() == "TmForever.exe", psutil.process_iter())
        rwm = ReadWriteMemory()
        for p in process:
            pid = int(p.pid)
            process = rwm.get_process_by_id(pid)
            process.open()
            process.write(0x005292F1, 4294919657)
            process.write(0x005292F1 + 4, 2425393407)
            process.write(0x005292F1 + 8, 2425393296)
            process.close()
            print(f"Disabled FPS cap of process {pid}")

    def remove_map_begin_camera_zoom_in():
        # from @Kim on TrackMania Tool Assisted Discord server
        process = filter(lambda p: p.name() == "TmForever.exe", psutil.process_iter())
        rwm = ReadWriteMemory()
        for p in process:
            pid = int(p.pid)
            process = rwm.get_process_by_id(pid)
            process.open()
            process.write(0x00CE8E9C, 0)
            process.close()

    def custom_resolution(width, height):  # @aijundi TMI-discord
        process = filter(lambda p: p.name() == "TmForever.exe", psutil.process_iter())
        rwm = ReadWriteMemory()
        for p in process:
            pid = int(p.pid)
            process = rwm.get_process_by_id(pid)
            process.open()
            address = process.read(0xD66FF8) + 0x60
            address = process.read(address) + 0x9C0
            process.write(address, width)
            process.write(address + 4, height)
            process.close()
