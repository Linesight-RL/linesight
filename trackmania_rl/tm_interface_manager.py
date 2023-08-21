import ctypes
import math
import os
import time

import cv2
import dxcam
import numpy as np
import numpy.typing as npt
import psutil
import win32.lib.win32con as win32con
import win32com.client

# noinspection PyPackageRequirements
import win32gui
from ReadWriteMemory import ReadWriteMemory
from tminterface.interface import Message, MessageType, TMInterface

from . import contact_materials, map_loader, misc, time_parsing

# DXShot: https://github.com/AI-M-BOT/DXcam/releases/tag/1.0


def _set_window_focus(
    trackmania_window,
):  # https://stackoverflow.com/questions/14295337/win32gui-setactivewindow-error-the-specified-procedure-could-not-be-found
    shell = win32com.client.Dispatch("WScript.Shell")
    shell.SendKeys("%")
    win32gui.SetForegroundWindow(trackmania_window)


def is_fullscreen(trackmania_window):
    rect = win32gui.GetWindowPlacement(trackmania_window)[4]
    return rect[0] == 0 and rect[1] == 0 and rect[2] == misc.W_screen and rect[3] == misc.H_screen


def ensure_not_minimized(trackmania_window):
    if win32gui.IsIconic(trackmania_window):  # https://stackoverflow.com/questions/54560987/restore-window-without-setting-to-foreground
        win32gui.ShowWindow(trackmania_window, win32con.SW_SHOWNORMAL)  # Unminimize window
        if is_fullscreen(trackmania_window):
            _set_window_focus(trackmania_window)


def _get_window_position(trackmania_window):
    monitor_width = ctypes.windll.user32.GetSystemMetrics(0)
    rect = win32gui.GetWindowPlacement(trackmania_window)[
        4
    ]  # Seems to be an alternative to win32gui.GetWindowRect(trackmania_window) which returns proper coordinates even for a minimized window
    top = rect[1]
    left = rect[0]
    output_idx = 0
    if not is_fullscreen(trackmania_window):
        clientRect = win32gui.GetClientRect(
            trackmania_window
        )  # https://stackoverflow.com/questions/51287338/python-2-7-get-ui-title-bar-size
        windowOffset = math.floor(((rect[2] - rect[0]) - clientRect[2]) / 2)
        titleOffset = ((rect[3] - rect[1]) - clientRect[3]) - windowOffset
        rect = (rect[0] + windowOffset, rect[1] + titleOffset, rect[2] - windowOffset, rect[3] - windowOffset)
        top = rect[1] + round(((rect[3] - rect[1]) - misc.H_screen) / 2)
        left = rect[0] + round(((rect[2] - rect[0]) - misc.W_screen) / 2)  # Could there be a 1 pixel error with these roundings?
        if left >= monitor_width:
            left -= monitor_width
            output_idx += 1
        if left < 0:
            Secondary_Width = ctypes.windll.user32.GetSystemMetrics(78) - monitor_width
            left = Secondary_Width + left
            output_idx += 1
    right = left + misc.W_screen
    bottom = top + misc.H_screen
    return (left, top, right, bottom), output_idx


def grab_screen2():
    frame = None
    while frame is None:
        frame = camera.grab(frame_timeout=500)
    return frame


class TMInterfaceCustom(TMInterface):
    def _wait_for_server_response(self, clear: bool = True):
        if self.mfile is None:
            return

        response_time = time.perf_counter()
        self.mfile.seek(0)
        while (self._read_int32() != MessageType.S_RESPONSE | 0xFF00) and time.perf_counter() - response_time < 2:
            self.mfile.seek(0)
            time.sleep(0)

        if clear:
            self._clear_buffer()


class TMInterfaceManager:
    def __init__(
        self,
        base_dir,
        running_speed=1,
        run_steps_per_action=10,
        max_overall_duration_ms=2000,
        max_minirace_duration_ms=2000,
        interface_name="TMInterface0",
    ):
        # Create TMInterface we will be using to interact with the game client
        self.iface = None
        self.set_timeout_is_done = False
        self.snapshot_before_start_is_made = False
        self.latest_tm_engine_speed_requested = 1
        self.running_speed = running_speed
        self.run_steps_per_action = run_steps_per_action
        self.max_overall_duration_ms = max_overall_duration_ms
        self.max_minirace_duration_ms = max_minirace_duration_ms
        self.timeout_has_been_set = False
        self.interface_name = interface_name
        self.digits_library = time_parsing.DigitsLibrary(base_dir / "data" / "digits_file.npy")
        self.msgtype_response_to_wakeup_TMI = None
        self.latest_map_path_requested = None
        self.last_rollout_crashed = False
        self.last_game_reboot = time.perf_counter()

    def launch_game(self):
        os.system("start .\\TMInterface.lnk")
        time.sleep(5)
        process_prepare()
        self.last_game_reboot = time.perf_counter()
        self.latest_map_path_requested = -1
        self.msgtype_response_to_wakeup_TMI = None

    def close_game(self):
        os.system("taskkill /IM TmForever.exe /f")
        time.sleep(5)

    def ensure_game_launched(self):
        if "TmForever.exe" not in (p.name() for p in psutil.process_iter()):
            if os.path.exists(".\\TMInterface.lnk"):
                print("Game not found. Restarting TMInterface.")
                self.launch_game()
            else:
                print("Game needs to be restarted but cannot be. Add TMInterface shortcut to directory.")

    def create_dxcam(self):
        global camera
        trackmania_window = win32gui.FindWindow("TmForever", None)
        ensure_not_minimized(trackmania_window)
        region, output_idx = _get_window_position(trackmania_window)
        print(f"CREATE {region=}, {output_idx=}")
        camera = dxcam.create(output_idx=output_idx, output_color="BGRA", region=region, max_buffer_len=1)

    def recreate_dxcam(self):
        global camera
        print("RECREATE")
        if "camera" in locals() or "camera" in globals():
            del camera
        try:
            self.create_dxcam()
        except Exception as e:
            print(e)
            time.sleep(1)

    def grab_screen(self):
        global camera
        while True:
            try:
                return grab_screen2()
            except:
                pass
            self.recreate_dxcam()

    def rewind_to_state(self, state):
        msg = Message(MessageType.C_SIM_REWIND_TO_STATE)
        msg.write_buffer(state.data)
        self.iface._send_message(msg)
        self.iface._wait_for_server_response()

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

    def rollout(self, exploration_policy, map_path: str, zone_centers: npt.NDArray):
        (
            zone_transitions,
            distance_between_zone_transitions,
            distance_from_start_track_to_prev_zone_transition,
            normalized_vector_along_track_axis,
        ) = map_loader.precalculate_virtual_checkpoints_information(zone_centers)

        self.ensure_game_launched()
        if time.perf_counter() - self.last_game_reboot > misc.game_reboot_interval and is_fullscreen(
            win32gui.FindWindow("TmForever", None)
        ):  # If the game is windowed, user may be using the machine and would not want the game to open and close
            self.close_game()
            self.iface = None
            self.launch_game()

        end_race_stats = {}

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

        print("S ", end="")

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
            print("Initialize connection to TMInterface ", end="")
            self.iface = TMInterfaceCustom(self.interface_name)
            self.iface.registered = False

            while not self.iface._ensure_connected():
                time.sleep(0)
                continue

            if not self.iface.registered:
                msg = Message(MessageType.C_REGISTER)
                self.iface._send_message(msg)
                self.iface._wait_for_server_response()
                self.iface.registered = True

        else:
            assert self.msgtype_response_to_wakeup_TMI is not None or self.last_rollout_crashed

            self.request_speed(self.running_speed)
            if self.msgtype_response_to_wakeup_TMI is not None:
                self.iface._respond_to_call(self.msgtype_response_to_wakeup_TMI)
                self.msgtype_response_to_wakeup_TMI = None

        assert self.iface._ensure_connected()

        if "camera" not in globals():
            self.create_dxcam()
            process_prepare()

        self.last_rollout_crashed = False

        compute_action_asap = False

        _time = -3000
        cpcount = 0
        current_zone_idx = 1

        give_up_signal_has_been_sent = False
        this_rollout_has_seen_t_negative = False
        this_rollout_is_finished = False
        n_th_action_we_compute = 0

        n_two_consecutive_frames_equal = 0
        n_frames_tmi_protection_triggered = 0

        do_not_exit_main_loop_before_time = 0
        do_not_compute_action_before_time = 0
        last_known_simulation_state = None

        prev_msgtype = 0
        time_first_message0 = time.perf_counter_ns()
        time_last_on_run_step = time.perf_counter()

        print("L ", end="")
        while not (this_rollout_is_finished and time.perf_counter_ns() > do_not_exit_main_loop_before_time):
            if not self.iface._ensure_connected():
                time.sleep(0)
                continue

            if self.iface.mfile is None:
                continue

            if time.perf_counter() - time_last_on_run_step > misc.tmi_protection_timeout_s and self.latest_tm_engine_speed_requested > 0:
                print("Cutoff rollout due to TMI timeout")
                self.iface.registered = False
                end_race_stats["tmi_protection_cutoff"] = True
                self.last_rollout_crashed = True
                ensure_not_minimized(win32gui.FindWindow("TmForever", None))
                break

            self.iface.mfile.seek(0)

            msgtype = self.iface._read_int32()

            ignore_message0 = (
                ((msgtype & 0xFF) == 0) and prev_msgtype == 0 and (time.perf_counter_ns() > time_first_message0 + 1000_000_000)
            )

            if (msgtype & 0xFF != 14) and (((msgtype & 0xFF00) == 0) or ignore_message0):
                # No message is ready, or we are spammed with message 0
                if ((msgtype & 0xFF00) != 0) and ignore_message0:
                    # We are spammed with message 0
                    time_first_message0 = time.perf_counter_ns()
                    print(
                        "TMI PROTECTION TRIGGER         TMI PROTECTION TRIGGER         TMI PROTECTION TRIGGER         TMI PROTECTION TRIGGER "
                    )
                    n_frames_tmi_protection_triggered += 1

                if (
                    compute_action_asap
                    and give_up_signal_has_been_sent
                    and this_rollout_has_seen_t_negative
                    and not this_rollout_is_finished
                    and time.perf_counter_ns() > do_not_compute_action_before_time
                ):
                    assert self.latest_tm_engine_speed_requested == 0

                    # We need to calculate a move AND we have left enough time for the set_speed(0) to have been properly applied
                    # print("Compute action")

                    pc2 = time.perf_counter_ns()

                    # ===================================================================================================

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

                    sim_state_car_gear_and_wheels = np.hstack(
                        (
                            np.array(
                                [
                                    *(ws.is_sliding for ws in wheel_state),  # Bool
                                    *(ws.has_ground_contact for ws in wheel_state),  # Bool
                                    *(ws.damper_absorb for ws in wheel_state),  # 0.005 min, 0.15 max, 0.01 typically
                                    gearbox_state,  # Bool, except 2 at startup
                                    sim_state_mobil_engine.gear,  # 0 -> 5 approx
                                    sim_state_mobil_engine.actual_rpm,  # 0-10000 approx
                                    counter_gearbox_state,  # Up to typically 28 when changing gears
                                ],
                                dtype=np.float32,
                            ),
                            *(
                                (
                                    np.arange(misc.n_contact_material_physics_behavior_types)
                                    == contact_materials.physics_behavior_fromint[ws.contact_material_id & 0xFFFF]
                                ).astype(np.float32)
                                for ws in wheel_state
                            ),
                        )
                    )
                    while True:  # Emulate do while loop
                        d1 = np.linalg.norm(zone_centers[current_zone_idx + 1] - sim_state_position)
                        d2 = np.linalg.norm(zone_centers[current_zone_idx] - sim_state_position)
                        d3 = np.linalg.norm(zone_centers[current_zone_idx - 1] - sim_state_position)
                        if current_zone_idx >= 1 and d3 < d2 and d3 <= misc.max_allowable_distance_to_checkpoint:
                            current_zone_idx -= 1
                        elif (
                            d1 <= d2
                            and d1 <= misc.max_allowable_distance_to_checkpoint
                            and current_zone_idx < len(zone_centers) - 1 - misc.n_zone_centers_extrapolate_after_end_of_map
                            # We can never enter the final virtual zone
                        ):
                            # Move from one virtual zone to another
                            current_zone_idx += 1
                            if current_zone_idx > rollout_results["furthest_zone_idx"]:
                                last_progress_improvement_ms = sim_state_race_time
                                rollout_results["furthest_zone_idx"] = current_zone_idx
                        else:
                            break

                    rollout_results["current_zone_idx"].append(current_zone_idx)

                    meters_in_current_zone = np.clip(
                        (sim_state_position - zone_transitions[current_zone_idx - 1]).dot(
                            normalized_vector_along_track_axis[current_zone_idx - 1]
                        ),
                        0,
                        distance_between_zone_transitions[current_zone_idx - 1],
                    )

                    # ===================================================================================================

                    time_between_grab_frame += time.perf_counter_ns() - pc2
                    pc2 = time.perf_counter_ns()

                    iterations = 0
                    parsed_time = None
                    frame = None

                    while parsed_time != sim_state_race_time and iterations < misc.tmi_protection_timeout_s:
                        frame = self.grab_screen()
                        iterations += 1
                        parsed_time = time_parsing.parse_time(frame, self.digits_library)
                        if iterations > 10:
                            self.recreate_dxcam()
                    if iterations >= misc.tmi_protection_timeout_s:
                        print(
                            f"Cutoff rollout due to {iterations} failed attempts to OCR {sim_state_race_time}. Got {parsed_time} instead."
                        )
                        end_race_stats["tmi_protection_cutoff"] = True
                        self.last_rollout_crashed = True
                        break

                    time_to_grab_frame += time.perf_counter_ns() - pc2
                    pc2 = time.perf_counter_ns()

                    frame = np.expand_dims(
                        cv2.resize(
                            cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY),
                            (misc.W_downsized, misc.H_downsized),
                            interpolation=cv2.INTER_AREA,
                        ),
                        0,
                    )

                    rollout_results["frames"].append(frame)

                    time_A_rgb2gray += time.perf_counter_ns() - pc2
                    pc2 = time.perf_counter_ns()

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

                    time_A_geometry += time.perf_counter_ns() - pc2
                    pc2 = time.perf_counter_ns()

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
                        )
                    ).astype(np.float32)

                    time_A_stack += time.perf_counter_ns() - pc2
                    pc2 = time.perf_counter_ns()

                    (
                        action_idx,
                        action_was_greedy,
                        q_value,
                        q_values,
                    ) = exploration_policy(rollout_results["frames"][-1], floats)

                    time_exploration_policy += time.perf_counter_ns() - pc2
                    pc2 = time.perf_counter_ns()

                    # action_idx = misc.action_forward_idx if _time < 3000 else misc.action_backward_idx
                    # action_was_greedy = True
                    # q_value = 0
                    # q_values = np.zeros(len(misc.inputs))

                    # import random
                    # action_idx = random.randint(0, 8)

                    # print("ACTION ", action_idx, " ", simulation_state.scene_mobil.input_gas)
                    self.request_inputs(action_idx, rollout_results)
                    self.request_speed(self.running_speed)

                    time_to_iface_set_set += time.perf_counter_ns() - pc2
                    pc2 = time.perf_counter_ns()

                    if n_th_action_we_compute == 0:
                        end_race_stats["value_starting_frame"] = q_value
                        for i, val in enumerate(np.nditer(q_values)):
                            end_race_stats[f"q_value_{i}_starting_frame"] = val
                    rollout_results["meters_advanced_along_centerline"].append(
                        distance_from_start_track_to_prev_zone_transition[current_zone_idx - 1] + meters_in_current_zone
                    )
                    rollout_results["input_w"].append(misc.inputs[action_idx]["accelerate"])
                    rollout_results["actions"].append(action_idx)
                    rollout_results["action_was_greedy"].append(action_was_greedy)
                    rollout_results["car_gear_and_wheels"].append(sim_state_car_gear_and_wheels)
                    rollout_results["q_values"].append(q_values)
                    rollout_results["state_float"].append(floats)

                    compute_action_asap = False
                    n_th_action_we_compute += 1

                    time_after_iface_set_set += time.perf_counter_ns() - pc2

                continue

            msgtype &= 0xFF
            self.iface._skip(4)
            # =============================================
            #        READ INCOMING MESSAGES
            # =============================================
            if msgtype == MessageType.S_SHUTDOWN:
                self.iface.close()
            elif msgtype == MessageType.S_ON_RUN_STEP:
                # print("msg_on_run_step")
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

                if not give_up_signal_has_been_sent:
                    if map_path != self.latest_map_path_requested:
                        self.request_map(map_path)
                    else:
                        self.iface.give_up()
                    give_up_signal_has_been_sent = True
                else:
                    if (
                        (_time > self.max_overall_duration_ms or _time > last_progress_improvement_ms + self.max_minirace_duration_ms)
                        and this_rollout_has_seen_t_negative
                        and not this_rollout_is_finished
                    ):
                        # FAILED TO FINISH IN TIME
                        simulation_state = self.iface.get_simulation_state()
                        print(f"      --- {simulation_state.race_time:>6} ", end="")
                        race_time = max([simulation_state.race_time, 1e-12])  # Epsilon trick to avoid division by zero

                        end_race_stats["race_finished"] = False
                        end_race_stats["race_time"] = misc.cutoff_rollout_if_race_not_finished_within_duration_ms
                        end_race_stats["race_time_for_ratio"] = race_time
                        end_race_stats["n_two_consecutive_frames_equal"] = n_two_consecutive_frames_equal
                        end_race_stats["n_frames_tmi_protection_triggered"] = n_frames_tmi_protection_triggered
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

                        self.msgtype_response_to_wakeup_TMI = msgtype
                        self.iface.set_timeout(misc.timeout_between_runs_ms)
                        self.rewind_to_state(simulation_state)
                        do_not_exit_main_loop_before_time = time.perf_counter_ns() + 120_000_000
                        this_rollout_is_finished = True

                    if not this_rollout_is_finished:
                        this_rollout_has_seen_t_negative |= _time < 0

                        if _time == -1000:
                            # Press forward before the race starts
                            self.iface.set_timeout(misc.timeout_during_run_ms)
                            self.request_inputs(misc.action_forward_idx, rollout_results)
                        elif _time >= 0 and _time % (10 * self.run_steps_per_action) == 0 and this_rollout_has_seen_t_negative:
                            last_known_simulation_state = self.iface.get_simulation_state()
                            self.rewind_to_state(last_known_simulation_state)
                            self.request_speed(0)
                            compute_action_asap = True
                            do_not_compute_action_before_time = time.perf_counter_ns() + 1_000_000
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

            elif msgtype == MessageType.S_ON_SIM_BEGIN:
                print("msg_on_sim_begin")
                self.iface._respond_to_call(msgtype)
            elif msgtype == MessageType.S_ON_SIM_STEP:
                print("msg_on_sim_step")
                _time = self.iface._read_int32()
                self.iface._respond_to_call(msgtype)
            elif msgtype == MessageType.S_ON_SIM_END:
                print("msg_on_sim_end")
                self.iface._read_int32()
                self.iface._respond_to_call(msgtype)
            elif msgtype == MessageType.S_ON_CHECKPOINT_COUNT_CHANGED:
                print("CP ", end="")
                current = self.iface._read_int32()
                target = self.iface._read_int32()
                cpcount += 1
                # ============================
                # BEGIN ON CP COUNT
                # ============================

                print(
                    f"CTNF=({current}, {target}, {this_rollout_has_seen_t_negative}, {this_rollout_is_finished})",
                    end="",
                )
                if current == target:  # Finished the race !!
                    simulation_state = self.iface.get_simulation_state()

                    cp_times_bug_handling_attempts = 0
                    while len(simulation_state.cp_data.cp_times) == 0 and cp_times_bug_handling_attempts < 5:
                        simulation_state = self.iface.get_simulation_state()
                        cp_times_bug_handling_attempts += 1
                    if len(simulation_state.cp_data.cp_times) != 0:
                        simulation_state.cp_data.cp_times[-1].time = -1  # Equivalent to prevent_simulation_finish()
                        self.rewind_to_state(simulation_state)
                    else:
                        self.iface.prevent_simulation_finish()

                    if (
                        this_rollout_has_seen_t_negative and not this_rollout_is_finished
                    ):  # We shouldn't take into account a race finished after we ended the rollout
                        print(f"Z=({rollout_results['current_zone_idx'][-1]})", end="")
                        end_race_stats["race_finished"] = True
                        end_race_stats["race_time"] = simulation_state.race_time
                        rollout_results["race_time"] = simulation_state.race_time
                        end_race_stats["race_time_for_ratio"] = simulation_state.race_time
                        end_race_stats["n_two_consecutive_frames_equal"] = n_two_consecutive_frames_equal
                        end_race_stats["n_frames_tmi_protection_triggered"] = n_frames_tmi_protection_triggered
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

                        # self.iface.set_speed(0)
                        # self.latest_tm_engine_speed_requested = 0
                        do_not_exit_main_loop_before_time = time.perf_counter_ns() + 150_000_000
                        print(f"+++    {simulation_state.race_time:>6} ", end="")

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
            elif msgtype == MessageType.S_ON_LAPS_COUNT_CHANGED:
                print("LAP ", end="")
                self.iface._read_int32()
                self.iface._respond_to_call(msgtype)
            elif msgtype == MessageType.S_ON_BRUTEFORCE_EVALUATE:
                print("msg_on_bruteforce_evaluate")
                self.iface._on_bruteforce_validate_call(msgtype)
            elif msgtype == MessageType.S_ON_REGISTERED:
                print("REGISTERED ", end="")
                self.iface.registered = True
                self.request_speed(1)
                self.iface.execute_command(f"set countdown_speed {self.running_speed}")
                self.iface.execute_command(f"set autologin {'pb4608' if misc.is_pb_desktop else 'agade09'}")
                self.iface.execute_command(f"set skip_map_load_screens true")
                self.iface.execute_command(f"cam 1")
                if self.latest_map_path_requested == -1:  # Game was relaunched and is in the main menu
                    self.iface.execute_command("toggle_console")
                    self.request_map(map_path)
                self.iface._respond_to_call(msgtype)
            elif msgtype == MessageType.S_ON_CUSTOM_COMMAND:
                print("msg_on_custom_command")
                self.iface._read_int32()
                self.iface._read_int32()
                n_args = self.iface._read_int32()
                [self.iface._read_string() for _ in range(n_args)]
                self.iface._respond_to_call(msgtype)
            elif msgtype == 0:
                if prev_msgtype != 0:
                    time_first_message0 = time.perf_counter_ns()
            else:
                pass

            prev_msgtype = msgtype
            time.sleep(0)

        print("E", end="")
        return rollout_results, end_race_stats


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


def process_prepare():
    remove_fps_cap()
    remove_map_begin_camera_zoom_in()
    # custom_resolution(misc.W_screen, misc.H_screen)
    _set_window_focus(win32gui.FindWindow("TmForever", None))
