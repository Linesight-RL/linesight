import math
import time

import numba
import numpy as np
import psutil
import win32com.client
import win32con

# noinspection PyPackageRequirements
import win32gui
from ReadWriteMemory import ReadWriteMemory
from tminterface.interface import Message, MessageType, TMInterface

# from . import dxcam
from . import dxshot as dxcam  # https://github.com/AI-M-BOT/DXcam/releases/tag/1.0
from . import misc, time_parsing
from .geometry import fraction_time_spent_in_current_zone


class TMInterfaceManager:
    def __init__(
            self,
            base_dir,
            running_speed=1,
            run_steps_per_action=10,
            max_overall_duration_ms=2000,
            max_minirace_duration_ms=2000,
            interface_name="TMInterface0",
            zone_centers=None,
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
        # int(interface_name[-1])
        self.camera = dxcam.create(output_idx=0, output_color="BGRA")
        remove_fps_cap()
        self.trackmania_window = win32gui.FindWindow("TmForever", None)
        _set_window_focus(self.trackmania_window)
        self.digits_library = time_parsing.DigitsLibrary(base_dir / "data" / "digits_file.npy")

        # if interface_name == "TMInterface0":
        #     self.zou = (1854, 37, 2494, 517)
        # elif interface_name == "TMInterface1":
        #     self.zou = (1854, 725, 2494, 1205)
        # elif interface_name == "TMInterface2":
        #     self.zou = (1174, 37, 1814, 517)
        # else:
        #     self.zou = (1174, 725, 1814, 1205)

        self.zone_centers = zone_centers

    def rollout(self, exploration_policy, stats_tracker):

        zone_centers_delta = (np.random.rand(*self.zone_centers.shape) - 0.5) * misc.zone_centers_jitter
        zone_centers_delta[:, 1] = 0 # Don't change the elevation
        zone_centers_delta[-3:, :] = 0 # Don't change the final zones
        if misc.epsilon <= 0:
            zone_centers_delta *= 0
        zone_centers = self.zone_centers + zone_centers_delta

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

        rv = {
            "current_zone_idx": [],
            "frames": [],
            "zone_entrance_time_ms": [],
            "display_speed": [],
            "input_w": [],
            "actions": [],
            "action_was_greedy": [],
            "car_position": [],
            "car_orientation": [],
            "car_velocity": [],
            "car_angular_speed": [],
            "car_gear_and_wheels": [],
            "q_values": [],
            "fraction_time_in_previous_zone": [],
        }

        rv["zone_entrance_time_ms"].append(0)  # We start the race in zone zero, and assume we just entered that zone

        if self.iface is None:
            print("Initialize connection to TMInterface ", end="")
            self.iface = TMInterface(self.interface_name)
            self.iface.registered = False

            while not self.iface._ensure_connected():
                time.sleep(0)
                continue

            if not self.iface.registered:
                msg = Message(MessageType.C_REGISTER)
                self.iface._send_message(msg)
                self.iface._wait_for_server_response()
                self.iface.registered = True

            self.iface.execute_command(f"set countdown_speed {self.running_speed}")
            self.iface.execute_command(f"set autologin pb4608")
            self.iface.execute_command(f"set set skip_map_load_screens true")

        assert self.iface._ensure_connected()

        if self.latest_tm_engine_speed_requested == 0:
            self.iface.set_speed(self.running_speed)
            self.latest_tm_engine_speed_requested = self.running_speed

        compute_action_asap = False
        trackmania_window_region = _get_window_position(self.trackmania_window)

        #
        # trackmania_window_region = (
        #     trackmania_window_region[0] - 2560,
        #     trackmania_window_region[1],
        #     trackmania_window_region[2] - 2560,
        #     trackmania_window_region[3]
        #                             )

        # This code is extracted nearly as-is from TMInterfacePythonClient and modified to run on a single thread
        _time = -3000
        cpcount = 0
        current_zone_idx = 0
        current_zone_center = zone_centers[0, :]
        next_zone_center = zone_centers[1, :]
        prev_sim_state_position = zone_centers[0, :]

        give_up_signal_has_been_sent = False
        this_rollout_has_seen_t_negative = False
        this_rollout_is_finished = False
        n_th_action_we_compute = 0

        n_ors_light_desynchro = 0
        n_two_consecutive_frames_equal = 0
        n_frames_tmi_protection_triggered = 0

        do_not_exit_main_loop_before_time = 0
        do_not_compute_action_before_time = 0
        last_known_simulation_state = None

        prev_msgtype = 0
        time_first_message0 = time.perf_counter_ns()

        print("L ", end="")
        while not (this_rollout_is_finished and time.perf_counter_ns() > do_not_exit_main_loop_before_time):
            if not self.iface._ensure_connected():
                time.sleep(0)
                continue

            if self.iface.mfile is None:
                continue

            self.iface.mfile.seek(0)

            msgtype = self.iface._read_int32()
            ignore_message0 = (
                    ((msgtype & 0xFF) == 0) and prev_msgtype == 0 and (
                    time.perf_counter_ns() > time_first_message0 + 1000_000_000)
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

                    if current_zone_idx == len(zone_centers) - 1:
                        # This might happen if the car enters my last virtual zone, but has not finished the race yet.
                        # Just press forward and do not record any experience
                        self.iface.set_input_state(**misc.inputs[misc.action_forward_idx])
                        self.iface.set_speed(self.running_speed)
                    else:
                        # ===================================================================================================

                        pc2 = time.perf_counter_ns()

                        iterations = 0
                        frame = None
                        while frame is None:
                            frame = self.camera.grab(region=trackmania_window_region)
                        parsed_time = time_parsing.parse_time(frame, self.digits_library)

                        time_to_grab_frame += time.perf_counter_ns() - pc2
                        pc2 = time.perf_counter_ns()

                        # ===================================================================================================

                        sim_state_race_time = last_known_simulation_state.race_time
                        sim_state_display_speed = last_known_simulation_state.display_speed
                        sim_state_position = np.array(last_known_simulation_state.dyna.current_state.position,
                                                      dtype=np.float32)  # (3,)
                        sim_state_orientation = last_known_simulation_state.dyna.current_state.rotation.to_numpy()  # (3, 3)
                        sim_state_velocity = np.array(last_known_simulation_state.dyna.current_state.linear_speed,
                                                      dtype=np.float32)  # (3,)
                        sim_state_angular_speed = np.array(last_known_simulation_state.dyna.current_state.angular_speed,
                                                           dtype=np.float32)  # (3,)
                        sim_state_car_gear_and_wheels = np.array(
                            [last_known_simulation_state.simulation_wheels[0].real_time_state.is_sliding,
                             # Bool
                             last_known_simulation_state.simulation_wheels[1].real_time_state.is_sliding,
                             # Bool
                             last_known_simulation_state.simulation_wheels[2].real_time_state.is_sliding,
                             # Bool
                             last_known_simulation_state.simulation_wheels[3].real_time_state.is_sliding,
                             # Bool
                             last_known_simulation_state.scene_mobil.gearbox_state,  # Bool
                             last_known_simulation_state.scene_mobil.engine.gear,  # 0 -> 5 approx
                             last_known_simulation_state.scene_mobil.engine.actual_rpm
                             # 0-10000 aoorox
                             ], dtype=np.float32)
                        d1 = np.linalg.norm(next_zone_center - sim_state_position)
                        d2 = np.linalg.norm(
                            current_zone_center - sim_state_position
                        )
                        if d1 <= d2 and d1 <= misc.max_allowable_distance_to_checkpoint:
                            # Move from one virtual zone to another
                            rv["fraction_time_in_previous_zone"].append(
                                fraction_time_spent_in_current_zone(
                                    current_zone_center, next_zone_center, prev_sim_state_position, sim_state_position
                                )
                            )
                            current_zone_idx += 1
                            rv["zone_entrance_time_ms"].append(sim_state_race_time)
                            current_zone_center = zone_centers[current_zone_idx]

                        else:
                            rv["fraction_time_in_previous_zone"].append(np.nan)  # Won't be used

                        rv["current_zone_idx"].append(current_zone_idx)

                        if current_zone_idx == len(zone_centers) - 1:
                            rv["frames"].append(np.nan)
                            rv["display_speed"].append(sim_state_display_speed)
                            rv["input_w"].append(np.nan)
                            rv["actions"].append(np.nan)
                            rv["action_was_greedy"].append(np.nan)
                            rv["car_position"].append(np.nan)
                            rv["car_orientation"].append(np.nan)
                            rv["car_velocity"].append(np.nan)
                            rv["car_angular_speed"].append(np.nan)
                            rv["car_gear_and_wheels"].append(np.nan)

                            assert 0 <= rv["fraction_time_in_previous_zone"][-1] <= 1

                            stats_tracker["race_finished"].append(True)
                            stats_tracker["race_time"].append(sim_state_race_time)
                            stats_tracker["race_time_for_ratio"].append(sim_state_race_time)
                            stats_tracker["n_ors_light_desynchro"].append(n_ors_light_desynchro)
                            stats_tracker["n_two_consecutive_frames_equal"].append(n_two_consecutive_frames_equal)
                            stats_tracker["n_frames_tmi_protection_triggered"].append(n_frames_tmi_protection_triggered)

                            this_rollout_is_finished = True
                            assert self.latest_tm_engine_speed_requested == 0
                            do_not_exit_main_loop_before_time = time.perf_counter_ns() + 150_000_000
                            print(f"+V+    {sim_state_race_time:>6} ", end="")
                        else:

                            next_zone_center = zone_centers[1 + current_zone_idx]
                            # ===================================================================================================

                            time_between_grab_frame += time.perf_counter_ns() - pc2
                            pc2 = time.perf_counter_ns()

                            while parsed_time != sim_state_race_time:
                                frame = None
                                iterations += 1
                                while frame is None:
                                    frame = self.camera.grab(region=trackmania_window_region)
                                parsed_time = time_parsing.parse_time(frame, self.digits_library)

                                if iterations > 10:
                                    print(f"warning capturing {iterations=}, {parsed_time=}, {sim_state_race_time=}")

                            time_to_grab_frame += time.perf_counter_ns() - pc2
                            pc2 = time.perf_counter_ns()

                            rv["frames"].append(rgb2gray(frame))  # shape = (1, 480, 640)

                            time_A_rgb2gray += time.perf_counter_ns() - pc2
                            pc2 = time.perf_counter_ns()

                            prev_sim_state_position = sim_state_position

                            # ==== Construct features
                            first_zone_idx_in_input = min(current_zone_idx,
                                                          len(zone_centers) - misc.n_checkpoints_in_inputs)
                            time_mini_race_start_ms = rv["zone_entrance_time_ms"][first_zone_idx_in_input]
                            current_overall_time_ms = sim_state_race_time  # TODO CHECK IF OFF BY ONE
                            mini_race_duration_ms = current_overall_time_ms - time_mini_race_start_ms

                            state_zone_center_coordinates_in_car_reference_system = sim_state_orientation.T.dot(
                                (
                                        zone_centers[
                                        first_zone_idx_in_input: first_zone_idx_in_input + misc.n_checkpoints_in_inputs,
                                        :]
                                        - sim_state_position
                                ).T
                            ).T  # (n_checkpoints_in_inputs, 3)
                            state_y_map_vector_in_car_reference_system = sim_state_orientation.T.dot(
                                np.array([0, 1, 0]))
                            state_car_velocity_in_car_reference_system = sim_state_orientation.T.dot(sim_state_velocity)
                            state_car_angular_velocity_in_car_reference_system = sim_state_orientation.T.dot(
                                sim_state_angular_speed)

                            previous_action = misc.inputs[0 if len(rv["actions"]) == 0 else rv["actions"][-1]]

                            time_A_geometry += time.perf_counter_ns() - pc2
                            pc2 = time.perf_counter_ns()

                            floats = np.hstack(
                                (
                                    mini_race_duration_ms,
                                    np.array([previous_action['accelerate'], previous_action['brake'],
                                              previous_action['left'],
                                              previous_action['right']]),  # NEW
                                    sim_state_car_gear_and_wheels.ravel(),  # NEW
                                    state_car_angular_velocity_in_car_reference_system.ravel(),  # NEW
                                    state_car_velocity_in_car_reference_system.ravel(),
                                    state_y_map_vector_in_car_reference_system.ravel(),
                                    state_zone_center_coordinates_in_car_reference_system.ravel(),
                                    current_zone_idx
                                    >= np.arange(first_zone_idx_in_input + 1,
                                                 first_zone_idx_in_input + misc.n_checkpoints_in_inputs - 1),
                                )
                            ).astype(np.float32)

                            time_A_stack += time.perf_counter_ns() - pc2
                            pc2 = time.perf_counter_ns()

                            action_idx, action_was_greedy, q_value, q_values = exploration_policy(rv["frames"][-1],
                                                                                                  floats)

                            time_exploration_policy += time.perf_counter_ns() - pc2
                            pc2 = time.perf_counter_ns()

                            # action_idx = misc.action_forward_idx if _time < 100000000 else misc.action_backward_idx
                            # action_was_greedy = True
                            # q_value = 0
                            # q_values = 0

                            # import random
                            # action_idx = random.randint(0, 8)

                            # print("ACTION ", action_idx, " ", simulation_state.scene_mobil.input_gas)

                            self.iface.set_input_state(**misc.inputs[action_idx])
                            self.iface.set_speed(self.running_speed)

                            time_to_iface_set_set += time.perf_counter_ns() - pc2
                            pc2 = time.perf_counter_ns()

                            if n_th_action_we_compute == 0:
                                stats_tracker["value_starting_frame"].append(q_value)
                                for i, val in enumerate(np.nditer(q_values)):
                                    stats_tracker[f"q_value_{i}_starting_frame"].append(val)

                            rv["display_speed"].append(sim_state_display_speed)
                            rv["input_w"].append(misc.inputs[action_idx]["accelerate"])
                            rv["actions"].append(action_idx)
                            rv["action_was_greedy"].append(action_was_greedy)
                            rv["car_position"].append(sim_state_position)
                            rv["car_orientation"].append(sim_state_orientation)
                            rv["car_velocity"].append(sim_state_velocity)
                            rv["car_angular_speed"].append(sim_state_angular_speed)
                            rv["car_gear_and_wheels"].append(sim_state_car_gear_and_wheels)
                            rv["q_values"].append(q_values)

                            self.latest_tm_engine_speed_requested = self.running_speed
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
                    self.iface.set_timeout(7_000)
                    self.timeout_has_been_set = True

                if not give_up_signal_has_been_sent:
                    self.iface.give_up()
                    give_up_signal_has_been_sent = True

                if (
                        (
                                _time > self.max_overall_duration_ms
                                or _time
                                > rv["zone_entrance_time_ms"][
                                    max(0, current_zone_idx + 2 - misc.n_checkpoints_in_inputs)]
                                + self.max_minirace_duration_ms
                        )
                        and this_rollout_has_seen_t_negative
                        and not this_rollout_is_finished
                ):
                    # FAILED TO FINISH IN TIME
                    simulation_state = self.iface.get_simulation_state()
                    print(f"      --- {simulation_state.race_time:>6} ", end="")

                    # has_lateral_contact = (
                    #     simulation_state.time - (1 + misc.run_steps_per_action * 10)
                    #     <= simulation_state.scene_mobil.last_has_any_lateral_contact_time
                    # )

                    # FAILED TO FINISH
                    stats_tracker["race_finished"].append(False)
                    stats_tracker["race_time"].append(misc.max_overall_duration_ms)
                    stats_tracker["race_time_for_ratio"].append(simulation_state.race_time)
                    stats_tracker["n_ors_light_desynchro"].append(n_ors_light_desynchro)
                    stats_tracker["n_two_consecutive_frames_equal"].append(n_two_consecutive_frames_equal)
                    stats_tracker["n_frames_tmi_protection_triggered"].append(n_frames_tmi_protection_triggered)
                    stats_tracker["time_to_answer_normal_step"].append(
                        time_to_answer_normal_step / simulation_state.race_time * 50)
                    stats_tracker["time_to_answer_action_step"].append(
                        time_to_answer_action_step / simulation_state.race_time * 50)
                    stats_tracker["time_between_normal_on_run_steps"].append(
                        time_between_normal_on_run_steps / simulation_state.race_time * 50)
                    stats_tracker["time_between_action_on_run_steps"].append(
                        time_between_action_on_run_steps / simulation_state.race_time * 50)

                    stats_tracker["time_to_grab_frame"].append(time_to_grab_frame / simulation_state.race_time * 50)
                    stats_tracker["time_between_grab_frame"].append(
                        time_between_grab_frame / simulation_state.race_time * 50)
                    stats_tracker["time_A_rgb2gray"].append(
                        time_A_rgb2gray / simulation_state.race_time * 50)
                    stats_tracker["time_A_geometry"].append(
                        time_A_geometry / simulation_state.race_time * 50)
                    stats_tracker["time_A_stack"].append(
                        time_A_stack / simulation_state.race_time * 50)
                    stats_tracker["time_exploration_policy"].append(
                        time_exploration_policy / simulation_state.race_time * 50)
                    stats_tracker["time_to_iface_set_set"].append(
                        time_to_iface_set_set / simulation_state.race_time * 50)
                    stats_tracker["time_after_iface_set_set"].append(
                        time_after_iface_set_set / simulation_state.race_time * 50)

                    this_rollout_is_finished = True

                    self.iface.set_speed(0)
                    self.latest_tm_engine_speed_requested = 0
                    do_not_exit_main_loop_before_time = time.perf_counter_ns() + 120_000_000

                if not this_rollout_is_finished:
                    this_rollout_has_seen_t_negative |= _time < 0

                    if _time == -10:
                        # Press forward before the race starts
                        self.iface.set_input_state(**(misc.inputs[misc.action_forward_idx]))  # forward
                    elif _time >= 0 and _time % (
                            10 * self.run_steps_per_action) == 0 and this_rollout_has_seen_t_negative:

                        # BEGIN AGADE TRICK
                        msg = Message(MessageType.C_SIM_REWIND_TO_STATE)
                        last_known_simulation_state = self.iface.get_simulation_state()
                        msg.write_buffer(last_known_simulation_state.data)
                        self.iface._send_message(msg)
                        self.iface._wait_for_server_response()
                        # END AGADE TRICK

                        self.iface.set_speed(0)
                        self.latest_tm_engine_speed_requested = 0
                        compute_action_asap = True
                        do_not_compute_action_before_time = time.perf_counter_ns() + 1_000_000

                    elif (
                            _time >= 0 and this_rollout_has_seen_t_negative and self.latest_tm_engine_speed_requested == 0
                    ):  # TODO for next run : switch to elif instead of if
                        n_ors_light_desynchro += 1
                # ============================
                # END ON RUN STEP
                # ============================
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

                print(f"CTNF=({current}, {target}, {this_rollout_has_seen_t_negative}, {this_rollout_is_finished})",
                      end='')
                if current == target:  # Finished the race !!

                    #         S
                    #         L
                    #         CP
                    #         CTNF = (1, 11, True, False)
                    #         CP
                    #         CTNF = (2, 11, True, False)
                    #         CP
                    #         CTNF = (3, 11, True, False)
                    #         CP
                    #         CTNF = (4, 11, True, False)
                    #         CP
                    #         CTNF = (5, 11, True, False)
                    #         CP
                    #         CTNF = (6, 11, True, False)
                    #         CP
                    #         CTNF = (7, 11, True, False)
                    #         CP
                    #         CTNF = (8, 11, True, False)
                    #         CP
                    #         CTNF = (9, 11, True, False)
                    #         CP
                    #         CTNF = (10, 11, True, False)
                    #         CP
                    #         CTNF = (11, 11, True, False)
                    #         Traceback(most
                    #         recent
                    #         call
                    #         last):
                    #         File
                    #         "C:\Users\chopi\projects\trackmania_rl\scripts\train.py", line
                    #         218, in < module >
                    #         rollout_results = tmi.rollout(
                    #             File
                    #         "c:\users\chopi\projects\trackmania_rl\trackmania_rl\tm_interface_manager.py", line
                    #         426, in rollout
                    #         simulation_state.cp_data.cp_times[-1].time = -1  # Equivalent to prevent_simulation_finish()
                    #         File
                    #         "C:\Users\chopi\tools\mambaforge\envs\tm309\lib\site-packages\bytefield\array_proxy.py", line
                    #         95, in __getitem__
                    #         index = self._validate_index(index)
                    #         File
                    #         "C:\Users\chopi\tools\mambaforge\envs\tm309\lib\site-packages\bytefield\array_proxy.py", line
                    #         127, in _validate_index
                    #         raise IndexError(f'index {user_index} is out of bounds for shape {self.shape}')
                    # IndexError: index(-1, ) is out
                    # of
                    # bounds
                    # for shape(0, )
                    #
                    # (tm309)
                    # C:\Users\chopi\projects\trackmania_rl\scripts >

                    # # BEGIN AGADE TRICK
                    # msg = Message(MessageType.C_SIM_REWIND_TO_STATE)
                    # simulation_state = self.iface.get_simulation_state()
                    # simulation_state.cp_data.cp_times[-1].time = -1  # Equivalent to prevent_simulation_finish()
                    # msg.write_buffer(simulation_state.data)
                    # self.iface._send_message(msg)
                    # self.iface._wait_for_server_response()
                    # # END AGADE TRICK

                    self.iface.prevent_simulation_finish()  # Agade claims his trick above is better. Don't poke Agade.

                    if this_rollout_has_seen_t_negative and not this_rollout_is_finished:  # We shouldn't take into account a race finished after we ended the rollout
                        print(
                            f"Z=({rv['current_zone_idx'][-1]})", end='')
                        simulation_state = self.iface.get_simulation_state()
                        stats_tracker["race_finished"].append(True)
                        stats_tracker["race_time"].append(simulation_state.race_time)
                        stats_tracker["race_time_for_ratio"].append(simulation_state.race_time)
                        stats_tracker["n_ors_light_desynchro"].append(n_ors_light_desynchro)
                        stats_tracker["n_two_consecutive_frames_equal"].append(n_two_consecutive_frames_equal)
                        stats_tracker["n_frames_tmi_protection_triggered"].append(n_frames_tmi_protection_triggered)
                        stats_tracker["time_to_answer_normal_step"].append(
                            time_to_answer_normal_step / simulation_state.race_time * 50)
                        stats_tracker["time_to_answer_action_step"].append(
                            time_to_answer_action_step / simulation_state.race_time * 50)
                        stats_tracker["time_between_normal_on_run_steps"].append(
                            time_between_normal_on_run_steps / simulation_state.race_time * 50)
                        stats_tracker["time_between_action_on_run_steps"].append(
                            time_between_action_on_run_steps / simulation_state.race_time * 50)
                        stats_tracker["time_to_grab_frame"].append(time_to_grab_frame / simulation_state.race_time * 50)
                        stats_tracker["time_between_grab_frame"].append(
                            time_between_grab_frame / simulation_state.race_time * 50)
                        stats_tracker["time_A_rgb2gray"].append(
                            time_A_rgb2gray / simulation_state.race_time * 50)
                        stats_tracker["time_A_geometry"].append(
                            time_A_geometry / simulation_state.race_time * 50)
                        stats_tracker["time_A_stack"].append(
                            time_A_stack / simulation_state.race_time * 50)
                        stats_tracker["time_exploration_policy"].append(
                            time_exploration_policy / simulation_state.race_time * 50)
                        stats_tracker["time_to_iface_set_set"].append(
                            time_to_iface_set_set / simulation_state.race_time * 50)
                        stats_tracker["time_after_iface_set_set"].append(
                            time_after_iface_set_set / simulation_state.race_time * 50)

                        this_rollout_is_finished = True
                        self.iface.set_speed(0)
                        self.latest_tm_engine_speed_requested = 0
                        do_not_exit_main_loop_before_time = time.perf_counter_ns() + 150_000_000
                        print(f"+++    {simulation_state.race_time:>6} ", end="")

                        if rv["current_zone_idx"][-1] != len(zone_centers) - 1:
                            # We have not captured a frame where the car has entered our final virtual zone
                            # Let's put one in, artificially
                            assert rv["current_zone_idx"][-1] == len(zone_centers) - 2
                            rv["current_zone_idx"].append(len(zone_centers) - 1)
                            rv["frames"].append(np.nan)
                            rv["zone_entrance_time_ms"].append(simulation_state.race_time)
                            rv["display_speed"].append(simulation_state.display_speed)
                            rv["input_w"].append(np.nan)
                            rv["actions"].append(np.nan)
                            rv["action_was_greedy"].append(np.nan)
                            rv["car_position"].append(np.nan)
                            rv["car_orientation"].append(np.nan)
                            rv["car_velocity"].append(np.nan)
                            rv["car_angular_speed"].append(np.nan)
                            rv["car_gear_and_wheels"].append(np.nan)
                            rv["fraction_time_in_previous_zone"].append(
                                (simulation_state.race_time - (
                                        len(rv["fraction_time_in_previous_zone"]) - 1) * misc.ms_per_action)
                                / misc.ms_per_action
                            )
                            assert 0 <= rv["fraction_time_in_previous_zone"][-1] <= 1

                # ============================
                # END ON CP COUNT
                # ============================
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
                self.iface._respond_to_call(msgtype)
            elif msgtype == MessageType.S_ON_CUSTOM_COMMAND:
                print("msg_on_custom_command")
                self.iface._read_int32()
                self.iface._read_int32()
                self.iface._read_int32()
                self.iface._read_string()
                self.iface._respond_to_call(msgtype)
            elif msgtype == 0:
                if prev_msgtype != 0:
                    time_first_message0 = time.perf_counter_ns()
            else:
                pass

            prev_msgtype = msgtype
            time.sleep(0)

        assert self.latest_tm_engine_speed_requested == 0

        print("E", end="")
        return rv


def _set_window_position(trackmania_window):
    # Currently unused, might be used in the future for parallel environments
    win32gui.SetWindowPos(
        trackmania_window,
        win32con.HWND_TOPMOST,
        2560 - 654,
        120,
        misc.W + misc.wind32gui_margins["left"] + misc.wind32gui_margins["right"],
        misc.H + misc.wind32gui_margins["top"] + misc.wind32gui_margins["bottom"],
        0,
    )


def _set_window_focus(trackmania_window):
    shell = win32com.client.Dispatch("WScript.Shell")
    shell.SendKeys("%")
    win32gui.SetForegroundWindow(trackmania_window)


def pb__get_window_position(trackmania_window):
    rect = win32gui.GetWindowRect(trackmania_window)
    left = rect[0] + misc.wind32gui_margins["left"]
    top = rect[1] + misc.wind32gui_margins["top"]
    right = rect[2] - misc.wind32gui_margins["right"]
    bottom = rect[3] - misc.wind32gui_margins["bottom"]
    return left, top, right, bottom


def _get_window_position(trackmania_window):
    rect = win32gui.GetWindowRect(trackmania_window)
    clientRect = win32gui.GetClientRect(
        trackmania_window)  # https://stackoverflow.com/questions/51287338/python-2-7-get-ui-title-bar-size
    windowOffset = math.floor(((rect[2] - rect[0]) - clientRect[2]) / 2)
    titleOffset = ((rect[3] - rect[1]) - clientRect[3]) - windowOffset
    rect = (rect[0] + windowOffset, rect[1] + titleOffset, rect[2] - windowOffset, rect[3] - windowOffset)
    top = rect[1] + round(((rect[3] - rect[1]) - misc.H) / 2)
    left = rect[0] + round(((rect[2] - rect[0]) - misc.W) / 2)  # Could there be a 1 pixel error with these roundings?
    right = left + misc.W
    bottom = top + misc.H
    return left, top, right, bottom


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


@numba.njit(fastmath=True)
def rgb2gray(img):
    img_bnw = np.empty(shape=(1, misc.H, misc.W), dtype=np.uint8)
    for i in range(misc.H):
        for j in range(misc.W):
            img_bnw[0, i, j] = np.uint8(round(np.sum(img[i, j, :3], dtype=np.float32) / 3))
    return img_bnw


@numba.njit(fastmath=True)
def frames_equal(img1, img2):
    return np.all(img1 == img2)
