import ctypes
import math
import time

import cv2
import numpy as np
import psutil
import torch

# noinspection PyPackageRequirements
import win32gui
from ReadWriteMemory import ReadWriteMemory
from tminterface.interface import Message, MessageType, TMInterface

try:
    from . import dxshot as dxcam  # UNCOMMENT HERE TO USE DXSHOT
except:
    import dxcam
from . import misc, time_parsing
from .geometry import fraction_time_spent_in_current_zone

def _get_window_position():
    monitor_width = ctypes.windll.user32.GetSystemMetrics(0)
    trackmania_window = win32gui.FindWindow("TmForever", None)
    rect = win32gui.GetWindowRect(trackmania_window)
    clientRect = win32gui.GetClientRect(trackmania_window)  # https://stackoverflow.com/questions/51287338/python-2-7-get-ui-title-bar-size
    windowOffset = math.floor(((rect[2] - rect[0]) - clientRect[2]) / 2)
    titleOffset = ((rect[3] - rect[1]) - clientRect[3]) - windowOffset
    rect = (rect[0] + windowOffset, rect[1] + titleOffset, rect[2] - windowOffset, rect[3] - windowOffset)
    top = rect[1] + round(((rect[3] - rect[1]) - misc.H_screen) / 2)
    left = rect[0] + round(((rect[2] - rect[0]) - misc.W_screen) / 2)  # Could there be a 1 pixel error with these roundings?
    output_idx = 0
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


camera = None


def recreate_dxcam():
    global camera
    print("RECREATE")
    del camera
    create_dxcam()


def create_dxcam():
    global camera
    region, output_idx = _get_window_position()
    print(f"CREATE {region=}, {output_idx=}")
    camera = dxcam.create(output_idx=output_idx, output_color="BGRA", region=region, max_buffer_len=1)


def grab_screen():
    global camera
    try:
        return camera.grab()
    except:
        pass
    recreate_dxcam()
    return grab_screen()


create_dxcam()


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
        # self.trackmania_window = win32gui.FindWindow("TmForever", None)
        self.digits_library = time_parsing.DigitsLibrary(base_dir / "data" / "digits_file.npy")
        remove_fps_cap()
        self.zone_centers = zone_centers
        self.msgtype_response_to_wakeup_TMI = None
        self.pinned_buffer_size = (
            misc.memory_size + 100
        )  # We need some margin so we don't invalidate de the next ~n_step transitions when we overwrite images
        self.pinned_buffer = torch.empty((self.pinned_buffer_size, 1, misc.H_downsized, misc.W_downsized), dtype=torch.uint8)
        torch.cuda.cudart().cudaHostRegister(
            self.pinned_buffer.data_ptr(), self.pinned_buffer_size * misc.H_downsized * misc.W_downsized, 0
        )
        self.pinned_buffer_index = 0

    def rewind_to_state(self, state):
        msg = Message(MessageType.C_SIM_REWIND_TO_STATE)
        msg.write_buffer(state.data)
        self.iface._send_message(msg)
        self.iface._wait_for_server_response()

    def rollout(self, exploration_policy, is_eval):
        end_race_stats = {}
        zone_centers_delta = (np.random.rand(*self.zone_centers.shape) - 0.5) * misc.zone_centers_jitter
        zone_centers_delta[:, 1] *= 0.1  # Don't change the elevation
        zone_centers_delta[-(3 + misc.n_zone_centers_in_inputs) :, :] = 0  # Don't change the final zones
        if is_eval:  # TODO : zero jitter during eval round
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

        rollout_results = {
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
            "meters_advanced_along_centerline": [],
        }

        rollout_results["zone_entrance_time_ms"].append(0)  # We start the race in zone zero, and assume we just entered that zone

        if self.iface is None:
            assert self.msgtype_response_to_wakeup_TMI is None
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

        else:
            assert self.msgtype_response_to_wakeup_TMI is not None

            self.iface.set_speed(self.running_speed)
            self.latest_tm_engine_speed_requested = self.running_speed
            self.iface._respond_to_call(self.msgtype_response_to_wakeup_TMI)
            self.msgtype_response_to_wakeup_TMI = None

        assert self.iface._ensure_connected()

        compute_action_asap = False

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

        prev_zones_cumulative_distance = 0

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

                    if current_zone_idx == len(zone_centers) - 1 - misc.n_zone_centers_in_inputs:
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
                            # frame = self.camera.grab(region=trackmania_window_region)#,frame_timeout=2000)
                            frame = grab_screen()
                        parsed_time = time_parsing.parse_time(frame, self.digits_library)

                        time_to_grab_frame += time.perf_counter_ns() - pc2
                        pc2 = time.perf_counter_ns()

                        # ===================================================================================================

                        sim_state_race_time = last_known_simulation_state.race_time
                        sim_state_display_speed = last_known_simulation_state.display_speed
                        sim_state_position = np.array(
                            last_known_simulation_state.dyna.current_state.position,
                            dtype=np.float32,
                        )  # (3,)
                        sim_state_orientation = last_known_simulation_state.dyna.current_state.rotation.to_numpy()  # (3, 3)
                        sim_state_velocity = np.array(
                            last_known_simulation_state.dyna.current_state.linear_speed,
                            dtype=np.float32,
                        )  # (3,)
                        sim_state_angular_speed = np.array(
                            last_known_simulation_state.dyna.current_state.angular_speed,
                            dtype=np.float32,
                        )  # (3,)
                        sim_state_car_gear_and_wheels = np.array(
                            [
                                last_known_simulation_state.simulation_wheels[0].real_time_state.is_sliding,
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
                            ],
                            dtype=np.float32,
                        )
                        d1 = np.linalg.norm(next_zone_center - sim_state_position)
                        d2 = np.linalg.norm(current_zone_center - sim_state_position)
                        if (
                            d1 <= d2
                            and d1 <= misc.max_allowable_distance_to_checkpoint
                            and current_zone_idx < len(zone_centers) - 2 - misc.n_zone_centers_in_inputs
                            # We can never enter the final virtual zone
                        ):
                            # Move from one virtual zone to another
                            rollout_results["fraction_time_in_previous_zone"].append(
                                fraction_time_spent_in_current_zone(
                                    current_zone_center,
                                    next_zone_center,
                                    prev_sim_state_position,
                                    sim_state_position,
                                )
                            )

                            ###############
                            next_zone_center = zone_centers[1 + current_zone_idx]
                            previous_zone_center = (
                                zone_centers[current_zone_idx - 1] if current_zone_idx > 0 else (2 * zone_centers[0] - zone_centers[1])
                            )  # TODO : handle jitter
                            # TODO : don't duplicate this code with 3 lines below
                            pointA = 0.5 * (previous_zone_center + current_zone_center)
                            pointB = 0.5 * (current_zone_center + next_zone_center)
                            prev_zones_cumulative_distance += np.linalg.norm(pointB - pointA)
                            ###############

                            current_zone_idx += 1
                            rollout_results["zone_entrance_time_ms"].append(sim_state_race_time)
                            current_zone_center = zone_centers[current_zone_idx]

                        else:
                            rollout_results["fraction_time_in_previous_zone"].append(np.nan)  # Won't be used

                        rollout_results["current_zone_idx"].append(current_zone_idx)

                        next_zone_center = zone_centers[1 + current_zone_idx]
                        previous_zone_center = (
                            zone_centers[current_zone_idx - 1] if current_zone_idx > 0 else (2 * zone_centers[0] - zone_centers[1])
                        )  # TODO : handle jitter
                        # TODO : make these calculations less often
                        pointA = 0.5 * (previous_zone_center + current_zone_center)
                        pointB = 0.5 * (current_zone_center + next_zone_center)

                        dist_pointB_pointA = np.linalg.norm(pointB - pointA)
                        meters_in_current_zone = np.clip(
                            (sim_state_position - pointA).dot(pointB - pointA) / dist_pointB_pointA, 0, dist_pointB_pointA
                        )

                        # ===================================================================================================

                        time_between_grab_frame += time.perf_counter_ns() - pc2
                        pc2 = time.perf_counter_ns()

                        while parsed_time != sim_state_race_time:
                            frame = None
                            iterations += 1
                            while frame is None:
                                # frame = self.camera.grab(region=trackmania_window_region)#, frame_timeout=2000)
                                frame = grab_screen()
                            parsed_time = time_parsing.parse_time(frame, self.digits_library)

                            if iterations > 10:
                                print(f"warning capturing {iterations=}, {parsed_time=}, {sim_state_race_time=}")
                                recreate_dxcam()

                        time_to_grab_frame += time.perf_counter_ns() - pc2
                        pc2 = time.perf_counter_ns()

                        frame = torch.from_numpy(
                            np.expand_dims(
                                cv2.resize(
                                    cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY),
                                    (misc.W_downsized, misc.H_downsized),
                                    interpolation=cv2.INTER_AREA,
                                ),
                                0,
                            )
                        )
                        self.pinned_buffer[self.pinned_buffer_index].copy_(frame)
                        frame = self.pinned_buffer[self.pinned_buffer_index]
                        self.pinned_buffer_index += 1
                        if self.pinned_buffer_index >= self.pinned_buffer_size:
                            self.pinned_buffer_index = 0

                        rollout_results["frames"].append(frame)

                        time_A_rgb2gray += time.perf_counter_ns() - pc2
                        pc2 = time.perf_counter_ns()

                        prev_sim_state_position = sim_state_position

                        # ==== Construct features
                        state_zone_center_coordinates_in_car_reference_system = sim_state_orientation.T.dot(
                            (
                                zone_centers[
                                    current_zone_idx : current_zone_idx + misc.n_zone_centers_in_inputs,
                                    :,
                                ]
                                - sim_state_position
                            ).T
                        ).T  # (n_zone_centers_in_inputs, 3)
                        state_y_map_vector_in_car_reference_system = sim_state_orientation.T.dot(np.array([0, 1, 0]))
                        state_car_velocity_in_car_reference_system = sim_state_orientation.T.dot(sim_state_velocity)
                        state_car_angular_velocity_in_car_reference_system = sim_state_orientation.T.dot(sim_state_angular_speed)

                        previous_action = misc.inputs[0 if len(rollout_results["actions"]) == 0 else rollout_results["actions"][-1]]

                        time_A_geometry += time.perf_counter_ns() - pc2
                        pc2 = time.perf_counter_ns()

                        floats = np.hstack(
                            (
                                0,
                                np.array(
                                    [
                                        previous_action["accelerate"],
                                        previous_action["brake"],
                                        previous_action["left"],
                                        previous_action["right"],
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

                        self.iface.set_input_state(**misc.inputs[action_idx])
                        self.iface.set_speed(self.running_speed)

                        time_to_iface_set_set += time.perf_counter_ns() - pc2
                        pc2 = time.perf_counter_ns()

                        if n_th_action_we_compute == 0:
                            end_race_stats["value_starting_frame"] = q_value
                            for i, val in enumerate(np.nditer(q_values)):
                                end_race_stats[f"q_value_{i}_starting_frame"] = val
                        rollout_results["meters_advanced_along_centerline"].append(prev_zones_cumulative_distance + meters_in_current_zone)
                        rollout_results["display_speed"].append(sim_state_display_speed)
                        rollout_results["input_w"].append(misc.inputs[action_idx]["accelerate"])
                        rollout_results["actions"].append(action_idx)
                        rollout_results["action_was_greedy"].append(action_was_greedy)
                        rollout_results["car_position"].append(sim_state_position)
                        rollout_results["car_orientation"].append(sim_state_orientation)
                        rollout_results["car_velocity"].append(sim_state_velocity)
                        rollout_results["car_angular_speed"].append(sim_state_angular_speed)
                        rollout_results["car_gear_and_wheels"].append(sim_state_car_gear_and_wheels)
                        rollout_results["q_values"].append(q_values)

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
                    self.iface.set_timeout(misc.timeout_during_run_ms)
                    self.timeout_has_been_set = True

                if not give_up_signal_has_been_sent:
                    self.iface.give_up()
                    give_up_signal_has_been_sent = True

                if (
                    (
                        _time > self.max_overall_duration_ms
                        or _time
                        > rollout_results["zone_entrance_time_ms"][-1]
                        + self.max_minirace_duration_ms
                    )
                    and this_rollout_has_seen_t_negative
                    and not this_rollout_is_finished
                ):
                    # FAILED TO FINISH IN TIME
                    simulation_state = self.iface.get_simulation_state()
                    print(f"      --- {simulation_state.race_time:>6} ", end="")

                    end_race_stats["race_finished"] = False
                    end_race_stats["race_time"] = misc.cutoff_rollout_if_race_not_finished_within_duration_ms
                    end_race_stats["race_time_for_ratio"] = simulation_state.race_time
                    end_race_stats["n_ors_light_desynchro"] = n_ors_light_desynchro
                    end_race_stats["n_two_consecutive_frames_equal"] = n_two_consecutive_frames_equal
                    end_race_stats["n_frames_tmi_protection_triggered"] = n_frames_tmi_protection_triggered
                    end_race_stats["time_to_answer_normal_step"] = time_to_answer_normal_step / simulation_state.race_time * 50
                    end_race_stats["time_to_answer_action_step"] = time_to_answer_action_step / simulation_state.race_time * 50
                    end_race_stats["time_between_normal_on_run_steps"] = time_between_normal_on_run_steps / simulation_state.race_time * 50
                    end_race_stats["time_between_action_on_run_steps"] = time_between_action_on_run_steps / simulation_state.race_time * 50
                    end_race_stats["time_to_grab_frame"] = time_to_grab_frame / simulation_state.race_time * 50
                    end_race_stats["time_between_grab_frame"] = time_between_grab_frame / simulation_state.race_time * 50
                    end_race_stats["time_A_rgb2gray"] = time_A_rgb2gray / simulation_state.race_time * 50
                    end_race_stats["time_A_geometry"] = time_A_geometry / simulation_state.race_time * 50
                    end_race_stats["time_A_stack"] = time_A_stack / simulation_state.race_time * 50
                    end_race_stats["time_exploration_policy"] = time_exploration_policy / simulation_state.race_time * 50
                    end_race_stats["time_to_iface_set_set"] = time_to_iface_set_set / simulation_state.race_time * 50
                    end_race_stats["time_after_iface_set_set"] = time_after_iface_set_set / simulation_state.race_time * 50

                    this_rollout_is_finished = True  # FAILED TO FINISH IN TIME
                    self.msgtype_response_to_wakeup_TMI = msgtype

                    self.iface.set_timeout(misc.timeout_between_runs_ms)

                    self.rewind_to_state(simulation_state)
                    # self.iface.set_speed(0)
                    # self.latest_tm_engine_speed_requested = 0
                    do_not_exit_main_loop_before_time = time.perf_counter_ns() + 120_000_000

                if not this_rollout_is_finished:
                    this_rollout_has_seen_t_negative |= _time < 0

                    if _time == -1000:
                        # Press forward before the race starts
                        self.iface.set_timeout(misc.timeout_during_run_ms)
                        self.iface.set_input_state(**(misc.inputs[misc.action_forward_idx]))  # forward
                    elif _time >= 0 and _time % (10 * self.run_steps_per_action) == 0 and this_rollout_has_seen_t_negative:
                        last_known_simulation_state = self.iface.get_simulation_state()
                        self.rewind_to_state(last_known_simulation_state)
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
                    simulation_state.cp_data.cp_times[-1].time = -1  # Equivalent to prevent_simulation_finish()
                    self.rewind_to_state(simulation_state)

                    # self.iface.prevent_simulation_finish()  # Agade claims his trick above is better. Don't poke Agade.

                    if (
                        this_rollout_has_seen_t_negative and not this_rollout_is_finished
                    ):  # We shouldn't take into account a race finished after we ended the rollout
                        print(f"Z=({rollout_results['current_zone_idx'][-1]})", end="")
                        end_race_stats["race_finished"] = True
                        end_race_stats["race_time"] = simulation_state.race_time
                        end_race_stats["race_time_for_ratio"] = simulation_state.race_time
                        end_race_stats["n_ors_light_desynchro"] = n_ors_light_desynchro
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

                        this_rollout_is_finished = True  # SUCCESSFULLY FINISHED THE RACE
                        self.msgtype_response_to_wakeup_TMI = msgtype

                        self.iface.set_timeout(misc.timeout_between_runs_ms)

                        # self.iface.set_speed(0)
                        # self.latest_tm_engine_speed_requested = 0
                        do_not_exit_main_loop_before_time = time.perf_counter_ns() + 150_000_000
                        print(f"+++    {simulation_state.race_time:>6} ", end="")

                        if rollout_results["current_zone_idx"][-1] != len(zone_centers) - 1 - misc.n_zone_centers_in_inputs:
                            # We have not captured a frame where the car has entered our final virtual zone
                            # Let's put one in, artificially
                            assert rollout_results["current_zone_idx"][-1] == len(zone_centers) - 2 - misc.n_zone_centers_in_inputs
                            rollout_results["current_zone_idx"].append(len(zone_centers) - 1 - misc.n_zone_centers_in_inputs)
                            rollout_results["frames"].append(np.nan)
                            rollout_results["zone_entrance_time_ms"].append(simulation_state.race_time)

                            rollout_results["display_speed"].append(simulation_state.display_speed)
                            rollout_results["input_w"].append(np.nan)
                            rollout_results["actions"].append(np.nan)
                            rollout_results["action_was_greedy"].append(np.nan)
                            rollout_results["car_position"].append(np.nan)
                            rollout_results["car_orientation"].append(np.nan)
                            rollout_results["car_velocity"].append(np.nan)
                            rollout_results["car_angular_speed"].append(np.nan)
                            rollout_results["car_gear_and_wheels"].append(np.nan)
                            rollout_results["fraction_time_in_previous_zone"].append(
                                (
                                    simulation_state.race_time
                                    - (len(rollout_results["fraction_time_in_previous_zone"]) - 1) * misc.ms_per_action
                                )
                                / misc.ms_per_action
                            )

                            temp_sim_state_position = np.array(
                                last_known_simulation_state.dyna.current_state.position,
                                dtype=np.float32,
                            )  # (3,)
                            temp_sim_state_velocity = np.array(
                                last_known_simulation_state.dyna.current_state.linear_speed,
                                dtype=np.float32,
                            )
                            meters_in_current_zone = (
                                temp_sim_state_position
                                + (1 - rollout_results["fraction_time_in_previous_zone"][-1]) * temp_sim_state_velocity
                                - pointA
                            ).dot(
                                pointB - pointA
                            ) / dist_pointB_pointA  # TODO UGLYYY

                            assert meters_in_current_zone >= 0.8 * dist_pointB_pointA  # TODO : silly 0.8
                            rollout_results["meters_advanced_along_centerline"].append(
                                prev_zones_cumulative_distance + meters_in_current_zone
                            )

                            assert 0 <= rollout_results["fraction_time_in_previous_zone"][-1] <= 1

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
                self.iface.execute_command(f"set countdown_speed {self.running_speed}")
                self.iface.execute_command(f"set autologin {'pb4608' if misc.is_pb_desktop else 'agade09'}")
                self.iface.execute_command(f"set skip_map_load_screens true")
                self.iface.execute_command(f"cam 1")
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
