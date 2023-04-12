import time
from collections import defaultdict

from . import dxshot as dxcam

# as dxcam #https://github.com/AI-M-BOT/DXcam/releases/tag/1.0
# import dxcam
import numpy as np
import win32com.client
import win32con

# noinspection PyPackageRequirements
import win32gui
from tminterface.interface import Message, MessageType, TMInterface

from . import misc


def rgb2gray(rgb):
    # from https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    return np.dot(rgb, [0.333, 0.333, 0.333]).astype(np.uint8)


class TMInterfaceManager:
    def __init__(self, running_speed=1, run_steps_per_action=10, max_time=2000, interface_name="TMInterface0"):
        # Create TMInterface we will be using to interact with the game client
        self.iface = None
        self.set_timeout_is_done = False
        self.snapshot_before_start_is_made = False
        self.latest_tm_engine_speed_requested = 1
        self.running_speed = running_speed
        self.run_steps_per_action = run_steps_per_action
        self.max_time = max_time
        self.timeout_has_been_set = False
        self.interface_name = interface_name

    def rollout(self, exploration_policy, stats_tracker):
        print("S ", end="")
        rv = defaultdict(list)

        trackmania_window = win32gui.FindWindow("TmForever", None)
        _set_window_position(trackmania_window)
        _set_window_focus(trackmania_window)

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

        assert self.iface._ensure_connected()

        if self.latest_tm_engine_speed_requested == 0:
            self.iface.set_speed(self.running_speed)
            self.latest_tm_engine_speed_requested = self.running_speed

        compute_action_asap = False
        camera = dxcam.create(region=_get_window_position(trackmania_window), output_color="RGB")

        # This code is extracted nearly as-is from TMInterfacePythonClient and modified to run on a single thread
        _time = -3000
        cpcount = 0
        prev_cpcount = 0
        # noinspection PyUnusedLocal
        prev_display_speed = 0
        prev_input_gas = 0

        give_up_signal_has_been_sent = False
        this_rollout_has_seen_t_negative = False
        this_rollout_is_finished = False
        n_th_action_we_compute = 0

        n_ors_light_desynchro = 0
        n_two_consecutive_frames_equal = 0
        n_frames_tmi_protection_triggered = 0

        do_not_exit_main_loop_before_time = 0
        do_not_compute_action_before_time = 0
        prev_time = -1
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

            if ((msgtype & 0xFF00) == 0) or ignore_message0:
                # No message is ready, or we are spammed with message 0
                if ((msgtype & 0xFF00) != 0) and ignore_message0:
                    # We are spammed with message 0
                    time_first_message0 = time.perf_counter_ns()
                    print("TMI PROTECTION TRIGGER         TMI PROTECTION TRIGGER         TMI PROTECTION TRIGGER ")
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
                    simulation_state = self.iface.get_simulation_state()
                    camera.grab()  # Discard one frame to make sure we are on time
                    frame = None
                    while frame is None:
                        frame = camera.grab()

                    frame = np.expand_dims(rgb2gray(frame), axis=0)  # shape = (1, 480, 640)
                    rv["frames"].append(frame)

                    if len(rv["frames"]) > 2:
                        if np.all(rv["frames"][-2] == rv["frames"][-1]):
                            # Frames have not changged
                            n_two_consecutive_frames_equal += 1

                    has_lateral_contact = (
                        simulation_state.time - (1 + misc.run_steps_per_action * 10)
                        <= simulation_state.scene_mobil.last_has_any_lateral_contact_time
                    )

                    rv["floats"].append(
                        np.array(
                            [
                                simulation_state.display_speed,
                                simulation_state.race_time,
                                # simulation_state.scene_mobil.engine.gear,
                                # simulation_state.scene_mobil.input_gas,
                                # simulation_state.scene_mobil.input_brake,
                                # simulation_state.scene_mobil.input_steer,
                                # simulation_state.simulation_wheels[0].real_time_state.is_sliding,
                                # simulation_state.simulation_wheels[1].real_time_state.is_sliding,
                                # simulation_state.simulation_wheels[2].real_time_state.is_sliding,
                                # simulation_state.simulation_wheels[3].real_time_state.is_sliding,
                                # simulation_state.simulation_wheels[0].real_time_state.has_ground_contact,
                                # simulation_state.simulation_wheels[1].real_time_state.has_ground_contact,
                                # simulation_state.simulation_wheels[2].real_time_state.has_ground_contact,
                                # simulation_state.simulation_wheels[3].real_time_state.has_ground_contact,
                                # has_lateral_contact,
                            ]
                        ).astype(np.float32)
                    )

                    # agade_reward = (
                    #         (misc.agade_speed_reward * simulation_state.display_speed)
                    #         + (misc.agade_static_penalty if simulation_state.display_speed <= 1 else 0)
                    #         + (misc.agade_w_reward if simulation_state.scene_mobil.input_gas > 0 else 0)
                    #         + misc.agade_cp_reward * (cpcount - prev_cpcount)
                    #         + misc.agade_race_finish_reward * 0
                    #         + misc.paul_constant_reward
                    # )

                    # rv["rewards"].append(
                    #     agade_reward
                    #     + misc.reward_per_input_gas * (
                    #                 misc.gamma * simulation_state.scene_mobil.input_gas - prev_input_gas)
                    #     + misc.reward_per_lateral_contact * has_lateral_contact
                    #     # misc.reward_per_tm_engine_step * self.run_steps_per_action
                    #     # + misc.reward_per_cp_passed * (misc.gamma * cpcount - prev_cpcount)
                    #     # + misc.reward_per_velocity * (misc.gamma * simulation_state.display_speed - prev_display_speed)
                    #     # + misc.bogus_reward_per_speed * simulation_state.display_speed
                    #     # + misc.bogus_reward_per_input_gas * simulation_state.scene_mobil.input_gas
                    # )
                    rv["rewards"].append(
                        misc.reward_per_tm_engine_step * self.run_steps_per_action
                        + misc.reward_shaped_velocity * (misc.gamma * simulation_state.display_speed - prev_display_speed)
                        + misc.reward_bogus_velocity * simulation_state.display_speed
                        + misc.reward_bogus_gas * simulation_state.scene_mobil.input_gas
                    )
                    rv["done"].append(False)

                    prev_cpcount = cpcount
                    # noinspection PyUnusedLocal
                    prev_display_speed = simulation_state.display_speed
                    prev_time = _time
                    prev_input_gas = simulation_state.scene_mobil.input_gas
                    action_idx, action_was_greedy, q_value, q_values = exploration_policy(rv["frames"][-1], rv["floats"][-1])

                    # action_idx = misc.action_forward_idx if _time < 2_000 else misc.action_backward_idx
                    # action_was_greedy = True

                    self.iface.set_input_state(**misc.inputs[action_idx])
                    self.iface.set_speed(self.running_speed)

                    if n_th_action_we_compute == 0:
                        stats_tracker["q_value_starting_frame"].append(q_value)
                        for i, val in enumerate(np.nditer(q_values)):
                            stats_tracker[f"q_values_starting_frame_{i}"].append(val - q_value)
                        for i, val in enumerate(np.nditer(np.sort(q_values))):
                            stats_tracker[f"gap_q_values_starting_frame_{i}"].append(val - q_value)

                    rv["actions"].append(action_idx)
                    rv["action_was_greedy"].append(action_was_greedy)

                    self.latest_tm_engine_speed_requested = self.running_speed
                    compute_action_asap = False
                    n_th_action_we_compute += 1
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
                # ============================
                # BEGIN ON RUN STEP
                # ============================

                if not self.timeout_has_been_set:
                    self.iface.set_timeout(3_000)
                    self.timeout_has_been_set = True

                if not give_up_signal_has_been_sent:
                    self.iface.give_up()
                    give_up_signal_has_been_sent = True

                if _time > self.max_time and this_rollout_has_seen_t_negative and not this_rollout_is_finished:
                    simulation_state = self.iface.get_simulation_state()
                    print(f"      --- {simulation_state.race_time:>6} ", end="")
                    has_lateral_contact = (
                        simulation_state.time - (1 + misc.run_steps_per_action * 10)
                        <= simulation_state.scene_mobil.last_has_any_lateral_contact_time
                    )

                    # agade_reward = (
                    #     (misc.agade_speed_reward * simulation_state.display_speed)
                    #     + (misc.agade_static_penalty if simulation_state.display_speed <= 1 else 0)
                    #     + (misc.agade_w_reward if simulation_state.scene_mobil.input_gas > 0 else 0)
                    #     + misc.agade_cp_reward * (cpcount - prev_cpcount)
                    #     + misc.agade_race_finish_reward * 0
                    #     + misc.paul_constant_reward
                    # )
                    # rv["rewards"].append(  # TODO
                    #     agade_reward
                    #     + misc.reward_per_input_gas * (misc.gamma * simulation_state.scene_mobil.input_gas - prev_input_gas)
                    #     + misc.reward_per_lateral_contact * has_lateral_contact
                    #     # misc.reward_per_tm_engine_step * (simulation_state.race_time - prev_time) / 10
                    #     # + misc.reward_per_cp_passed * (misc.gamma * cpcount - prev_cpcount)
                    #     # # + misc.reward_per_velocity * (misc.gamma * simulation_state.display_speed - prev_display_speed)
                    # )

                    # FAILED TO FINISH
                    rv["rewards"].append(
                        misc.reward_per_tm_engine_step * self.run_steps_per_action
                        + misc.reward_on_failed_to_finish
                        + misc.reward_shaped_velocity * (misc.gamma * misc.bogus_terminal_state_display_speed - prev_display_speed)
                        + misc.reward_bogus_velocity * simulation_state.display_speed
                        + misc.reward_bogus_gas * simulation_state.scene_mobil.input_gas
                    )
                    rv["done"].append(True)
                    stats_tracker["race_finished"].append(False)
                    stats_tracker["race_time"].append(self.max_time)
                    stats_tracker["rollout_sum_rewards"].append(
                        np.sum(np.array(rv["rewards"][1:]) * (misc.gamma ** np.linspace(0, len(rv["rewards"]) - 2, len(rv["rewards"]) - 1)))
                    )
                    stats_tracker["n_ors_light_desynchro"].append(n_ors_light_desynchro)
                    stats_tracker["n_two_consecutive_frames_equal"].append(n_two_consecutive_frames_equal)
                    stats_tracker["n_frames_tmi_protection_triggered"].append(n_frames_tmi_protection_triggered)
                    this_rollout_is_finished = True

                    self.iface.set_speed(0)
                    self.latest_tm_engine_speed_requested = 0
                    do_not_exit_main_loop_before_time = time.perf_counter_ns() + 150_000_000

                if not this_rollout_is_finished:
                    this_rollout_has_seen_t_negative |= _time < 0

                    if _time == -2000:
                        # Press forward 2000ms before the race starts
                        self.iface.set_input_state(**(misc.inputs[misc.action_forward_idx]))  # forward
                    # elif _time == -100 and not self.snapshot_before_start_is_made:
                    #     print("Save simulation state")
                    #     self.simulation_state_to_rewind_to_for_restart = self.iface.get_simulation_state()
                    #     print(f"{self.simulation_state_to_rewind_to_for_restart.scene_mobil.input_gas=}")
                    #     print(f"{self.simulation_state_to_rewind_to_for_restart.scene_mobil.input_steer=}")
                    #     print(f"{self.simulation_state_to_rewind_to_for_restart.scene_mobil.input_brake=}")
                    #     # assert self.simulation_state_to_rewind_to_for_restart.scene_mobil.input_gas == 1.0 #TODO
                    #     # assert self.simulation_state_to_rewind_to_for_restart.scene_mobil.input_steer == 0.0
                    #     # assert self.simulation_state_to_rewind_to_for_restart.scene_mobil.input_brake == 0.0
                    #     self.snapshot_before_start_is_made = True
                    elif _time >= 0 and _time % (10 * self.run_steps_per_action) == 0 and this_rollout_has_seen_t_negative:
                        # print(f"{_time=}")

                        # BEGIN AGADE TRICK - UNTESTED YET
                        msg = Message(MessageType.C_SIM_REWIND_TO_STATE)
                        msg.write_buffer(self.iface.get_simulation_state().data)
                        self.iface._send_message(msg)
                        self.iface._wait_for_server_response()
                        # END AGADE TRICK

                        self.iface.set_speed(0)
                        self.latest_tm_engine_speed_requested = 0
                        compute_action_asap = True
                        do_not_compute_action_before_time = time.perf_counter_ns() + 1_000_000
                    if (
                        _time >= 0 and this_rollout_has_seen_t_negative and self.latest_tm_engine_speed_requested == 0
                    ):  # TODO for next run : switch to elif instead of if
                        n_ors_light_desynchro += 1
                # ============================
                # END ON RUN STEP
                # ============================
                self.iface._respond_to_call(msgtype)
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
                # ============================
                # BEGIN ON CP COUNT
                # ============================

                if this_rollout_has_seen_t_negative:
                    cpcount += 1
                    if current == target:  # Finished the race !!
                        self.iface.prevent_simulation_finish()
                        if not this_rollout_is_finished:  # We shouldn't take into account a race finished after we ended the rollout
                            simulation_state = self.iface.get_simulation_state()
                            # has_lateral_contact = (
                            #     simulation_state.time - (1 + misc.run_steps_per_action * 10)
                            #     <= simulation_state.scene_mobil.last_has_any_lateral_contact_time
                            # )
                            # agade_reward = (
                            #     (misc.agade_speed_reward * simulation_state.display_speed)
                            #     + (misc.agade_static_penalty if simulation_state.display_speed <= 1 else 0)
                            #     + (misc.agade_w_reward if simulation_state.scene_mobil.input_gas > 0 else 0)
                            #     + misc.agade_cp_reward * (cpcount - prev_cpcount)
                            #     + misc.agade_race_finish_reward * 1
                            #     + misc.paul_constant_reward
                            # )
                            # rv["rewards"].append(  # TODO
                            #     agade_reward
                            #     + misc.reward_per_input_gas * (misc.gamma * simulation_state.scene_mobil.input_gas - prev_input_gas)
                            #     + misc.reward_per_lateral_contact * has_lateral_contact
                            #     # misc.reward_per_tm_engine_step * (simulation_state.race_time - prev_time) / 10
                            #     # + misc.reward_per_cp_passed * (misc.gamma * cpcount - prev_cpcount)
                            #     # # + misc.reward_per_velocity * (misc.gamma * simulation_state.display_speed - prev_display_speed)
                            # )
                            rv["rewards"].append(
                                misc.reward_per_tm_engine_step
                                * (simulation_state.race_time / misc.ms_per_run_step - len(rv["frames"]) * self.run_steps_per_action)
                                + misc.reward_on_finish
                                + misc.reward_shaped_velocity * (
                                            misc.gamma * misc.bogus_terminal_state_display_speed - prev_display_speed)
                                + misc.reward_bogus_velocity * simulation_state.display_speed
                                + misc.reward_bogus_gas * simulation_state.scene_mobil.input_gas
                            )
                            rv["done"].append(True)
                            stats_tracker["race_finished"].append(True)
                            stats_tracker["race_time"].append(simulation_state.race_time)
                            stats_tracker["rollout_sum_rewards"].append(
                                np.sum(
                                    np.array(rv["rewards"][1:])
                                    * (misc.gamma ** np.linspace(0, len(rv["rewards"]) - 2, len(rv["rewards"]) - 1))
                                )
                            )
                            stats_tracker["n_ors_light_desynchro"].append(n_ors_light_desynchro)
                            stats_tracker["n_two_consecutive_frames_equal"].append(n_two_consecutive_frames_equal)
                            stats_tracker["n_frames_tmi_protection_triggered"].append(n_frames_tmi_protection_triggered)

                            this_rollout_is_finished = True
                            self.iface.set_speed(0)
                            self.latest_tm_engine_speed_requested = 0
                            do_not_exit_main_loop_before_time = time.perf_counter_ns() + 150_000_000
                            print(f"+++    {simulation_state.race_time:>6} ", end="")
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

        # Relase the DXCam resources
        camera.release()
        camera.stop()
        del camera

        print("E", end="")
        return rv


def _set_window_position(trackmania_window):
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


def _get_window_position(trackmania_window):
    rect = win32gui.GetWindowRect(trackmania_window)
    left = rect[0] + misc.wind32gui_margins["left"]
    top = rect[1] + misc.wind32gui_margins["top"]
    right = rect[2] - misc.wind32gui_margins["right"]
    bottom = rect[3] - misc.wind32gui_margins["bottom"]
    return left, top, right, bottom
