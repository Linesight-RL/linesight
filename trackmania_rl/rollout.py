import random
import time
from collections import defaultdict

import dxcam
import numpy as np
import win32com.client
import win32con
import win32gui
from tminterface.interface import Message, MessageType, TMInterface

from . import misc
import pydirectinput

keypress = lambda x, times: pydirectinput.press(x, presses=times, _pause=False, interval=0.002)


def rgb2gray(rgb):
    # from https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    return np.dot(rgb, [0.2989, 0.5870, 0.1140]).astype(np.uint8)


class TMInterfaceManager:
    def __init__(self):
        # Create TMInterface we will be using to interact with the game client
        self.iface = None
        self.set_timeout_is_done = False
        self.snapshot_before_start_is_made = False

    def rollout(self, *, running_speed=1, run_steps_per_action=10, max_time=2000, exploration_policy):
        print("Start rollout")
        rv = defaultdict(list)

        trackmania_window = win32gui.FindWindow("TmForever", None)
        _set_window_position(trackmania_window)
        _set_window_focus(trackmania_window)

        if self.iface is None:
            print("Initialize connection to TMInterface")
            self.iface = TMInterface()
            self.iface.registered = False

            while not self.iface._ensure_connected():
                time.sleep(0)
                continue

            if not self.iface.registered:
                msg = Message(MessageType.C_REGISTER)
                self.iface._send_message(msg)
                self.iface._wait_for_server_response()
                self.iface.registered = True

        time_to_do_next_thing = time.perf_counter_ns() + 15_000_000
        time_asked_compute_action_asap = time.perf_counter_ns()
        time_for_next_rollout_restart = time.perf_counter_ns()
        pause_end_of_rollout_asap = False
        time_asked_pause_end_of_rollout_asap = time.perf_counter_ns()

        pause_end_of_rollout_sent = False
        time_pause_end_of_rollout_sent = time.perf_counter_ns()

        try_n_rollout_restarts = 5
        this_rollout_initial_set_speed_is_done = False

        this_rollout_has_seen_t_negative = False

        compute_action_asap = False
        camera = dxcam.create(region=_get_window_position(trackmania_window), output_color="RGB")

        print("-")
        # This code is extracted nearly as-is from TMInterfacePythonClient and modified to run on a single thread
        _time = -3000
        cpcount = 0
        prev_cpcount = 0
        prev_display_speed = 0
        prev_input_gas = 0

        print("Start loop")
        while not (
            pause_end_of_rollout_sent and (time.perf_counter_ns() - time_pause_end_of_rollout_sent > 200_000_000)
        ):
            if not self.iface._ensure_connected():
                time.sleep(0)
                continue

            if self.iface.mfile is None:
                print("None")
                continue

            self.iface.mfile.seek(0)
            msgtype = self.iface._read_int32()

            if msgtype & 0xFF00 == 0:
                # No message received
                # Let's see if we want to send one
                if not self.set_timeout_is_done and time.perf_counter_ns() > time_to_do_next_thing:
                    # print("Set timeout : 6 seconds")
                    self.iface.set_timeout(6_000)
                    self.set_timeout_is_done = True
                    time_to_do_next_thing = (
                        time.perf_counter_ns() + 15_000_000
                    )  # should be rollout_initial_set_speed_is_done

                elif (not this_rollout_initial_set_speed_is_done) and (time.perf_counter_ns() > time_to_do_next_thing):
                    print("Wake up TMI with set_speed")
                    self.iface.set_speed(running_speed)
                    this_rollout_initial_set_speed_is_done = True
                    time_for_next_rollout_restart = time.perf_counter_ns() + 15_000_000  # should be try a restart

                elif (
                    not this_rollout_has_seen_t_negative
                    and try_n_rollout_restarts > 0
                    and this_rollout_initial_set_speed_is_done
                    and (time.perf_counter_ns() > time_for_next_rollout_restart)
                ):
                    print("Keypress to restart", try_n_rollout_restarts)
                    _set_window_position(trackmania_window)
                    keypress("enter", 3)
                    keypress("del", 1)
                    try_n_rollout_restarts -= 1
                    time_for_next_rollout_restart = time.perf_counter_ns() + 2_000_000_000

                elif (
                    compute_action_asap
                    and not pause_end_of_rollout_asap
                    and not pause_end_of_rollout_sent
                    and this_rollout_has_seen_t_negative
                    and (time.perf_counter_ns() - time_asked_compute_action_asap) > 1_000_000
                ):
                    # We need to calculate a move AND we have left enough time for the set_speed(0) to have been properly applied
                    print("Compute action")
                    simulation_state = self.iface.get_simulation_state()
                    frame = camera.grab()
                    while frame is None:
                        frame = camera.grab()

                    # if frame is not None:
                    frame = np.expand_dims(rgb2gray(frame), axis=0)
                    rv["frames"].append(frame)

                    rv["floats"].append(
                        np.array(
                            [
                                simulation_state.display_speed,
                                simulation_state.race_time,
                                simulation_state.scene_mobil.engine.gear,
                                simulation_state.scene_mobil.input_gas,
                                simulation_state.scene_mobil.input_brake,
                                simulation_state.scene_mobil.input_steer,
                                simulation_state.simulation_wheels[0].real_time_state.is_sliding,
                                simulation_state.simulation_wheels[1].real_time_state.is_sliding,
                                simulation_state.simulation_wheels[2].real_time_state.is_sliding,
                                simulation_state.simulation_wheels[3].real_time_state.is_sliding,
                                simulation_state.simulation_wheels[0].real_time_state.has_ground_contact,
                                simulation_state.simulation_wheels[1].real_time_state.has_ground_contact,
                                simulation_state.simulation_wheels[2].real_time_state.has_ground_contact,
                                simulation_state.simulation_wheels[3].real_time_state.has_ground_contact,
                            ]
                        )
                    )
                    rv["rewards"].append(
                        misc.reward_per_tm_engine_step * run_steps_per_action
                        + misc.reward_per_cp_passed * (misc.gamma * cpcount - prev_cpcount)
                        + misc.reward_per_velocity * (misc.gamma * simulation_state.display_speed - prev_display_speed)
                        + misc.reward_per_input_gas
                        * (misc.gamma * simulation_state.scene_mobil.input_gas - prev_input_gas)
                        + misc.bogus_reward_per_speed * simulation_state.display_speed
                        + misc.bogus_reward_per_input_gas * simulation_state.scene_mobil.input_gas
                    )
                    rv["simstates"].append(simulation_state)
                    rv["done"].append(False)
                    prev_cpcount = cpcount
                    prev_display_speed = simulation_state.display_speed
                    prev_time = _time
                    prev_input_gas = simulation_state.scene_mobil.input_gas
                    # time.sleep(0.01)  # Arbitrary time to calculate a move
                    action_idx, action_was_greedy = exploration_policy(rv["frames"][-1], rv["floats"][-1])

                    # action_idx = misc.action_forward_idx if _time < 2_000 else misc.action_backward_idx
                    # action_was_greedy = True

                    self.iface.set_input_state(**misc.inputs[action_idx])
                    rv["actions"].append(action_idx)
                    rv["action_was_greedy"].append(action_was_greedy)
                    self.iface.set_speed(running_speed)
                    compute_action_asap = False

                elif pause_end_of_rollout_asap and (time.perf_counter_ns() - time_asked_pause_end_of_rollout_asap > 0):
                    print("Set speed zero because pause_end_of_rollout_asap")
                    self.iface.set_speed(0)
                    pause_end_of_rollout_asap = False
                    pause_end_of_rollout_sent = True
                    time_pause_end_of_rollout_sent = time.perf_counter_ns()

                continue

            msgtype &= 0xFF

            self.iface._skip(4)

            if msgtype == MessageType.S_SHUTDOWN:
                print("msg_shutdown")
                self.iface.close()
            elif msgtype == MessageType.S_ON_RUN_STEP:
                print("msg_on_run_step")
                _time = self.iface._read_int32()
                # ============================
                # BEGIN ON RUN STEP
                # ============================
                this_rollout_has_seen_t_negative |= _time < 0
                if not pause_end_of_rollout_asap and not pause_end_of_rollout_sent:
                    if _time == -2000:
                        # Press forward 2000ms before the race starts
                        print("Press forward at beginning of race")
                        self.iface.set_input_state(**(misc.inputs[misc.action_forward_idx]))  # forward
                    elif _time == -100 and not self.snapshot_before_start_is_made:
                        print("Save simulation state")
                        self.simulation_state_to_rewind_to_for_restart = self.iface.get_simulation_state()
                        print(f"{self.simulation_state_to_rewind_to_for_restart.scene_mobil.input_gas=}")
                        print(f"{self.simulation_state_to_rewind_to_for_restart.scene_mobil.input_steer=}")
                        print(f"{self.simulation_state_to_rewind_to_for_restart.scene_mobil.input_brake=}")
                        # assert self.simulation_state_to_rewind_to_for_restart.scene_mobil.input_gas == 1.0 #TODO
                        # assert self.simulation_state_to_rewind_to_for_restart.scene_mobil.input_steer == 0.0
                        # assert self.simulation_state_to_rewind_to_for_restart.scene_mobil.input_brake == 0.0
                        self.snapshot_before_start_is_made = True
                    elif _time >= 0 and _time % (10 * run_steps_per_action) == 0 and this_rollout_has_seen_t_negative:
                        print(f"{_time=}")
                        self.iface.set_speed(0)
                        compute_action_asap = True
                        time_asked_compute_action_asap = time.perf_counter_ns()
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
                result = self.iface._read_int32()
                self.iface._respond_to_call(msgtype)
            elif msgtype == MessageType.S_ON_CHECKPOINT_COUNT_CHANGED:
                print("msg_on_cp_count_changed")
                current = self.iface._read_int32()
                target = self.iface._read_int32()
                # ============================
                # BEGIN ON CP COUNT
                # ============================
                cpcount += 1
                if current == target:  # Finished the race !!
                    self.iface.prevent_simulation_finish()
                    if (
                        not pause_end_of_rollout_asap
                    ):  # We shouldn't take into account a race finished after we ended the rollout
                        simulation_state = self.iface.get_simulation_state()

                        print("END", prev_time, simulation_state.race_time)
                        rv["rewards"].append(  # TODO
                            misc.reward_per_tm_engine_step * (simulation_state.race_time - prev_time) / 10
                            + misc.reward_per_cp_passed * (misc.gamma * cpcount - prev_cpcount)
                            # + misc.reward_per_velocity * (misc.gamma * simulation_state.display_speed - prev_display_speed)
                        )
                        rv["done"].append(True)
                        pause_end_of_rollout_asap = True
                        print(f"Set pause_end_rollout_asap to True because race finished")
                # ============================
                # END ON CP COUNT
                # ============================
                self.iface._respond_to_call(msgtype)
            elif msgtype == MessageType.S_ON_LAPS_COUNT_CHANGED:
                print("msg_on_laps_count_changed")
                current = self.iface._read_int32()
                self.iface._respond_to_call(msgtype)
            elif msgtype == MessageType.S_ON_BRUTEFORCE_EVALUATE:
                print("msg_on_bruteforce_evaluate")
                self.iface._on_bruteforce_validate_call(msgtype)
            elif msgtype == MessageType.S_ON_REGISTERED:
                print("msg_on_registered")
                self.iface.registered = True
                self.iface._respond_to_call(msgtype)
            elif msgtype == MessageType.S_ON_CUSTOM_COMMAND:
                print("msg_on_custom_command")
                _from = self.iface._read_int32()
                to = self.iface._read_int32()
                n_args = self.iface._read_int32()
                command = self.iface._read_string()
                args = []
                for _ in range(n_args):
                    args.append(self.iface._read_string())
                self.iface._respond_to_call(msgtype)
            else:
                print("Unknown msgtype")

            time.sleep(0)

            if (
                _time > max_time
                and not pause_end_of_rollout_asap
                and not pause_end_of_rollout_sent
                and this_rollout_initial_set_speed_is_done
                and this_rollout_has_seen_t_negative
            ):
                # This is taking too long: abort the race and give a large penalty
                rv["rewards"].append(misc.reward_failed_to_finish)
                rv["done"].append(True)
                pause_end_of_rollout_asap = True
                print(f"Set pause_end_rollout_asap to True because {_time=}")

        # ======================================
        # Pause the game until next time
        # iface.set_speed(0)

        # Relase the DXCam resources
        camera.release()
        camera.stop()
        del camera

        print("end rollout")
        return rv


def _set_window_position(trackmania_window):
    print(
        win32gui.SetWindowPos(
            trackmania_window,
            win32con.HWND_TOPMOST,
            2560 - 654,
            120,
            misc.W + misc.wind32gui_margins["left"] + misc.wind32gui_margins["right"],
            misc.H + misc.wind32gui_margins["top"] + misc.wind32gui_margins["bottom"],
            0,
        )
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
    return (left, top, right, bottom)
