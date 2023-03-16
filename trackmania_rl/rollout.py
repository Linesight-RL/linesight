from tminterface.interface import TMInterface, Message, MessageType
import time
import dxcam
import win32gui, win32con
import win32com.client
from . import misc
import random


def rollout(*, running_speed=1, run_steps_per_action=10, max_time=2000):
    trackmania_window = win32gui.FindWindow("TmForever", None)
    camera = None

    # Create TMInterface we will be using to interact with the game client
    iface = TMInterface()
    iface.registered = False

    # Connect
    while not iface._ensure_connected():
        time.sleep(0)
        continue

    _compute_action_asap = False
    _pc_sent_speed_zero = time.perf_counter_ns()
    _set_window_position(trackmania_window)
    _set_window_focus(trackmania_window)

    camera = dxcam.create(region=_get_window_position(trackmania_window), output_color="BGR")

    frames = []
    # =====================================
    # Cleanup

    # Empty all messages received previously
    # TODO

    restart_asap = True

    print("_interface_loop")
    # This code is extracted nearly as-is from TMInterfacePythonClient and modified to run on a single thread
    _time = 0
    while _time <= max_time:
        if not iface._ensure_connected():
            time.sleep(0)
            continue

        if not iface.registered:
            msg = Message(MessageType.C_REGISTER)
            iface._send_message(msg)
            iface._wait_for_server_response()
            iface.registered = True

        if iface.mfile is None:
            print("None")
            continue

        iface.mfile.seek(0)
        msgtype = iface._read_int32()

        if msgtype & 0xFF00 == 0:
            # No message received
            # Let's see if we want to send one
            if _compute_action_asap and (time.perf_counter_ns() - _pc_sent_speed_zero) > 1_000_000:
                # We need to calculate a move AND we have left enough time for the set_speed(0) to have been properly applied
                frame = camera.grab()
                frames.append(frame if frame is not None else frames[-1])
                time.sleep(0.01)  # Arbitrary time to calculate a move
                iface.set_input_state(
                    **random.choices(misc.inputs, weights=[10 if i["accelerate"] else 1 for i in misc.inputs])[0]
                )
                iface.set_speed(running_speed)
                _compute_action_asap = False

            elif restart_asap:
                _restart_race(iface)
                restart_asap = False

            continue

        msgtype &= 0xFF

        # error_code = self.__read_int32()
        iface._skip(4)

        if msgtype == MessageType.S_SHUTDOWN:
            iface.close()
        elif msgtype == MessageType.S_ON_RUN_STEP:
            _time = iface._read_int32()
            # ============================
            # BEGIN ON RUN STEP
            # ============================
            print("-")
            if _time == -100:
                # Press forward 100ms before the race starts
                iface.set_input_state(**(misc.inputs[7]))  # forward
            elif _time < 0:
                # Coutdown: do nothing
                pass
            elif _time % (10 * run_steps_per_action) == 0:
                print(_time)
                iface.set_speed(0)
                _compute_action_asap = True
                _pc_sent_speed_zero = time.perf_counter_ns()
            # ============================
            # END ON RUN STEP
            # ============================
            iface._respond_to_call(msgtype)
        elif msgtype == MessageType.S_ON_SIM_BEGIN:
            print("msg_on_sim_begin")
            iface._respond_to_call(msgtype)
        elif msgtype == MessageType.S_ON_SIM_STEP:
            print("msg_on_sim_step")
            _time = iface._read_int32()
            iface._respond_to_call(msgtype)
        elif msgtype == MessageType.S_ON_SIM_END:
            print("msg_on_sim_end")
            result = iface._read_int32()
            iface._respond_to_call(msgtype)
        elif msgtype == MessageType.S_ON_CHECKPOINT_COUNT_CHANGED:
            print("msg_on_cp_count_changed")
            current = iface._read_int32()
            target = iface._read_int32()
            iface._respond_to_call(msgtype)
        elif msgtype == MessageType.S_ON_LAPS_COUNT_CHANGED:
            print("msg_on_laps_count_changed")
            current = iface._read_int32()
            iface._respond_to_call(msgtype)
        elif msgtype == MessageType.S_ON_BRUTEFORCE_EVALUATE:
            print("msg_on_bruteforce_evaluate")
            iface._on_bruteforce_validate_call(msgtype)
        elif msgtype == MessageType.S_ON_REGISTERED:
            print("msg_on_registered")
            iface.registered = True
            iface._respond_to_call(msgtype)
        elif msgtype == MessageType.S_ON_CUSTOM_COMMAND:
            print("msg_on_custom_command")
            _from = iface._read_int32()
            to = iface._read_int32()
            n_args = iface._read_int32()
            command = iface._read_string()
            args = []
            for _ in range(n_args):
                args.append(iface._read_string())
            iface._respond_to_call(msgtype)
        else:
            print("Unknown msgtype")

        time.sleep(0)

    # ======================================
    # Pause the game until next time
    iface.set_speed(0)
    # Close the interface
    msg = Message(MessageType.C_DEREGISTER)
    msg.write_int32(0)
    iface._send_message(msg)
    # Relase the DXCam resources
    camera.release()
    camera.stop()
    del camera

    return frames


def _restart_race(iface):
    print("_restart_race")
    iface.set_speed(1)
    iface.give_up()
    iface.set_input_state(**(misc.inputs[7]))  # forward


def _set_window_position(trackmania_window):
    win32gui.SetWindowPos(
        trackmania_window,
        win32con.HWND_TOPMOST,
        2560 - 654,
        120,
        misc.W + misc.margins["left"] + misc.margins["right"],
        misc.H + misc.margins["top"] + misc.margins["bottom"],
        0,
    )


def _set_window_focus(trackmania_window):
    shell = win32com.client.Dispatch("WScript.Shell")
    shell.SendKeys("%")
    win32gui.SetForegroundWindow(trackmania_window)


def _get_window_position(trackmania_window):
    rect = win32gui.GetWindowRect(trackmania_window)
    left = rect[0] + misc.margins["left"]
    top = rect[1] + misc.margins["top"]
    right = rect[2] - misc.margins["right"]
    bottom = rect[3] - misc.margins["bottom"]
    return (left, top, right, bottom)
