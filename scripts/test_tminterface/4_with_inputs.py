import mmap
import random
import signal
import sys
import time
from threading import Condition

from tminterface.client import Client, run_client
from tminterface.constants import DEFAULT_SERVER_SIZE
from tminterface.interface import Message, MessageType, TMInterface

from trackmania_rl.misc import inputs

random.seed(42)


def on_run_step(iface: TMInterface, _time: int):
    # print("on_run_step")
    pass


def on_registered(iface: TMInterface):
    # print("on_registered")
    pass


def restart_race(iface: TMInterface):
    iface.set_input_state(**(inputs[7]))
    iface.give_up()
    iface.set_speed(1)


iface = TMInterface()
iface.registered = False
prev_on_run_step_time = time.time()
theoretical_game_speed = 1


while not iface._ensure_connected():
    time.sleep(0)
    continue


# ==============================================================================

import cv2
import dxcam
import win32con
import win32gui

# sct.stop()

W = 640
H = 480  # Just as Terry Davis would have wanted it

trackmania_window = win32gui.FindWindow("TmForever", None)
# trackmania_window = win32gui.FindWindow('TrackMania Nations Forever (TMInterface 1.4.1)',None)


def get_window_position():
    trackmania_window = win32gui.FindWindow("TmForever", None)
    rect = win32gui.GetWindowRect(trackmania_window)
    left = rect[0] + margins["left"]
    top = rect[1] + margins["top"]
    right = rect[2] - margins["right"]
    bottom = rect[3] - margins["bottom"]
    return (left, top, right, bottom)


# Windows 10 has thin invisible borders on left, right, and bottom, it is used to grip the mouse for resizing.
# The borders might look like this: 7,0,7,7 (left, top, right, bottom)
margins = {"left": 7, "top": 32, "right": 7, "bottom": 7}

# To get 640x460 à la louche
# rect width : 654
# rect height : 487


win32gui.SetWindowPos(
    trackmania_window,
    win32con.HWND_TOPMOST,
    # win32con.HWND_TOP,
    2560 - 740,
    100,
    W + margins["left"] + margins["right"],
    H + margins["top"] + margins["bottom"],
    0,
)

# Add this import
import win32com.client

# Add this to __ini__
shell = win32com.client.Dispatch("WScript.Shell")
# And SetAsForegroundWindow becomes
shell.SendKeys("%")
win32gui.SetForegroundWindow(trackmania_window)


target_fps = 20
camera = dxcam.create(region=get_window_position(), output_color="BGR")
writer = cv2.VideoWriter("video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (W, H))

# ==============================================================================

time.sleep(1)


restart_race(iface)

####################
## BEGIN _main_thread ()
####################

while True:
    # fine-grained sleep equivalent
    # j = 0
    # for i in range(1000000):
    #     j += i

    if not iface._ensure_connected():
        time.sleep(0)
        continue

    if not iface.registered:
        msg = Message(MessageType.C_REGISTER)
        iface._send_message(msg)
        iface._wait_for_server_response()
        iface.registered = True

    if theoretical_game_speed == 0 and (time.time() - time_when_game_paused) > 1:
        print("set_speed 1")
        #################################### (below is equivalent to iface.rewind_to_state() )
        # msg = Message(MessageType.C_SIM_REWIND_TO_STATE)
        # msg.write_buffer(saved_state.data)
        # iface._send_message(msg)
        # iface._wait_for_server_response()
        ####################################
        iface.set_speed(1)
        theoretical_game_speed = 1

    #####################
    ## BEGIN _process_server_message()
    #####################
    if iface.mfile is None:
        continue

    iface.mfile.seek(0)
    msgtype = iface._read_int32()
    # print("msg", msgtype)
    # if msgtype != 0:
    #     print("YEAAAAAAAAAAAAAHHHHHHHH")
    if msgtype & 0xFF00 == 0:
        continue

    msgtype &= 0xFF

    # error_code = self.__read_int32()
    iface._skip(4)

    if msgtype == MessageType.S_SHUTDOWN:
        print("msg_shutdown")
        iface.close()
        # iface.client.on_shutdown(iface)
    elif msgtype == MessageType.S_ON_RUN_STEP:
        print(
            f"msg_on_run_step  {theoretical_game_speed=} {(time.time() - prev_on_run_step_time)*1000:.1f} {'AAAAAAAAAAAAAAAAAAA' if theoretical_game_speed == 0 else ''}"
        )
        prev_on_run_step_time = time.time()
        _time = iface._read_int32()
        # iface.client.on_run_step(iface, _time)
        on_run_step(iface, _time)
        if _time > 0 and _time % 50 == 0:
            writer.write(camera.grab())
            saved_state = iface.get_simulation_state()
            print(f"{saved_state.race_time=} {_time=}")
            time_when_game_paused = time.time()
            c = random.choices(inputs, weights=[10 if i["accelerate"] else 1 for i in inputs])[0]
            print(c)
            iface.set_input_state(**c)
            # print("set_speed 0")
            # iface.set_speed(0)
            # theoretical_game_speed = 0
        iface._respond_to_call(msgtype)

        if _time > 5000:
            writer.release()
            assert False
    elif msgtype == MessageType.S_ON_SIM_BEGIN:
        print("msg_on_sim_begin")
        # iface.client.on_simulation_begin(iface)
        iface._respond_to_call(msgtype)
    elif msgtype == MessageType.S_ON_SIM_STEP:
        print("msg_on_sim_step")
        _time = iface._read_int32()
        # iface.client.on_simulation_step(iface, _time)
        iface._respond_to_call(msgtype)
    elif msgtype == MessageType.S_ON_SIM_END:
        print("msg_on_sim_end")
        result = iface._read_int32()
        # iface.client.on_simulation_end(iface, result)
        iface._respond_to_call(msgtype)
    elif msgtype == MessageType.S_ON_CHECKPOINT_COUNT_CHANGED:
        print("msg_on_cp_count_changed")
        current = iface._read_int32()
        target = iface._read_int32()
        # iface.client.on_checkpoint_count_changed(iface, current, target)
        iface._respond_to_call(msgtype)
    elif msgtype == MessageType.S_ON_LAPS_COUNT_CHANGED:
        print("msg_on_laps_count_changed")
        current = iface._read_int32()
        # iface.client.on_laps_count_changed(iface, current)
        iface._respond_to_call(msgtype)
    elif msgtype == MessageType.S_ON_BRUTEFORCE_EVALUATE:
        print("msg_on_bruteforce_evaluate")
        iface._on_bruteforce_validate_call(msgtype)
    elif msgtype == MessageType.S_ON_REGISTERED:
        print("msg_on_registered")
        iface.registered = True
        # iface.client.on_registered(iface)
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

        # iface.client.on_custom_command(iface, _from, to, command, args)
        iface._respond_to_call(msgtype)
    else:
        print("Unknown msgtype")
    time.sleep(0)


# flag = False
# while flag == False:
#     try:
#         iface.mfile = mmap.mmap(-1, iface.buffer_size, tagname=iface.server_name)
#         flag = True
#     except Exception as e:
#         print("fail mmap")

# print("B1")
# msg = Message(MessageType.C_REGISTER)
# print("C")
# iface._send_message(msg)
# print("D")
# iface._wait_for_server_response()
# print("E")


# while True:

#     if iface.mfile is None:
#         #No message
#         print("G")
#         continue

#     iface.mfile.seek(0)
#     msgtype = iface._read_int32()
#     if msgtype & 0xFF00 == 0:
#         #No message
#         print("H")
#         continue

#     print("I")

#     msgtype &= 0xFF

#     iface._skip(4)

#     if msgtype == MessageType.S_SHUTDOWN:
#         iface.close()
#         # iface.client.on_shutdown(self)
#     elif msgtype == MessageType.S_ON_RUN_STEP:
#         _time = iface._read_int32()
#         on_run_step(iface, _time)
#         iface._respond_to_call(msgtype)
#     elif msgtype == MessageType.S_ON_SIM_BEGIN:
#         # iface.client.on_simulation_begin(self)
#         iface._respond_to_call(msgtype)
#     elif msgtype == MessageType.S_ON_SIM_STEP:
#         _time = iface._read_int32()
#         # iface.client.on_simulation_step(self, _time)
#         iface._respond_to_call(msgtype)
#     elif msgtype == MessageType.S_ON_SIM_END:
#         result = iface._read_int32()
#         # iface.client.on_simulation_end(self, result)
#         iface._respond_to_call(msgtype)
#     elif msgtype == MessageType.S_ON_CHECKPOINT_COUNT_CHANGED:
#         current = iface._read_int32()
#         target = iface._read_int32()
#         # iface.client.on_checkpoint_count_changed(self, current, target)
#         iface._respond_to_call(msgtype)
#     elif msgtype == MessageType.S_ON_LAPS_COUNT_CHANGED:
#         current = iface._read_int32()
#         # iface.client.on_laps_count_changed(self, current)
#         iface._respond_to_call(msgtype)
#     elif msgtype == MessageType.S_ON_BRUTEFORCE_EVALUATE:
#         iface._on_bruteforce_validate_call(msgtype)
#     elif msgtype == MessageType.S_ON_REGISTERED:
#         iface.registered = True
#         on_registered(iface)
#         iface._respond_to_call(msgtype)
#     elif msgtype == MessageType.S_ON_CUSTOM_COMMAND:
#         _from = iface._read_int32()
#         to = iface._read_int32()
#         n_args = iface._read_int32()
#         command = iface._read_string()
#         args = []
#         for _ in range(n_args):
#             args.append(iface._read_string())

#         # iface.client.on_custom_command(self, _from, to, command, args)
#         iface._respond_to_call(msgtype)
