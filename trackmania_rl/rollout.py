from tminterface.interface import TMInterface, Message, MessageType
import time
import dxcam
import win32gui, win32con
import win32com.client
from . import misc
import random

class RolloutWorker:
    def __init__(self, running_speed=1, run_steps_per_action=10):
        # Worker configuration
        self.running_speed = running_speed
        self.run_steps_per_action = run_steps_per_action
        self.trackmania_window = win32gui.FindWindow("TmForever", None)
        self.camera = None

        self._set_window_position()

        # Create TMInterface we will be using to interact with the game client
        self.iface = TMInterface()
        self.iface.registered = False

        # Connect
        while not self.iface._ensure_connected():
            time.sleep(0)
            continue

        # # Register
        # msg = Message(MessageType.C_REGISTER)
        # self.iface._send_message(msg)
        # self.iface._wait_for_server_response()
        # self.iface.registered = True

        # # Pause
        # self.iface.set_speed(0)

        pass

    def _restart_race(self):
        print("_restart_race")
        self.iface.give_up()
        self.iface.set_speed(self.running_speed)
        self.iface.set_input_state(**(misc.inputs[7]))  # forward

    def _set_window_position(self):
        win32gui.SetWindowPos(
            self.trackmania_window,
            win32con.HWND_TOPMOST,
            2560 - 654,
            120,
            misc.W + misc.margins["left"] + misc.margins["right"],
            misc.H + misc.margins["top"] + misc.margins["bottom"],
            0,
        )

    def _set_window_focus(self):
        shell = win32com.client.Dispatch("WScript.Shell")
        shell.SendKeys("%")
        win32gui.SetForegroundWindow(self.trackmania_window)

    def _get_window_position(self):
        rect = win32gui.GetWindowRect(self.trackmania_window)
        left = rect[0] + misc.margins["left"]
        top = rect[1] + misc.margins["top"]
        right = rect[2] - misc.margins["right"]
        bottom = rect[3] - misc.margins["bottom"]
        return (left, top, right, bottom)

    def play_one_race(self, actor):

        self._set_window_position()
        self._set_window_focus()

        if self.camera is not None:
            del self.camera
        self.camera = dxcam.create(region=self._get_window_position(), output_color="BGR")

        self.screenshots = []
        # =====================================
        # Cleanup

        # Empty all messages received previously
        # TODO

        # Restart the race
        self._restart_race()

        self._interface_loop()

        # ======================================
        # Pause the game until next time
        self.iface.set_speed(0)
        self.camera.release()
        self.camera.stop()
        memories = None
        return memories

    def _interface_loop(self):
        print("_interface_loop")
        # This code is extracted nearly as-is from TMInterfacePythonClient and modified to run on a single thread
        _time = 0
        while _time < 2000:

            if not self.iface._ensure_connected():
                time.sleep(0)
                continue

            if not self.iface.registered:
                msg = Message(MessageType.C_REGISTER)
                self.iface._send_message(msg)
                self.iface._wait_for_server_response()
                self.iface.registered = True


            if self.iface.mfile is None:
                continue

            self.iface.mfile.seek(0)
            msgtype = self.iface._read_int32()

            if msgtype & 0xFF00 == 0:
                continue

            msgtype &= 0xFF

            # error_code = self.__read_int32()
            self.iface._skip(4)

            if msgtype == MessageType.S_SHUTDOWN:
                self.iface.close()
            elif msgtype == MessageType.S_ON_RUN_STEP:
                _time = self.iface._read_int32()
                self._on_run_step(self.iface, _time)
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

    def _on_run_step(self, iface: TMInterface, time: int):
        # time is the race time in milliseconds.
        # It is negative during coutdown at the beginning of a race.
        # It is always guaranteed to be a multiple of 10 as the game engine works in 10ms increments

        if time == -100:
            self.iface.set_input_state(**(misc.inputs[7]))  # forward

        if time < 0:
            # Coutdown: do nothing
            return
        if time % (10 * self.run_steps_per_action) != 0:
            # This is not a frame we're interested in, do nothing
            return
        
        c = random.choices(misc.inputs, weights=[10 if i["accelerate"] else 1 for i in misc.inputs])[0]
        iface.set_input_state(**c)
        self.screenshots.append(self.camera.grab())
        pass
