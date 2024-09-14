"""
This script reads a .npy file containing a list of VCP, and connects to a TMInterface instance to add the VCP as triggers.

This script would typically be used to check that the .npy file contains Virtual CheckPoints (VCP) that are properly placed.
"""

import argparse
from pathlib import Path

import numpy as np

from trackmania_rl.tmi_interaction.tminterface2 import MessageType, TMInterface


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_path", type=Path)
    parser.add_argument("--tmi_port", "-p", type=int, default=8477)
    args = parser.parse_args()

    iface = TMInterface(args.tmi_port)
    vcp = np.load(args.npy_path)

    if not iface.registered:
        while True:
            try:
                iface.register(2)
                break
            except ConnectionRefusedError as e:
                print(e)

    while True:
        msgtype = iface._read_int32()
        # =============================================
        #        READ INCOMING MESSAGES
        # =============================================
        if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
            _time = iface._read_int32()
            # ============================
            # BEGIN ON RUN STEP
            # ============================
            # ============================
            # END ON RUN STEP
            # ============================
            iface.respond_to_call(msgtype)
        elif msgtype == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
            current = iface._read_int32()
            target = iface._read_int32()
            # ============================
            # BEGIN ON CP COUNT
            # ============================
            # ============================
            # END ON CP COUNT
            # ============================
            iface.respond_to_call(msgtype)
        elif msgtype == int(MessageType.SC_LAP_COUNT_CHANGED_SYNC):
            iface._read_int32()
            iface.respond_to_call(msgtype)
        elif msgtype == int(MessageType.SC_REQUESTED_FRAME_SYNC):
            iface.respond_to_call(msgtype)
        elif msgtype == int(MessageType.C_SHUTDOWN):
            iface.close()
        elif msgtype == int(MessageType.SC_ON_CONNECT_SYNC):
            for i in range(0, len(vcp), 10):
                iface.execute_command(
                    f"add_trigger {vcp[i][0] - 0.4:.2f} {vcp[i][1] - 0.4:.2f} {vcp[i][2] - 0.4:.2f} {vcp[i][0] + 0.4:.2f} {vcp[i][1] + 0.4:.2f} {vcp[i][2] + 0.4:.2f}"
                )
                # print(
                #     f"add_trigger {vcp[i][0] - 0.4:.2f} {vcp[i][1] - 0.4:.2f} {vcp[i][2] - 0.4:.2f} {vcp[i][0] + 0.4:.2f} {vcp[i][1] + 0.4:.2f} {vcp[i][2] + 0.4:.2f}"
                # )
            iface.respond_to_call(msgtype)
        else:
            pass


if __name__ == "__main__":
    main()
