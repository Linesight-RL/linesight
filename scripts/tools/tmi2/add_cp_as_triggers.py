"""
This script reads a .challenge.gbx file, extracts the checkpoints, and connects to a TMInterface instance to add the checkpoints as triggers.

This script would typically be used to check that the checkpoints are properly read from the .challenge.gbx file.
"""

import argparse
from pathlib import Path

from trackmania_rl.map_loader import get_checkpoint_positions_from_gbx
from trackmania_rl.tmi_interaction.tminterface2 import MessageType, TMInterface


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gbx_path", type=Path)
    parser.add_argument("--tmi_port", "-p", type=int, default=8477)
    args = parser.parse_args()

    checkpoint_positions = get_checkpoint_positions_from_gbx(str(args.gbx_path))

    iface = TMInterface(args.tmi_port)

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
            iface._respond_to_call(msgtype)
        elif msgtype == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
            current = iface._read_int32()
            target = iface._read_int32()
            # ============================
            # BEGIN ON CP COUNT
            # ============================
            # ============================
            # END ON CP COUNT
            # ============================
            iface._respond_to_call(msgtype)
        elif msgtype == int(MessageType.SC_LAP_COUNT_CHANGED_SYNC):
            iface._read_int32()
            iface._respond_to_call(msgtype)
        elif msgtype == int(MessageType.SC_REQUESTED_FRAME_SYNC):
            iface._respond_to_call(msgtype)
        elif msgtype == int(MessageType.C_SHUTDOWN):
            iface.close()
        elif msgtype == int(MessageType.SC_ON_CONNECT_SYNC):
            for i in range(0, len(checkpoint_positions), 1):
                iface.execute_command(
                    f"add_trigger {checkpoint_positions[i][0] - 2} {checkpoint_positions[i][1] - 2} {checkpoint_positions[i][2] - 2} {checkpoint_positions[i][0] + 2} {checkpoint_positions[i][1] + 2} {checkpoint_positions[i][2] + 2}"
                )
            iface._respond_to_call(msgtype)
        else:
            pass


if __name__ == "__main__":
    main()
