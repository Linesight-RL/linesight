from pathlib import Path

import numpy as np

from trackmania_rl.tmi_interaction.tminterface2 import MessageType, TMInterface

iface = TMInterface(8477)
base_dir = Path(__file__).resolve().parents[2]

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
        iface._respond_to_call(msgtype)
    else:
        pass
