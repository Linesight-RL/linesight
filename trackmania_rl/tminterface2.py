import socket
import struct
import time
import signal
import numpy as np
from enum import IntEnum, auto
from tminterface.structs import SimStateData, CheckpointData

HOST = "127.0.0.1"
PORT = 8477

class MessageType(IntEnum):
	SC_RUN_STEP_SYNC = auto()
	SC_CHECKPOINT_COUNT_CHANGED_SYNC = auto()
	SC_LAP_COUNT_CHANGED_SYNC = auto()
	SC_REQUESTED_FRAME_SYNC = auto()
	C_SET_SPEED = auto()
	C_REWIND_TO_STATE = auto()
	C_RREWIND_TO_CURRENT_STATE = auto()
	C_GET_SIMULATION_STATE = auto()
	C_SET_INPUT_STATE = auto()
	C_GIVE_UP = auto()
	C_PREVENT_SIMULATION_FINISH = auto()
	C_SHUTDOWN = auto()
	C_EXECUTE_COMMAND = auto()
	C_SET_TIMEOUT = auto()
	C_RACE_FINISHED = auto()
	C_REQUEST_FRAME = auto()
	SC_REQUEST_DESIRED_MAP_SYNC = auto()

class TMInterface:
	registered = False
	def __init__(self, name):
		pass

	def close(self):
		self.sock.sendall(struct.pack('i', MessageType.C_SHUTDOWN))
		self.sock.close()
		self.registered = False

	def signal_handler(self, sig, frame):
		print('Shutting down...')
		self.close()

	def register(self):
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		signal.signal(signal.SIGINT, self.signal_handler)
		self.sock.connect((HOST, PORT))
		self.registered = True
		print('Connected')

	def rewind_to_state(self, state):
		self.sock.sendall(struct.pack('i', MessageType.C_REWIND_TO_STATE))
		self.sock.sendall(struct.pack('i', np.int32(len(state.data))))
		self.sock.sendall(state.data)

	def rewind_to_current_state(self):
		self.sock.sendall(struct.pack('i', MessageType.C_RREWIND_TO_CURRENT_STATE))

	def get_simulation_state(self):
		self.sock.sendall(struct.pack('i', MessageType.C_GET_SIMULATION_STATE))
		state_length = self._read_int32()
		state = SimStateData(self.sock.recv(state_length))
		state.cp_data.resize(CheckpointData.cp_states_field, state.cp_data.cp_states_length)
		state.cp_data.resize(CheckpointData.cp_times_field, state.cp_data.cp_times_length)
		return state

	def set_input_state(self, left: bool, right: bool, accelerate: bool, brake: bool):
		self.sock.sendall(struct.pack('i', MessageType.C_SET_INPUT_STATE))
		self.sock.sendall(struct.pack('B', np.uint8(left)))
		self.sock.sendall(struct.pack('B', np.uint8(right)))
		self.sock.sendall(struct.pack('B', np.uint8(accelerate)))
		self.sock.sendall(struct.pack('B', np.uint8(brake)))

	def give_up(self):
		self.sock.sendall(struct.pack('i', MessageType.C_GIVE_UP))

	def prevent_simulation_finish(self):
		self.sock.sendall(struct.pack('i', MessageType.C_PREVENT_SIMULATION_FINISH))

	def execute_command(self, command: str):
		self.sock.sendall(struct.pack('i', MessageType.C_EXECUTE_COMMAND))
		self.sock.sendall(struct.pack('i', np.int32(len(command))))
		self.sock.sendall(command.encode()) #https://www.delftstack.com/howto/python/python-socket-send-string/

	def set_timeout(self, new_timeout: int):
		self.sock.sendall(struct.pack('i', MessageType.C_SET_TIMEOUT))
		self.sock.sendall(struct.pack('I', np.uint32(new_timeout)))

	def set_speed(self, new_speed):
		self.sock.sendall(struct.pack('i', MessageType.C_SET_SPEED))
		self.sock.sendall(struct.pack('f', np.float32(new_speed)))

	def race_finished(self):
		self.sock.sendall(struct.pack('i', MessageType.C_RACE_FINISHED))
		a = self._read_int32()
		return a

	def request_frame(self,frames_to_skip: int):
		self.sock.sendall(struct.pack('i', MessageType.C_REQUEST_FRAME))
		self.sock.sendall(struct.pack('i', np.int32(frames_to_skip)))

	def _respond_to_call(self, response_type):
		self.sock.sendall(struct.pack('i', np.int32(response_type)))

	def _read_int32(self):
		return struct.unpack('i', self.sock.recv(4))[0]