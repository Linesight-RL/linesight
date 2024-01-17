import sys

import numpy as np
from tminterface.client import Client, run_client
from tminterface.interface import TMInterface


class MainClient(Client):
    def __init__(self) -> None:
        super(MainClient, self).__init__()
        self.prev_state = None

    def on_registered(self, iface: TMInterface) -> None:
        print(f"Registered to {iface.server_name}")

    def on_run_step(self, iface: TMInterface, _time: int):
        if _time >= 0:
            state = iface.get_simulation_state()
            # print(
            #     f'Time: {_time}\n'
            #     f'Display Speed: {state.display_speed}\n'
            #     f'Position: {state.position}\n'
            #     f'Velocity: {state.velocity}\n'
            #     f'YPW: {state.yaw_pitch_roll}\n'
            # )
            # print(state.scene_mobil.has_any_lateral_contact)
            if _time % 2000 == 0:
                # iface.execute_command("tp 0+ 0.8+ 1+")
                state.dyna.current_state.position += np.array([0.0, 0.8, 2])
                state.dyna.current_state.linear_speed = np.array([0, 0, 0])
                state.dyna.current_state.angular_speed = np.array([0, 0, 0])
                iface.rewind_to_state(state)

            state = iface.get_simulation_state()

            if _time % 2000 < 100:
                if self.prev_state is not None:
                    print(
                        _time % 2000,
                        np.round(
                            np.array(state.position) - np.array(self.prev_state.position),
                            2,
                        ),
                    )

            self.prev_state = state


server_name = f"TMInterface{sys.argv[1]}" if len(sys.argv) > 1 else "TMInterface0"
print(f"Connecting to {server_name}...")
client = MainClient()
run_client(client, server_name)


# [SimStateData object at 0x26454aaee60]
# version (int): 1
# context_mode (int): 1
# flags (int): 159
# timers (ndarray):
# 	[64550 64540 0 64550 64540 0 64550 64540 0 64700 64600 0 64550 64540 0
# 	 64550]   (37 more items...)
# dyna (HmsDynaStruct):
# 	previous_state (HmsDynaStateStruct):
# 		quat (ndarray):
# 			[0.7071026563644409 -2.780207069008611e-07 0.7071109414100647
# 			 2.1565574570558965e-07]
# 		rotation (ndarray):
# 			[[-1.1801719665527344e-05 -6.981645128689706e-07 1.0]
# 			 [-8.820146035759535e-08 1.0 6.981634328440123e-07]
# 			 [-1.0 -8.819330332698883e-08 -1.1801719665527344e-05]]
# 		position (ndarray): [173.78851318359375 90.01425170898438 687.9995727539062]
# 		linear_speed (ndarray): [-8.382259863992658e-08 -0.00015273242024704814 0.008355516940355301]
# 		add_linear_speed (ndarray): [0.0 0.0 0.0]
# 		angular_speed (ndarray): [0.0009078684961423278 5.472696740227434e-10 -2.6550021630100673e-06]
# 		force (ndarray): [2.012505501625128e-05 7.535386248491704e-05 1.6707931756973267]
# 		torque (ndarray): [0.1494792401790619 1.0803341865539551e-07 0.00017152931832242757]
# 		inverse_inertia_tensor (ndarray):
# 			[[1.2000000476837158 5.026777216698974e-07 -8.497238013660535e-06]
# 			 [5.026777216698974e-07 0.48000001907348633 -5.972345014371161e-12]
# 			 [-8.497238013660535e-06 -5.972345014371161e-12 0.48000001907348633]]
# 		unknown (float): 0.0
# 		not_tweaked_linear_speed (ndarray): [0.0 0.0 0.0]
# 		owner (int): 507863080
# 	current_state (HmsDynaStateStruct):
# 		quat (ndarray):
# 			[0.7071026563644409 2.9411476134555414e-06 0.7071109414100647
# 			 3.4160877930844435e-06]
# 		rotation (ndarray):
# 			[[-1.180174331238959e-05 -6.716140887874644e-07 1.0]
# 			 [8.990484275273047e-06 1.0 6.717195901728701e-07]
# 			 [-1.0 8.990493370220065e-06 -1.1801719665527344e-05]]
# 		position (ndarray): [173.78851318359375 90.01425170898438 687.9996337890625]
# 		linear_speed (ndarray): [-2.714770630518615e-07 -0.00015244152746163309 -0.00835232250392437]
# 		add_linear_speed (ndarray): [0.0 0.0 0.0]
# 		angular_speed (ndarray): [-0.0008858568035066128 -7.047975536522699e-10 -1.6836244185469695e-06]
# 		force (ndarray): [-1.876544592960272e-05 2.9086832000757568e-05 -1.6707837581634521]
# 		torque (ndarray): [-0.14947709441184998 -1.043081283569336e-07 0.0001997241924982518]
# 		inverse_inertia_tensor (ndarray):
# 			[[1.2000000476837158 4.836378479922132e-07 -8.49722982820822e-06]
# 			 [4.836378479922132e-07 0.48000001907348633 -5.420209420181621e-12]
# 			 [-8.49722982820822e-06 -5.420210287543359e-12 0.48000001907348633]]
# 		unknown (float): 0.0
# 		not_tweaked_linear_speed (ndarray): [0.0 0.0 0.0]
# 		owner (int): 507863080
# 	temp_state (HmsDynaStateStruct):
# 		quat (ndarray):
# 			[0.7071026563644409 -2.780207069008611e-07 0.7071109414100647
# 			 2.1565574570558965e-07]
# 		rotation (ndarray):
# 			[[-1.1801719665527344e-05 -6.981645128689706e-07 1.0]
# 			 [-8.820146035759535e-08 1.0 6.981634328440123e-07]
# 			 [-1.0 -8.819330332698883e-08 -1.1801719665527344e-05]]
# 		position (ndarray): [173.78851318359375 90.01425170898438 687.9995727539062]
# 		linear_speed (ndarray): [-8.382259863992658e-08 -0.00015273242024704814 0.008355516940355301]
# 		add_linear_speed (ndarray): [0.0 0.0 0.0]
# 		angular_speed (ndarray): [0.0009078684961423278 5.472696740227434e-10 -2.6550021630100673e-06]
# 		force (ndarray): [2.012505501625128e-05 7.535386248491704e-05 1.6707931756973267]
# 		torque (ndarray): [0.1494792401790619 1.0803341865539551e-07 0.00017152931832242757]
# 		inverse_inertia_tensor (ndarray):
# 			[[1.2000000476837158 5.026777216698974e-07 -8.497238013660535e-06]
# 			 [5.026777216698974e-07 0.48000001907348633 -5.972345014371161e-12]
# 			 [-8.497238013660535e-06 -5.972345014371161e-12 0.48000001907348633]]
# 		unknown (float): 0.0
# 		not_tweaked_linear_speed (ndarray): [0.0 0.0 0.0]
# 		owner (int): 507863080
# 	rest (bytearray): [ 34 61 45 1E E8 61 45 1E 04 00 00 00 C8 62 BD 1E  (600 more bytes...) ]
# scene_mobil (SceneVehicleCar):
# 	is_update_async (bool): True
# 	input_gas (float): 0.0
# 	input_brake (float): 0.0
# 	input_steer (float): 0.0
# 	is_light_trials_set (bool): False
# 	horn_limit (int): 3
# 	quality (int): 2
# 	max_linear_speed (float): 277.77777099609375
# 	gearbox_state (int): 0
# 	block_flags (int): 4
# 	prev_sync_vehicle_state (SceneVehicleCarState):
# 		speed_forward (float): 1.980953456692158e-20
# 		speed_sideward (float): 5.605193857299268e-45
# 		input_steer (float): 1.2214310164305515e-38
# 		input_gas (float): -1.8253869882300933e-07
# 		input_brake (float): -0.008355516940355301
# 		is_turbo (bool): False
# 		rpm (float): 0.008355516940355301
# 		gearbox_state (int): 0
# 		rest (bytearray): [ 00 00 00 00 01 00 00 00 00 00 00 00 2C 00 2C 00  (12 more bytes...) ]
# 	sync_vehicle_state (SceneVehicleCarState):
# 		speed_forward (float): 0.0
# 		speed_sideward (float): 0.0
# 		input_steer (float): 0.0
# 		input_gas (float): -1.7300769172834407e-07
# 		input_brake (float): 0.008352321572601795
# 		is_turbo (bool): False
# 		rpm (float): -0.008352321572601795
# 		gearbox_state (int): 0
# 		rest (bytearray): [ 01 00 E6 42 01 00 00 00 00 00 00 00 2C 00 2C 00  (12 more bytes...) ]
# 	async_vehicle_state (SceneVehicleCarState):
# 		speed_forward (float): 0.0
# 		speed_sideward (float): 0.0
# 		input_steer (float): 0.0
# 		input_gas (float): -1.916562553105905e-07
# 		input_brake (float): 0.006681636441498995
# 		is_turbo (bool): False
# 		rpm (float): -0.0066816359758377075
# 		gearbox_state (int): 0
# 		rest (bytearray): [ 00 00 1C 42 01 00 00 00 00 00 00 00 2C 00 2C 00  (12 more bytes...) ]
# 	prev_async_vehicle_state (SceneVehicleCarState):
# 		speed_forward (float): 0.0
# 		speed_sideward (float): 0.0
# 		input_steer (float): 0.0
# 		input_gas (float): -2.5565441319486126e-07
# 		input_brake (float): 0.0016694297082722187
# 		is_turbo (bool): False
# 		rpm (float): -0.0016694292426109314
# 		gearbox_state (int): 0
# 		rest (bytearray): [ 01 00 8A 42 01 00 00 00 00 00 00 00 2C 00 2C 00  (12 more bytes...) ]
# 	engine (Engine):
# 		max_rpm (float): 11000.0
# 		braking_factor (float): -0.0
# 		clamped_rpm (float): 115.00000762939453
# 		actual_rpm (float): 6.90467277308926e-05
# 		slide_factor (float): 1.0
# 		rear_gear (int): 0
# 		gear (int): 1
# 	has_any_lateral_contact (bool): False
# 	last_has_any_lateral_contact_time (int): -1
# 	water_forces_applied (bool): False
# 	turning_rate (float): 0.0
# 	turbo_boost_factor (float): 0.0
# 	last_turbo_type_change_time (int): 1042198257
# 	last_turbo_time (int): 1042921428
# 	turbo_type (int): 0
# 	roulette_value (float): 0.126658096909523
# 	is_freewheeling (bool): False
# 	is_sliding (bool): False
# 	wheel_contact_absorb_counter (int): 0
# 	burnout_state (int): 0
# 	current_local_speed (ndarray): [-0.008355516940355301 -0.00015273316239472479 -1.8253869882300933e-07]
# 	total_central_force_added (ndarray): [1.670780897140503 30.000028610229492 2.18976001633564e-05]
# 	is_rubber_ball (bool): False
# 	saved_state (ndarray):
# 		[[0.0 0.0 0.0]
# 		 [0.0 0.0 0.0]
# 		 [0.0 0.0 0.0]
# 		 [0.0 0.0 0.0]]
# simulation_wheels (ndarray):
# 	[[SimulationWheel object at 0x26454ca56f0]
# 	 steerable (bool): True
# 	 field_8 (int): 1052401205
# 	 surface_handler (SurfaceHandler):
# 	 	unknown (ndarray):
# 	 		[[1.0 0.0 0.0]
# 	 		 [0.0 1.0 0.0]
# 	 		 [0.0 0.0 1.0]
# 	 		 [0.8630120158195496 0.35249999165534973 1.7820889949798584]]
# 	 	rotation (ndarray):
# 	 		[[1.0 0.0 0.0]
# 	 		 [0.0 1.0 0.0]
# 	 		 [0.0 0.0 1.0]]
# 	 	position (ndarray): [0.8630120158195496 0.33988508582115173 1.7820889949798584]
# 	 field_112 (ndarray):
# 	 	[[1.0 0.0 0.0]
# 	 	 [0.0 1.0 0.0]
# 	 	 [0.0 0.0 1.0]
# 	 	 [0.0 0.0 0.0]]
# 	 field_160 (int): 1065353216
# 	 field_164 (int): 1065353216
# 	 offset_from_vehicle (ndarray): [0.8630120158195496 0.35249999165534973 1.7820889949798584]
# 	 real_time_state (RealTimeState):
# 	 	damper_absorb (float): 0.012614906765520573
# 	 	field_4 (float): -0.00041611489723436534
# 	 	field_8 (float): 0.00985108781605959
# 	 	field_12 (ndarray):
# 	 		[[1.0 0.0 0.0]
# 	 		 [0.0 1.0 0.0]
# 	 		 [0.0 0.0 1.0]]
# 	 	field_48 (ndarray):
# 	 		[[1.0 -8.679245411258307e-08 0.0]
# 	 		 [8.679245411258307e-08 1.0 -6.942739787518803e-07]
# 	 		 [6.025774090855432e-14 6.942739787518803e-07 1.0]]
# 	 	field_84 (ndarray): [0.8630159497261047 -0.014260664582252502 1.7820841073989868]
# 	 	field_108 (float): -5.014799739910814e-07
# 	 	has_ground_contact (bool): True
# 	 	contact_material_id (int): -634978288
# 	 	is_sliding (bool): False
# 	 	relative_rotz_axis (ndarray): [-1.1801719665527344e-05 -6.981645128689706e-07 1.0]
# 	 	nb_ground_contacts (int): 1
# 	 	field_144 (ndarray): [8.938776772993151e-06 0.9999999403953552 6.726838819304248e-07]
# 	 	rest (bytearray): [ EB 8C E3 40 00 00 00 80 00 00 00 80 ]
# 	 field_348 (int): 0
# 	 contact_relative_local_distance (ndarray): [0.0 0.0 0.0]
# 	 prev_sync_wheel_state (WheelState):
# 	 	rest (bytearray): [ 35 C0 4E 3C EB 8C E3 40 00 00 00 80 10 00 59 3F  (84 more bytes...) ]
# 	 sync_wheel_state (WheelState):
# 	 	rest (bytearray): [ C1 AE 4E 3C EB 8C E3 40 00 00 00 80 10 00 59 3F  (84 more bytes...) ]
# 	 field_564 (WheelState):
# 	 	rest (bytearray): [ D7 BB 4E 3C EC 8C E3 40 00 00 00 80 10 00 48 B5  (84 more bytes...) ]
# 	 async_wheel_state (WheelState):
# 	 	rest (bytearray): [ 5D B3 4E 3C EB 8C E3 40 00 00 00 80 10 00 48 B5  (84 more bytes...) ]
# 	 [SimulationWheel object at 0x26454ca5c90]
# 	 steerable (bool): True
# 	 field_8 (int): 1052401205
# 	 surface_handler (SurfaceHandler):
# 	 	unknown (ndarray):
# 	 		[[1.0 0.0 0.0]
# 	 		 [0.0 1.0 0.0]
# 	 		 [0.0 0.0 1.0]
# 	 		 [-0.8629900217056274 0.35249999165534973 1.7820889949798584]]
# 	 	rotation (ndarray):
# 	 		[[1.0 0.0 0.0]
# 	 		 [0.0 1.0 0.0]
# 	 		 [0.0 0.0 1.0]]
# 	 	position (ndarray): [-0.8629900217056274 0.3398878276348114 1.7820889949798584]
# 	 field_112 (ndarray):
# 	 	[[1.0 0.0 0.0]
# 	 	 [0.0 1.0 0.0]
# 	 	 [0.0 0.0 1.0]
# 	 	 [0.0 0.0 0.0]]
# 	 field_160 (int): 1065353216
# 	 field_164 (int): 1065353216
# 	 offset_from_vehicle (ndarray): [-0.8629900217056274 0.35249999165534973 1.7820889949798584]
# 	 real_time_state (RealTimeState):
# 	 	damper_absorb (float): 0.012612167745828629
# 	 	field_4 (float): 0.00037504357169382274
# 	 	field_8 (float): 0.009866427630186081
# 	 	field_12 (ndarray):
# 	 		[[1.0 0.0 0.0]
# 	 		 [0.0 1.0 0.0]
# 	 		 [0.0 0.0 1.0]]
# 	 	field_48 (ndarray):
# 	 		[[1.0 -8.679250385057458e-08 0.0]
# 	 		 [8.679250385057458e-08 1.0 -6.942740924387181e-07]
# 	 		 [6.025778834239937e-14 6.942740924387181e-07 1.0]]
# 	 	field_84 (ndarray): [-0.8629972338676453 -0.01424514688551426 1.7820943593978882]
# 	 	field_108 (float): -5.014799739910814e-07
# 	 	has_ground_contact (bool): True
# 	 	contact_material_id (int): 1065222160
# 	 	is_sliding (bool): False
# 	 	relative_rotz_axis (ndarray): [-1.1801719665527344e-05 -6.981645128689706e-07 1.0]
# 	 	nb_ground_contacts (int): 1
# 	 	field_144 (ndarray): [9.025562576425727e-06 0.9999999403953552 6.726849051119643e-07]
# 	 	rest (bytearray): [ EB 8C E3 40 00 00 00 80 00 00 00 80 ]
# 	 field_348 (int): 0
# 	 contact_relative_local_distance (ndarray): [0.0 0.0 0.0]
# 	 prev_sync_wheel_state (WheelState):
# 	 	rest (bytearray): [ 89 93 4E 3C EB 8C E3 40 00 00 00 80 10 00 00 00  (84 more bytes...) ]
# 	 sync_wheel_state (WheelState):
# 	 	rest (bytearray): [ 44 A3 4E 3C EB 8C E3 40 00 00 00 80 10 00 00 00  (84 more bytes...) ]
# 	 field_564 (WheelState):
# 	 	rest (bytearray): [ A4 B0 4E 3C EC 8C E3 40 00 00 00 80 10 00 00 00  (84 more bytes...) ]
# 	 async_wheel_state (WheelState):
# 	 	rest (bytearray): [ 84 A1 4E 3C EB 8C E3 40 00 00 00 80 10 00 00 00  (84 more bytes...) ]
# 	 [SimulationWheel object at 0x26454ca5720]
# 	 steerable (bool): False
# 	 field_8 (int): 1052401205
# 	 surface_handler (SurfaceHandler):
# 	 	unknown (ndarray):
# 	 		[[1.0 0.0 0.0]
# 	 		 [0.0 1.0 0.0]
# 	 		 [0.0 0.0 1.0]
# 	 		 [-0.8849999904632568 0.3525039851665497 -1.2055020332336426]]
# 	 	rotation (ndarray):
# 	 		[[1.0 0.0 0.0]
# 	 		 [0.0 1.0 0.0]
# 	 		 [0.0 0.0 1.0]]
# 	 	position (ndarray): [-0.8849999904632568 0.3398894965648651 -1.2055020332336426]
# 	 field_112 (ndarray):
# 	 	[[1.0 0.0 0.0]
# 	 	 [0.0 1.0 0.0]
# 	 	 [0.0 0.0 1.0]
# 	 	 [0.0 0.0 0.0]]
# 	 field_160 (int): 1065353216
# 	 field_164 (int): 1065353216
# 	 offset_from_vehicle (ndarray): [-0.8849999904632568 0.3525039851665497 -1.2055020332336426]
# 	 real_time_state (RealTimeState):
# 	 	damper_absorb (float): 0.0126144764944911
# 	 	field_4 (float): 0.0003672204620670527
# 	 	field_8 (float): 0.009866210632026196
# 	 	field_12 (ndarray):
# 	 		[[1.0 0.0 0.0]
# 	 		 [0.0 1.0 0.0]
# 	 		 [0.0 0.0 1.0]]
# 	 	field_48 (ndarray):
# 	 		[[1.0 -8.679250385057458e-08 0.0]
# 	 		 [8.679250385057458e-08 1.0 -6.942740924387181e-07]
# 	 		 [6.025778834239937e-14 6.942740924387181e-07 1.0]]
# 	 	field_84 (ndarray): [-0.8849956393241882 -0.014242943376302719 -1.205500602722168]
# 	 	field_108 (float): -5.014799739910814e-07
# 	 	has_ground_contact (bool): True
# 	 	contact_material_id (int): 16
# 	 	is_sliding (bool): False
# 	 	relative_rotz_axis (ndarray): [-1.1801719665527344e-05 -6.981645128689706e-07 1.0]
# 	 	nb_ground_contacts (int): 1
# 	 	field_144 (ndarray): [9.025562576425727e-06 0.9999999403953552 6.726849051119643e-07]
# 	 	rest (bytearray): [ EB 8C E3 40 00 00 00 00 00 00 00 00 ]
# 	 field_348 (int): 0
# 	 contact_relative_local_distance (ndarray): [0.0 0.0 0.0]
# 	 prev_sync_wheel_state (WheelState):
# 	 	rest (bytearray): [ 8C 9D 4E 3C EB 8C E3 40 00 00 00 00 10 00 13 44  (84 more bytes...) ]
# 	 sync_wheel_state (WheelState):
# 	 	rest (bytearray): [ F3 AC 4E 3C EB 8C E3 40 00 00 00 00 10 00 13 44  (84 more bytes...) ]
# 	 field_564 (WheelState):
# 	 	rest (bytearray): [ 68 A8 4E 3C EC 8C E3 40 00 00 00 00 10 00 14 44  (84 more bytes...) ]
# 	 async_wheel_state (WheelState):
# 	 	rest (bytearray): [ 87 AC 4E 3C EB 8C E3 40 00 00 00 00 10 00 14 44  (84 more bytes...) ]
# 	 [SimulationWheel object at 0x26454ca5b40]
# 	 steerable (bool): False
# 	 field_8 (int): 1052401205
# 	 surface_handler (SurfaceHandler):
# 	 	unknown (ndarray):
# 	 		[[1.0 0.0 0.0]
# 	 		 [0.0 1.0 0.0]
# 	 		 [0.0 0.0 1.0]
# 	 		 [0.8850020170211792 0.3525039851665497 -1.2055020332336426]]
# 	 	rotation (ndarray):
# 	 		[[1.0 0.0 0.0]
# 	 		 [0.0 1.0 0.0]
# 	 		 [0.0 0.0 1.0]]
# 	 	position (ndarray): [0.8850020170211792 0.33988550305366516 -1.2055020332336426]
# 	 field_112 (ndarray):
# 	 	[[1.0 0.0 0.0]
# 	 	 [0.0 1.0 0.0]
# 	 	 [0.0 0.0 1.0]
# 	 	 [0.0 0.0 0.0]]
# 	 field_160 (int): 1065353216
# 	 field_164 (int): 1065353216
# 	 offset_from_vehicle (ndarray): [0.8850020170211792 0.3525039851665497 -1.2055020332336426]
# 	 real_time_state (RealTimeState):
# 	 	damper_absorb (float): 0.0126184755936265
# 	 	field_4 (float): -0.000434927613241598
# 	 	field_8 (float): 0.009858638048171997
# 	 	field_12 (ndarray):
# 	 		[[1.0 0.0 0.0]
# 	 		 [0.0 1.0 0.0]
# 	 		 [0.0 0.0 1.0]]
# 	 	field_48 (ndarray):
# 	 		[[1.0 -8.679245411258307e-08 0.0]
# 	 		 [8.679245411258307e-08 1.0 -6.942739787518803e-07]
# 	 		 [6.025774090855432e-14 6.942739787518803e-07 1.0]]
# 	 	field_84 (ndarray): [0.8850238919258118 -0.0142588559538126 -1.2054948806762695]
# 	 	field_108 (float): -5.014799739910814e-07
# 	 	has_ground_contact (bool): True
# 	 	contact_material_id (int): 1142095888
# 	 	is_sliding (bool): False
# 	 	relative_rotz_axis (ndarray): [-1.1801719665527344e-05 -6.981645128689706e-07 1.0]
# 	 	nb_ground_contacts (int): 1
# 	 	field_144 (ndarray): [8.938776772993151e-06 0.9999999403953552 6.726838819304248e-07]
# 	 	rest (bytearray): [ EB 8C E3 40 00 00 00 00 00 00 00 00 ]
# 	 field_348 (int): 0
# 	 contact_relative_local_distance (ndarray): [0.0 0.0 0.0]
# 	 prev_sync_wheel_state (WheelState):
# 	 	rest (bytearray): [ F7 CF 4E 3C EB 8C E3 40 00 00 00 00 10 00 02 43  (84 more bytes...) ]
# 	 sync_wheel_state (WheelState):
# 	 	rest (bytearray): [ B9 BD 4E 3C EB 8C E3 40 00 00 00 00 10 00 02 43  (84 more bytes...) ]
# 	 field_564 (WheelState):
# 	 	rest (bytearray): [ 16 BB 4E 3C EC 8C E3 40 00 00 00 00 10 00 04 43  (84 more bytes...) ]
# 	 async_wheel_state (WheelState):
# 	 	rest (bytearray): [ 5E C4 4E 3C EB 8C E3 40 00 00 00 00 10 00 04 43  (84 more bytes...) ]]
# plug_solid (bytearray): [ 00 00 80 3F 90 C2 F5 3E 00 00 00 00 00 00 00 00  (52 more bytes...) ]
# cmd_buffer_core (bytearray): [ 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  (248 more bytes...) ]
# player_info (PlayerInfoStruct):
# 	team (int): 0
# 	prev_race_time (int): -1
# 	race_start_time (int): 2600
# 	race_time (int): 61930
# 	race_best_time (int): 24530
# 	lap_start_time (int): 2600
# 	lap_time (int): 61930
# 	lap_best_time (int): -1
# 	min_respawns (int): 0
# 	nb_completed (int): 0
# 	max_completed (int): 0
# 	stunts_score (int): 0
# 	best_stunts_score (int): 0
# 	cur_checkpoint (int): 0
# 	average_rank (float): 0.0
# 	current_race_rank (int): 32
# 	current_round_rank (int): 0
# 	current_time (int): 0
# 	race_state (int): 1
# 	ready_enum (int): 0
# 	round_num (int): 0
# 	offset_current_cp (float): 0.0
# 	cur_lap_cp_count (int): 0
# 	cur_cp_count (int): 0
# 	cur_lap (int): 0
# 	race_finished (bool): False
# 	display_speed (int): 0
# 	finish_not_passed (bool): True
# 	countdown_time (int): 2600
# 	rest (bytearray): [ EA F1 00 00 00 FF 00 00 00 00 00 00 78 C8 B2 00  (16 more bytes...) ]
# internal_input_state (ndarray):
# 	[[CachedInput object at 0x26454ca4d60]
# 	 time (int): 0
# 	 event (Event):
# 	 	time (int): 0
# 	 	input_data (int): 0
# 	 [CachedInput object at 0x26454ca5f90]
# 	 time (int): 0
# 	 event (Event):
# 	 	time (int): 0
# 	 	input_data (int): 0
# 	 [CachedInput object at 0x26454ca5ff0]
# 	 time (int): 0
# 	 event (Event):
# 	 	time (int): 0
# 	 	input_data (int): 0
# 	 [CachedInput object at 0x26454ca5bd0]
# 	 time (int): 0
# 	 event (Event):
# 	 	time (int): 0
# 	 	input_data (int): 0
# 	 [CachedInput object at 0x26454ca5e40]
# 	 time (int): 0
# 	 event (Event):
# 	 	time (int): 0
# 	 	input_data (int): 0
# 	 [CachedInput object at 0x26454ca55d0]
# 	 time (int): 0
# 	 event (Event):
# 	 	time (int): 0
# 	 	input_data (int): 0
# 	 [CachedInput object at 0x26454ca50c0]
# 	 time (int): 0
# 	 event (Event):
# 	 	time (int): 0
# 	 	input_data (int): 0
# 	 [CachedInput object at 0x26454ca55a0]
# 	 time (int): 0
# 	 event (Event):
# 	 	time (int): 0
# 	 	input_data (int): 0
# 	 [CachedInput object at 0x26454ca5c00]
# 	 time (int): 0
# 	 event (Event):
# 	 	time (int): 0
# 	 	input_data (int): 0
# 	 [CachedInput object at 0x26454ca4a90]
# 	 time (int): 0
# 	 event (Event):
# 	 	time (int): 0
# 	 	input_data (int): 0                 ]
# input_running_event (Event):
# 	time (int): 100000
# 	input_data (int): 150994945
# input_finish_event (Event):
# 	time (int): 0
# 	input_data (int): 0
# input_accelerate_event (Event):
# 	time (int): 100290
# 	input_data (int): 67108864
# input_brake_event (Event):
# 	time (int): 0
# 	input_data (int): 0
# input_left_event (Event):
# 	time (int): 0
# 	input_data (int): 0
# input_right_event (Event):
# 	time (int): 0
# 	input_data (int): 0
# input_steer_event (Event):
# 	time (int): 0
# 	input_data (int): 0
# input_gas_event (Event):
# 	time (int): 0
# 	input_data (int): 0
# num_respawns (int): 0
# cp_data (CheckpointData):
# 	reserved (int): 0
# 	cp_states_length (int): 3
# 	cp_states (ndarray): [False False False]
# 	cp_times_length (int): 3
# 	cp_times (ndarray):
# 		[[CheckpointTime object at 0x26454ca59c0]
# 		 time (int): -1
# 		 stunts_score (int): 0
# 		 [CheckpointTime object at 0x26454ca5ae0]
# 		 time (int): -1
# 		 stunts_score (int): 0
# 		 [CheckpointTime object at 0x26454ca5630]
# 		 time (int): -1
# 		 stunts_score (int): 0                   ]
