import copy
import itertools
import os
from pathlib import Path
from typing import List

import numpy as np
import numpy.typing as npt
from pygbx import Gbx, GbxType
from scipy.interpolate import make_interp_spline

from config_files import config_copy


def load_next_map_zone_centers(zone_centers_filename, base_dir):
    """
    Load a map.npy file, and artificially add more zone centers before the start line and after the finish line
    """
    zone_centers = np.load(str(base_dir / "maps" / zone_centers_filename))
    zone_centers = np.vstack(
        (
            zone_centers[0]
            + np.expand_dims(zone_centers[0] - zone_centers[1], axis=0)
            * np.expand_dims(np.arange(config_copy.n_zone_centers_extrapolate_before_start_of_map, 0, -1), axis=1),
            zone_centers,
            zone_centers[-1]
            + np.expand_dims(zone_centers[-1] - zone_centers[-2], axis=0)
            * np.expand_dims(np.arange(1, 1 + config_copy.n_zone_centers_extrapolate_after_end_of_map, 1), axis=1),
        )
    )
    # Smoothen the trajectory defined by virtual checkpoints
    zone_centers[5:-5] = 0.5 * (zone_centers[:-10] + zone_centers[10:])
    return zone_centers


def precalculate_virtual_checkpoints_information(zone_centers):
    """
    zone_centers is a 2D array of shape (n_points, 3), containing a list of points on the centerline of the map.
    During the rollout, we will need to use the middle between two consecutive zone_centers.
    We precalculate the coordinates of these middle positions in the "zone_transitions" array.
    If we are in zone_centers[i]:
    - We will calculate distance advanced on segment (zone_transitions[i-1], zone_transitions[i])
    - distance_between_zone_transitions[i-1] represents the length of the current segment (zone_transitions[i-1], zone_transitions[i])
    - distance_from_start_track_to_prev_zone_transition[i-1] contains the sum of segments until zone_transitions[i-1]
    """
    zone_transitions = 0.5 * (zone_centers[1:] + zone_centers[:-1])  # shape: (n_points - 1, 3)
    delta_zone_transitions = zone_transitions[1:] - zone_transitions[:-1]  # shape: (n_points - 1, 3)
    distance_between_zone_transitions = np.linalg.norm(delta_zone_transitions, axis=1)  # shape: (n_points - 2, )
    distance_from_start_track_to_prev_zone_transition = np.hstack(
        (0, np.cumsum(distance_between_zone_transitions))
    )  # shape: (n_points - 1, )
    normalized_vector_along_track_axis = delta_zone_transitions / np.expand_dims(
        distance_between_zone_transitions, axis=-1
    )  # shape: (n_points - 2, 3)
    return (
        zone_transitions,
        distance_between_zone_transitions,
        distance_from_start_track_to_prev_zone_transition,
        normalized_vector_along_track_axis,
    )


def gbx_to_raw_pos_list(gbx_path: Path):
    """
    Read a .gbx file, extract the raw positions of the best ghost included in that file.
    """
    gbx = Gbx(str(gbx_path))
    ghosts = gbx.get_classes_by_ids([GbxType.CTN_GHOST])
    assert len(ghosts) > 0, "The file does not contain any ghost."
    ghost = min(ghosts, key=lambda g: g.cp_times[-1])
    if ghost.num_respawns != 0:
        print("")
        print("------------    Warning: The ghost contains respawns  ---------------")
        print("")
    records_to_keep = round(ghost.race_time / 100)

    print(ghost.race_time, f"ghost has {len(ghost.records)} records and {len(ghost.control_entries)} control entries")
    print("Keeping", records_to_keep, "out of", len(ghost.records), "records for a race time of", ghost.race_time / 1000)

    raw_positions_list = []
    for r in ghost.records[:records_to_keep]:
        raw_positions_list.append(np.array([r.position.x, r.position.y, r.position.z]))

    return raw_positions_list


def densify_raw_pos_list_n_times(raw_pos_list: List[npt.NDArray], n: int):
    interpolation_function = make_interp_spline(x=range(0, n * len(raw_pos_list), n), y=raw_pos_list, k=1)
    return list(interpolation_function(range(0, n * len(raw_pos_list))))


def map_name_from_map_path(map_path):
    gbx = Gbx(str(config_copy.trackmania_base_path / "Tracks" / "Challenges" / Path(map_path.strip("'\""))))
    gbx_challenge = gbx.get_class_by_id(GbxType.CHALLENGE)
    return gbx_challenge.map_name


def replay_personal_record(map_path):
    trackmania_replay_username = config_copy.username if config_copy.is_linux else os.getlogin()
    filename = trackmania_replay_username + "_" + map_name_from_map_path(map_path) + ".Replay.gbx"
    filepath = config_copy.trackmania_base_path / "Tracks" / "Replays" / "Autosaves"
    return filename, filepath


def hide_personal_record_replay(map_path, is_hide):
    filename, filepath = replay_personal_record(map_path)
    if is_hide:
        if os.path.isfile(filepath / filename):
            os.replace(filepath / filename, filepath / (filename + ".bak"))
    else:
        if os.path.isfile(filepath / filename + ".bak"):
            os.replace(filepath / (filename + ".bak"), filepath / filename)


def get_checkpoint_positions_from_gbx(map_path: str):
    """
    Given a challenge.gbx file, return an unordered list of the checkpoint positions on that track.
    <!> Warning: this function assumes that the block size for that map is 32x8x32. This is true for campaign maps, but not for all custom maps.
    """
    g = Gbx(str(config_copy.trackmania_base_path / "Tracks" / "Challenges" / map_path.strip("'\"").replace("\\", "/")))

    challenges = g.get_classes_by_ids([GbxType.CHALLENGE, GbxType.CHALLENGE_OLD])
    if not challenges:
        quit()

    checkpoint_positions = []
    challenge = challenges[0]
    for block in challenge.blocks:
        if "Checkpoint" in block.name or "Line" in block.name:
            checkpoint_positions.append(np.array(block.position.as_array(), dtype="float"))
            if "High" in block.name:  # Added for E03
                checkpoint_positions[-1] += np.array([0, 7 / 8, 0])
            elif block.name in ["StadiumRoadMainCheckpointRight", "StadiumRoadMainCheckpointLeft"]:  # Added for "exceed my tech"
                checkpoint_positions[-1] += np.array([0, 5 / 8, 0])
    checkpoint_positions = np.array(checkpoint_positions) * np.array([32, 8, 32]) + np.array((16, 0, 16))
    return checkpoint_positions


def sync_virtual_and_real_checkpoints(zone_centers: npt.NDArray, map_path: str):
    """
    Given a challenge.gbx file and a list of VCP, return:
        - next_real_checkpoint_positions: a list of points with the same length as the list of VCP
        - max_allowable_distance_to_real_checkpoint: a list of distances with the same length as the list of VCP

    In this function we match each checkpoint with its corresponding closest VCP.
    In game_instance_manager.py, we will enforce that the car can only advance towards the next VCP if it was within 12 meters of the center of the real checkpoint.
    """
    next_real_checkpoint_positions = np.zeros((len(zone_centers), 3))
    max_allowable_distance_to_real_checkpoint = 9999999 * np.ones(len(zone_centers))
    if config_copy.sync_virtual_and_real_checkpoints:
        checkpoint_positions = get_checkpoint_positions_from_gbx(map_path)
        for checkpoint_position in checkpoint_positions:
            dist_vcp_cp = np.linalg.norm(zone_centers - checkpoint_position, axis=1)
            while np.min(dist_vcp_cp) < 12:
                # This while is necessary for multi-lap maps, to identify the multiple VCP that are linked to the same CP
                idx = dist_vcp_cp.argmin()
                next_real_checkpoint_positions[idx, :] = checkpoint_position
                max_allowable_distance_to_real_checkpoint[idx] = 12
                dist_vcp_cp[max(0, idx - 200) : idx + 200] = 99999

    return next_real_checkpoint_positions, max_allowable_distance_to_real_checkpoint


def find_indices_of_positions_near_cut_position(pos_list: List[npt.NDArray], cut_position: npt.NDArray, margin: float):
    """
    Given a list of positions, find the index of the position closest to cut_position.
    Once an index has been chosen, no other index within +- 200 may be found.
    Repeat until there are no points close enough (typically in a multiple lap race, we would repeat n_lap times)
    """
    indices = []
    dist_pos_cut_pos = np.linalg.norm(np.array(pos_list) - cut_position, axis=1)
    while np.min(dist_pos_cut_pos) < margin:
        index = dist_pos_cut_pos.argmin()
        dist_pos_cut_pos[max(0, index - 50) : index + 50] = 9999999
        indices.append(index)
    return sorted(indices)


def analyze_map_cycle(map_cycle):
    """
    Given a map cycle, identify which maps are used for training and testing, and which maps are only used for testing.
    """
    set_all_maps = set(map(lambda x: x[0], (a for a in itertools.chain(*copy.deepcopy(config_copy.map_cycle)))))
    set_maps_trained = set(map(lambda x: x[0], filter(lambda x: x[4], (a for a in itertools.chain(*copy.deepcopy(map_cycle))))))
    set_maps_blind = set_all_maps - set_maps_trained
    return set_maps_trained, set_maps_blind
