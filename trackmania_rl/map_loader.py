import copy
import itertools

import numpy as np
import numpy.typing as npt
from pygbx import Gbx, GbxType

from config_files import misc_copy


def load_next_map_zone_centers(zone_centers_filename, base_dir):
    zone_centers = np.load(str(base_dir / "maps" / zone_centers_filename))
    # ==================================================================================
    # ARTIFICIALLY ADD MORE ZONE CENTERS AFTER THE FINISH LINE AND BEFORE THE START LINE
    # ==================================================================================
    zone_centers = np.vstack(
        (
            zone_centers[0]
            + np.expand_dims(zone_centers[0] - zone_centers[1], axis=0)
            * np.expand_dims(np.arange(misc_copy.n_zone_centers_extrapolate_before_start_of_map, 0, -1), axis=1),
            zone_centers,
            zone_centers[-1]
            + np.expand_dims(zone_centers[-1] - zone_centers[-2], axis=0)
            * np.expand_dims(np.arange(1, 1 + misc_copy.n_zone_centers_extrapolate_after_end_of_map, 1), axis=1),
        )
    )
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


def sync_virtual_and_real_checkpoints(zone_centers: npt.NDArray, map_path: str):
    next_real_checkpoint_positions = np.zeros((len(zone_centers), 3))
    max_allowable_distance_to_real_checkpoint = 9999999 * np.ones(len(zone_centers))
    if misc_copy.sync_virtual_and_real_checkpoints:
        g = Gbx(str(misc_copy.trackmania_maps_base_path / map_path.strip("'\"")))

        challenges = g.get_classes_by_ids([GbxType.CHALLENGE, GbxType.CHALLENGE_OLD])
        if not challenges:
            quit()

        cp = []
        challenge = challenges[0]
        for block in challenge.blocks:
            if "Checkpoint" in block.name:
                cp.append(np.array(block.position.as_array(), dtype="float"))
        cp = np.array(cp) * 32 + np.array((16, -22, 16))

        for i in range(len(cp)):
            dist_vcp_cp = np.linalg.norm(zone_centers - cp[i], axis=1)
            while np.min(dist_vcp_cp) < 26:
                # This while is necessary for multi-lap maps, to identify the multiple VCP that are linked the the same CPF
                idx = dist_vcp_cp.argmin()
                next_real_checkpoint_positions[idx, :] = cp[i]
                max_allowable_distance_to_real_checkpoint[idx] = 13
                dist_vcp_cp[max(0, idx - 300) : idx + 300] = 99999

    return next_real_checkpoint_positions, max_allowable_distance_to_real_checkpoint


def analyze_map_cycle(map_cycle):
    set_all_maps = set(map(lambda x: x[0], (a for a in itertools.chain(*copy.deepcopy(misc_copy.map_cycle)))))
    set_maps_trained = set(map(lambda x: x[0], filter(lambda x: x[4], (a for a in itertools.chain(*copy.deepcopy(map_cycle))))))
    set_maps_blind = set_all_maps - set_maps_trained
    return set_maps_trained, set_maps_blind
