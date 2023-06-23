import numpy as np

from . import misc


def load_next_map_zone_centers(zone_centers_filename, base_dir):
    zone_centers = np.load(str(base_dir / "maps" / zone_centers_filename))
    # ========================================================
    # ARTIFICIALLY ADD MORE ZONE CENTERS AFTER THE FINISH LINE
    # ========================================================
    for i in range(misc.n_zone_centers_in_inputs):
        zone_centers = np.vstack(
            (
                zone_centers,
                (2 * zone_centers[-1] - zone_centers[-2])[None, :],
            )
        )
    return zone_centers


# TODO VCP relative positions in inputs
