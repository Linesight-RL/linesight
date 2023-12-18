import numpy as np


# largely inspired from https://github.com/TomashuTTTT7/TM-AlgoCrack/blob/main/cracks/speedslide_quality.py, yet also largely simplified
def speedslide_quality_tarmac(speed_x: float, speed_z: float):
    material_max_side_friction_multiplier = 1.0  # will need to be changed in the future for dirt & grass
    max_side_friction = (
        np.interp(speed_z * 3.6, [0, 100, 200, 300, 400, 500], [80, 80, 75, 67, 60, 55]) * material_max_side_friction_multiplier
    )
    side_friction = 20 * abs(speed_x)
    if side_friction > max_side_friction:
        return (side_friction - max_side_friction) / max_side_friction
    return 0.0
