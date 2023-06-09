def line_plane_collision_point(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    # https://gist.github.com/TimSC/8c25ca941d614bf48ebba6b473747d72
    # All inputs: 3D numpy arrays. No need for them to be normalized.
    # Output : the intersection point between the line and the plane
    ndotu = planeNormal.dot(rayDirection)

    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    intersection_point = rayPoint + si * rayDirection
    return intersection_point


def fraction_time_spent_in_current_zone(current_zone_center, next_zone_center, current_pos, next_pos):
    # All inputs: 3D numpy arrays. No need for them to be normalized.
    # Output : the intersection point between the line and the plane
    planeNormal = next_zone_center - current_zone_center
    si = -planeNormal.dot(current_pos - (next_zone_center + current_zone_center) / 2) / planeNormal.dot(next_pos - current_pos)
    return 0 if si < 0 else (1 if si > 1 else si)
    # assert 0 <= si <= 1, si
    # return si
