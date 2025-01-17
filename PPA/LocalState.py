from PPA.State import *


class LocalState:
    """
    A Local State is a geometrical representation of a given state in continuous
    2D coordinates. Converting Absolute states to Local states allows us to take
    advantage of symmetry.
    Refer to README for more details about the choice of local features.
    """

    def __init__(self, r_do, theta_do, v_do, r_io, theta_io, psi_io_nr_io, v_i):
        # Distance (ft) ownship to the destination.
        self.distance_ownship_destination = r_do
        # Angle ownship heading relative to destination.
        self.theta_destintation_ownship = theta_do
        # ownship velocity.
        self.ownship_vel = v_do
        # intruder velocity.
        self.intruder_vel = v_i
        # Distance between intruder and ownship.
        self.distance_int_own = r_io
        # Angle intruder heading relative to ownship heading.
        self.theta_int_own_track = theta_io
        self.angle_rel_vel_neg_rel_pos = psi_io_nr_io

    # Return a string representation of a LocalState object.
    def __str__(self):
        return f"""
            distance ownship destination (r_do) = {self.distance_ownship_destination},
            angle destintation ownship (theta_do) = {self.theta_destintation_ownship},
            ownship speed (v_do) = ({self.ownship_vel}),
            distance intruder ownship (r_io) = ({self.distance_int_own}),
            angle intruder ownship track (theta_io) = {self.theta_int_own_track},
            angle of relative velocity w.r.t -(relative position) (psi_io_nr_io) = {self.angle_rel_vel_neg_rel_pos},
            intruder speed (v_i) = {self.intruder_vel}
        """


def convertAbsToLocal(absolute_encounter):
    """
    Given an absolute state S, convert S to a local state L.
    """
    # Ownship position [x, y]
    ownship_pos = np.array(absolute_encounter.ownship_pos)
    # Intruder position [x, y]
    intruder_pos = np.array(absolute_encounter.intruder_pos)

    # Ownship velocity [v_x, v_y]
    ownship_vel = np.array(absolute_encounter.ownship_vel)
    # Intruder velocity [v_x, v_y]
    intruder_vel = np.array(absolute_encounter.intruder_vel)

    # Distance to the destination (ownship).
    destination = np.array(DESTINATION_STATE)
    # [0,0] - [ownship_x, ownship_y].
    dest_ownship_vector = destination - ownship_pos
    # distance to the destination at [0,0].
    distance_ownship_destination = LA.norm(dest_ownship_vector)

    # atan2(y,x).
    theta_destintation_ownship_abs = math.degrees(
        math.atan2(dest_ownship_vector[0], dest_ownship_vector[1]))
    # ownship vel w.r.t y axis.
    psi_o = math.degrees(math.atan2(ownship_vel[0], ownship_vel[1]))

    # Convert the angle to angle between dest_ownship_vector and ownship_vel.
    theta_destintation_ownship = theta_destintation_ownship_abs - psi_o
    if theta_destintation_ownship > 180:
        theta_destintation_ownship = theta_destintation_ownship - 360
    elif theta_destintation_ownship <= -180:
        theta_destintation_ownship = theta_destintation_ownship + 360

    # speed of ownship.
    speed_destination_ownship = LA.norm(destination - ownship_vel)

    intruder_pos_relative_ownship = intruder_pos - ownship_pos
    # distance intruder and ownship.
    distance_intruder_ownship = LA.norm(intruder_pos_relative_ownship)

    # intruder angle w.r.t y axis.
    theta_int_own_orig = math.degrees(math.atan2(
        intruder_pos_relative_ownship[0], intruder_pos_relative_ownship[1]))
    # angle of the intruder pos w.r.t ownship's ground track.
    theta_intruder_own_track = theta_int_own_orig - psi_o

    # tan^-1 (-180,180]
    if theta_intruder_own_track < -180:
        theta_intruder_own_track += 360

    elif theta_intruder_own_track > 180:
        theta_intruder_own_track -= 360

    # speed of the intruder.
    speed_intruder = LA.norm(intruder_vel)

    # Compute the angle between -intruder_pos_relative_ownship and intruder_vel_relative_ownship.
    # This angle is 0 for a straight-to-collision geometry and increases in a clockwise fashion.
    intruder_vel_relative_ownship = intruder_vel - ownship_vel
    # w.r.t. the y-axis (-180, 180]
    psi_io_rel_vel = math.degrees(math.atan2(
        intruder_vel_relative_ownship[0], intruder_vel_relative_ownship[1]))

    # angle of the psi_io_rel_vel vector w.r.t. the -intruder_pos_relative_ownship vector.
    # This angle is 0 for a straight-to-collision geometry.
    angle_rel_vel_neg_rel_pos = psi_io_rel_vel - (theta_int_own_orig - 180)

    if angle_rel_vel_neg_rel_pos <= -180:
        angle_rel_vel_neg_rel_pos += 360
    elif angle_rel_vel_neg_rel_pos > 180:
        angle_rel_vel_neg_rel_pos -= 360

    # Create local state object.
    local_state = LocalState(distance_ownship_destination,
                             theta_destintation_ownship,
                             speed_destination_ownship,
                             distance_intruder_ownship,
                             theta_intruder_own_track,
                             angle_rel_vel_neg_rel_pos,
                             speed_intruder)
    return local_state


def isTerminalState(state: State):
    """
    Is a Local State terminal?
    Returns a non-zero reward for a final state:

        DESTINATION_STATE_REWARD = 1.
        ABANDON_STATE_REWARD = -0.5.
        LODWC_REWARD = -0.3.

    Otherwise return 0 for a non-final states.
    """
    local_state = convertAbsToLocal(state)

    if local_state.distance_ownship_destination <= DESTINATION_DIST_ERROR:
        # Close enough to the destination, reward it.
        return DESTINATION_STATE_REWARD
    if local_state.distance_ownship_destination > ABANDON_STATE_ERROR:
        return ABANDON_STATE_REWARD     # Too far from destination, penalty.
    if local_state.distance_int_own < DWC_DIST:
        return LODWC_REWARD     # Lost of well clear.

    return 0
