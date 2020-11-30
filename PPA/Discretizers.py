"""
Discretizers.py implements methods to discretize a local state.
"""

from PPA.Global_constants import *
from PPA.DiscreteLocalState import *
import numpy as np
# Refer to sklearn pre-processing KBinsDiscretizer for details about discretization.
from sklearn.preprocessing import KBinsDiscretizer


def setUpdiscretizers():
    """
    Generate and return a set of discretizers for every feature type.
    Discretizer for Distance features.
    Discretizer for Angle features.
    Discretizer for Speed features.
    :return: A set of discretizer objects.
    """

    # Generate the range of integer values for features: Defines all values to consider in discretization.
    # Depends on the MAX and MIN values set in Global_constants.py
    range_distance = (
        np.array([[x for x in range(MIN_DISTANCE, MAX_DISTANCE + 1)]])).T
    range_angle = (np.array([[x for x in range(MIN_ANGLE, MAX_ANGLE + 1)]])).T
    range_speed = (np.array([[x for x in range(MIN_SPEED, MAX_SPEED + 1)]])).T

    # Set the number of bins to use for every feature type.
    """
        The number of bins used for every feature type directly influences the performance of the algorithm
        both in training time (larger state space)  and quality of maneuvers.
    """
    distance_bins = DISTANCE_BINS
    angle_bins = ANGLE_BINS
    speed_bins = SPEED_BINS

    # Generate the discretizer objects using KBinsDiscretizer module.
    # Refer to sklearn KBinsDiscretizers documentation for discretizer types and options.
    distance_discretizer = KBinsDiscretizer(
        n_bins=distance_bins, encode='ordinal', strategy='uniform')
    angle_discretizer = KBinsDiscretizer(
        n_bins=angle_bins, encode='ordinal', strategy='uniform')
    speed_discretizer = KBinsDiscretizer(
        n_bins=speed_bins, encode='ordinal', strategy='uniform')

    # Fit the values for each range into bins using the discretization objects.
    distance_discretizer.fit(range_distance)
    angle_discretizer.fit(range_angle)
    speed_discretizer.fit(range_speed)

    # Compute the discrete state space size. Total of 7 features: 2 distance features, 3 angles features, 2 speed.
    space_size = (distance_bins**2) * (angle_bins**3) * (speed_bins**2)

    # Return the discretizers set to use them during training and testing.
    return [distance_discretizer, angle_discretizer, speed_discretizer, space_size]


def discretizeLocalState(local_state, distance_discretizer, angle_discretizer, speed_discretizer):
    """
    Given a local state find the discretized versiob: Place every continuous feature into bins.
    :param local_state: A local state to discretize.
    :param distance_discretizer: The discretizer to use for distance features.
    :param angle_discretizer: The discretizer to use for angle features.
    :param speed_discretizer: The discretizer to use for speed features.
    :return: A discrete state object.
    """

    # Vector set of distance features
    LocalStateVectorDistances = [
        local_state.distance_ownship_destination,
        local_state.distance_int_own
    ]
    # Vector set of Angle features
    LocalStateVectorAngles = [
        local_state.theta_destintation_ownship,
        local_state.theta_int_own_track,
        local_state.angle_rel_vel_neg_rel_pos
    ]
    # Vector set of Speed features
    LocalStateVectorSpeeds = [
        local_state.ownship_vel,
        local_state.intruder_vel
    ]

    # Discretize features of the local state using the specified discretizers generated in setUpdiscretizers().
    # Returns np.arrays with the bins.
    distance_bins = distance_discretizer.transform(
        (np.array([LocalStateVectorDistances])).T)
    angle_bins = angle_discretizer.transform(
        (np.array([LocalStateVectorAngles])).T)
    speed_bins = speed_discretizer.transform(
        (np.array([LocalStateVectorSpeeds])).T)

    # distance ownship to destination bin
    d_o_bin = distance_bins.T[0][0]
    # Distance intruder to ownship bin
    d_i_o_bin = distance_bins.T[0][1]

    # Angle ownship heading to destination bin.
    t_d_o_bin = angle_bins.T[0][0]
    # Angle Intruder heading relative to ownship heading bin.
    t_i_o_bin = angle_bins.T[0][1]
    a_r_v_p_bin = angle_bins.T[0][2]

    # Speed bins for ownship and intruder.
    o_v_bin = speed_bins.T[0][0]
    i_v_bin = speed_bins.T[0][1]

    # Generate the discrete local state object.
    discreteLocalState = DiscreteLocalState(
        d_o_bin, d_i_o_bin, t_d_o_bin, t_i_o_bin, a_r_v_p_bin, o_v_bin, i_v_bin)

    return discreteLocalState
