from PPA.global_constants import *
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


def setUpdiscretizers():
    
    range_distance = (np.array([[x for x in range(MIN_DISTANCE, MAX_DISTANCE + 1)]])).T
    range_angle = (np.array([[x for x in range(MIN_ANGLE, MAX_ANGLE + 1)]])).T
    range_speed = (np.array([[x for x in range(MIN_SPEED, MAX_SPEED + 1)]])).T

    distance_bins = 42  # 121 # len(range_distance) - 1
    angle_bins = 72     # 72     # len(range_distance) - 1
    speed_bins = 28     # 57     # len(range_distance) - 1

    distance_discretizer = KBinsDiscretizer(n_bins=distance_bins, encode='ordinal', strategy='uniform')
    angle_discretizer = KBinsDiscretizer(n_bins=angle_bins, encode='ordinal', strategy='uniform')
    speed_discretizer = KBinsDiscretizer(n_bins=speed_bins, encode='ordinal', strategy='uniform')

    distance_discretizer.fit(range_distance)
    angle_discretizer.fit(range_angle)
    speed_discretizer.fit(range_speed)

    space_size = (distance_bins**2) * (angle_bins**3) * (speed_bins**2)

    return [distance_discretizer, angle_discretizer, speed_discretizer, space_size]


class DiscreteLocalState:
    """
        Represents a local state after discretization.
    """
    def __init__(self, d_o_bin, d_i_o_bin, t_d_o_bin, t_i_o_bin, a_r_v_p_bin, o_v_bin, i_v_bin):
        """
            Initialize the discrete features of this discrete state using the resulting
            'bins' for every feature discretized in the continuous state space.
        """
        self.dis_ownship_destBIN = d_o_bin
        self.theta_destintation_ownshipBIN = t_d_o_bin
        self.ownship_velBIN = o_v_bin
        self.intruder_velBIN = i_v_bin
        self.dis_int_ownBIN = d_i_o_bin
        self.theta_int_own_trackBIN = t_i_o_bin
        self.angle_rel_vel_neg_rel_posBIN = a_r_v_p_bin
        
    def __str__(self):
        """
            A string representation of this discrete state.
        """
        return f'''
            d_o_bin = {self.dis_ownship_destBIN},
            t_d_o_bin = {self.theta_destintation_ownshipBIN},
            o_v_bin = {self.ownship_velBIN},
            i_v_bin = {self.intruder_velBIN},
            d_i_o_bin = {self.dis_int_ownBIN},
            t_d_o_bin = {self.theta_int_own_trackBIN},
            a_r_v_p_bin = {self.angle_rel_vel_neg_rel_posBIN}
        '''

    def __eq__(self, obj):
        """
            Two discrete states are the same if they share all discrete
            features.
        """
        if not isinstance(obj, DiscreteLocalState):
            return False
        if self.dis_ownship_destBIN != obj.dis_ownship_destBIN:
            return False
        if self.theta_destintation_ownshipBIN != obj.theta_destintation_ownshipBIN:
            return False
        if self.ownship_velBIN != obj.ownship_velBIN:
            return False
        if self.intruder_velBIN != obj.intruder_velBIN:
            return False
        if self.dis_int_ownBIN != obj.dis_int_ownBIN:
            return False
        if self.theta_int_own_trackBIN != obj.theta_int_own_trackBIN:
            return False
        if self.angle_rel_vel_neg_rel_posBIN != obj.angle_rel_vel_neg_rel_posBIN:
            return False
        
        return True


def discretizeLocalState(local_state, distance_discretizer, angle_discretizer, speed_discretizer):
   
    LocalStateVectorDistances = [
        local_state.distance_ownship_destination,
        local_state.distance_int_own
    ]
    LocalStateVectorAngles = [
        local_state.theta_destintation_ownship,
        local_state.theta_int_own_track,
        local_state.angle_rel_vel_neg_rel_pos
    ]  
    LocalStateVectorSpeeds = [
        local_state.ownship_vel,
        local_state.intruder_vel
    ]
   
    # Discretize features of the continuous local state using the specified discretizers.
    distance_bins = distance_discretizer.transform((np.array([LocalStateVectorDistances])).T)
    angle_bins = angle_discretizer.transform((np.array([LocalStateVectorAngles])).T)
    speed_bins = speed_discretizer.transform((np.array([LocalStateVectorSpeeds])).T)
    
    
    d_o_bin = distance_bins.T[0][0]
    d_i_o_bin = distance_bins.T[0][1]
    
    t_d_o_bin = angle_bins.T[0][0]
    t_i_o_bin = angle_bins.T[0][1]
    a_r_v_p_bin = angle_bins.T[0][2]

    o_v_bin = speed_bins.T[0][0]
    i_v_bin = speed_bins.T[0][1]
    
    discreteLocalState = DiscreteLocalState(d_o_bin, d_i_o_bin, t_d_o_bin, t_i_o_bin, a_r_v_p_bin, o_v_bin, i_v_bin)
    
    return discreteLocalState
