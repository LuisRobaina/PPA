class DiscreteLocalState:
    """
        Represents a local state after discretization.
        Every feature is turned into a corresponding bin using the discretizers.
    """

    def __init__(self, d_o_bin, d_i_o_bin, t_d_o_bin, t_i_o_bin, a_r_v_p_bin, o_v_bin, i_v_bin):
        """
            Initialize the discrete features of this discrete state using the resulting
            'bins' for every feature discretized in the continuous state space.
        """
        # Distance ownship to destination bin
        self.dis_ownship_destBIN = d_o_bin
        # Distance angle to destination bin
        self.theta_destintation_ownshipBIN = t_d_o_bin
        # Distance ownship speed bin
        self.ownship_velBIN = o_v_bin
        # Distance intruder speed bin
        self.intruder_velBIN = i_v_bin
        # Distance intruder to ownship bin
        self.dis_int_ownBIN = d_i_o_bin
        # Angle intruder heading relative to ownship heading bin.
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
    
    def __hash__(self):
        # Return hash of the tuple of features.
        return hash((self.dis_ownship_destBIN,
                    self.theta_destintation_ownshipBIN,
                    self.ownship_velBIN,
                    self.intruder_velBIN,
                    self.dis_int_ownBIN,
                    self.theta_int_own_trackBIN,
                    self.angle_rel_vel_neg_rel_posBIN))

    def __eq__(self, obj):
        """
            Two discrete states are the same if they share all bins for discrete
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

        # Discrete states are the same
        return True
