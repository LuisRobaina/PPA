"""
This script will log the set of parameters that are configured to be
used during training in PPA.Global_constants 
run make config to exectue this script.
"""
from PPA.Global_constants import *

info_str = f'''
        *************************PPA TRAINING PARAMETERS********************
        *                                               
        *    # MCTS ITERATIONS = {MCTS_ITERATIONS}  
        *    MCTS CUT = {MCTS_CUT}
        *    GAMMA = {GAMMA}                        
        *    EPISODE LENGTH = {EPISODE_LENGTH}      
        *    EXPLORATION FACTOR (C) = {UCB1_C}      
        *    TIME INCREMENT = {TIME_INCREMENT}     
        *    TRAINING SET = {TRAINING_SET}          
        *                                           
        *               DISCRETE BINS                
        *    ------------------------------------        
        *    DISTANCE BINS = {DISTANCE_BINS}        
        *    SPEED BINS = {SPEED_BINS}              
        *    ANGLE BINS = {ANGLE_BINS}              
        *                                           
        *            Final State Constants
        *     -------------------------------------
        *     DWC_DIST = {DWC_DIST}
        *     DESTINATION_DIST_ERROR = {DESTINATION_DIST_ERROR}
        *     ABANDON_STATE_ERROR = {ABANDON_STATE_ERROR}
        *      
        *            Rewards/Penalties Constants
        *     -------------------------------------
        *     DESTINATION_STATE_REWARD = {DESTINATION_STATE_REWARD}
        *     ABANDON_STATE_REWARD = {ABANDON_STATE_REWARD}
        *     LODWC_REWARD = {LODWC_REWARD}
        *     TURN_ACTION_REWARD = {TURN_ACTION_REWARD}
        *
        ********************************************************************
    '''
print(info_str)   