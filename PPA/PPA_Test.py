"""

PPA_Test is used to evaluate the performance of a model generated with PPA_Learn.
This script tries to construct a valid trajectory for a given set of encounters
and outputs that trajectory as a csv file placing each trajectory on the corresponding
encounter directory for later analysis.

"""
from PPA.MCTS import *
from PPA.StateActionQN import *
from PPA.Global_constants import *
import pandas as pd
import csv
import pickle

# Test performance counters.
failedTests = 0
passedTests = 0
LODWCCount = 0
UnknownStateCount = 0
AbandonStateCount = 0

# Keep track of encounters and their results by categories
SUCCESS_LIST = []       # Successful encounters.
LODWC_LIST = []         # Encounters that resulted in Lost Of Well Clear.
UNKNOWNSTATE_LIST = []  # Encounters that resulted in an Un-modeled state.
ABANDONSTATE_LIST = []  # Encounters that resulted in an Abandon state.


def constructPath(initial_state: State, encounter_path, encounter_index):
    """
    Try to construct a path using the knowledge of our model.
    :param initial_state: Initial encounter state.
    :param encounter_path: Directory to store the trajectory csv file.
    :param encounter_index: Index of this encounter.
    """
    global UnknownStateCount, AbandonStateCount, LODWCCount

    # Log
    print("TRAJ FOR:", encounter_path)

    trajectory_states = [initial_state]
    current_state = initial_state
    """
    While a terminal state is not reached keep taking actions as suggested by our model.
    """
    while isTerminalState(current_state) == 0:
        model_has_state = False

        current_local_state = convertAbsToLocal(current_state)
    
        current_discrete_local_state = discretizeLocalState(current_local_state, 
                                                            distance_discretizer, 
                                                            angle_discretizer, 
                                                            speed_discretizer)
        
        # Generate model object.
        model_lookup = StateActionQN(current_discrete_local_state, '', 0)
        for state_in_model in Learned_Model:
            if model_lookup == state_in_model:  # Same discrete local state.
                model_has_state = True
                action = state_in_model.getBestAction()
                # Log action taken.
                print("TOOK ACTION: ", action)
                current_state = getNewState(current_state, action, TEST_TIME_INCREMENT)
                trajectory_states.append(current_state)
                break
        
        if not model_has_state:
            print('STATE_NOT_MODELED')
            UNKNOWNSTATE_LIST.append(encounter_index)
            UnknownStateCount += 1

            writeTraj(encounter_path, trajectory_states)
            return -1   # Path couldn't be constructed missing states in model.
    
    # What final state did we reach?
    """
        Possible final states return values: 
        DESTINATION_STATE_REWARD: Close enough to the destination(Success!)
        ABANDON_STATE_REWARD: Too far from destination (Fails)
        LODWC_REWARD: Lost of well clear (Fails).
    """
    reward = isTerminalState(current_state)
    if reward is DESTINATION_STATE_REWARD:
        # Save trajectory to csv file
        writeTraj(encounter_path, trajectory_states)
        return 0    # Success path.
    else:
        # Save trajectory to csv file
        writeTraj(encounter_path, trajectory_states)
        if reward == ABANDON_STATE_REWARD:
            ABANDONSTATE_LIST.append(encounter_index)
            AbandonStateCount += 1
            print('ABANDON_STATE')

        elif reward == LODWC_REWARD:
            LODWCCount +=1
            LODWC_LIST.append(encounter_index)
            print('LODWC')
        return -1   # Failed path.


def writeTraj(encounter_path, trajectory_states):
    """
    Save the coordinates of each time on a trajectory to a csv file.
    """

    with open(encounter_path + "/" + "Trajectory.csv", 'w', ) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['O_X', 'O_Y', 'I_X', 'I_Y'])

        for state in trajectory_states:
            # Write the ownship and intruder position.
            writer.writerow([state.ownship_pos[0],
                             state.ownship_pos[1],
                             state.intruder_pos[0],
                             state.intruder_pos[1]
                             ])


"""
Main method: User prompts.
"""
if __name__ == "__main__":

    options_prompt = f"""
    ************************************************************************************************
                                            PPA TEST
    ENCOUNTER_DIR = Path to the set of Encounters to test the model on. (csv file).
    MODEL_DIR =  Path to the model pickle file.
    RESULTS_DIR = Path to the directory to store results for each encounter.
    ************************************************************************************************
    """
    print(options_prompt)
    ENCOUNTER_DIR = input("ENCOUNTER_DIR: ")
    while True:
        try:
            assert(os.path.exists(ENCOUNTER_DIR))
            break
        except AssertionError as e:
            print("INVALID ENCOUNTER_DIR.")
            ENCOUNTER_DIR = input("ENCOUNTER_DIR: ")

    MODEL_DIR = input("MODEL_DIR: ")

    while True:
        try:
            assert(os.path.exists(MODEL_DIR))
            break
        except AssertionError as e:
            print("INVALID MODEL_DIR.")
            MODEL_DIR = input("MODEL_DIR: ")

    RESULTS_DIR = input("RESULTS_DIR: ")
    while True:
        try:
            assert(os.path.exists(RESULTS_DIR))
            break
        except AssertionError as e:
            print("INVALID RESULTS_DIR.")
            RESULTS_DIR = input("RESULTS_DIR: ")

    # Header set to 0 because Test_Encounter_Geometries.csv contains headers on first row.
    ENCOUNTERS_GEOMETRIES = pd.read_csv(ENCOUNTER_DIR, header=0)
    NUMBER_OF_ENCOUNTERS = len(ENCOUNTERS_GEOMETRIES.index)  # Count the number of rows in set of encounters.

    print("Testing on ", NUMBER_OF_ENCOUNTERS, " encounters.")

    # Retrieve discretizers:
    distance_discretizer, angle_discretizer, speed_discretizer, space_size = setUpdiscretizers()

    # Set of StateActionQN that represent the model.
    Learned_Model = []

    with open(MODEL_DIR, 'rb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        Learned_Model = pickle.load(f)

    print("MODEL SIZE: ", len(Learned_Model))

    for encounter_index in range(NUMBER_OF_ENCOUNTERS):

        ENCOUNTER_NAME = f'ENCOUNTER_{encounter_index}'
        ENCOUNTER_PATH = RESULTS_DIR + '/' + ENCOUNTER_NAME
        print(ENCOUNTER_PATH)
        init_state = getInitStateFromEncounter(ENCOUNTER_PATH, encounter_index)

        # Try to construct path.
        outcome = constructPath(init_state, ENCOUNTER_PATH, encounter_index)

        if outcome == -1:  # Failed Path.
            failedTests += 1
            print("FAILED: ", ENCOUNTER_NAME)

        if outcome == 0:  # Success Path:
            print("SUCCESS: ", ENCOUNTER_NAME)
            SUCCESS_LIST.append(encounter_index)
            passedTests += 1

    # Log Results.
    results_str = f"""
    ************************************************************************************************
                                            PPA TEST REPORT
        PASSED =  {passedTests}
        FAILED = {failedTests}

        LIST OF SUCCESS = {SUCCESS_LIST}
        SUCCESS % = {passedTests/NUMBER_OF_ENCOUNTERS * 100}

        LODWC = {LODWCCount}
        LIST OF LOWCD = {LODWC_LIST}

        UNKNOWN STATES = {UnknownStateCount}
        LIST OF UNKNOWN STATES FAILS = {UNKNOWNSTATE_LIST}

        ABANDON STATES = {AbandonStateCount}
        ABANDON STATES List = {ABANDONSTATE_LIST}
    ************************************************************************************************
    """
    print(results_str)
