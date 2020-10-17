from PPA.Discretizers import *
from PPA.MCTS import *
from PPA.StateActionQN import *
from PPA.global_constants import *
import pandas as pd


def constructPath(initial_state: State, encounter_path):

    print("TRAJ FOR:", encounter_path)
    trajectory_states = [initial_state]
    
    current_state = initial_state
    
    # Is initial state a terminal state:
    assert(not isTerminalState(initial_state))

    while isTerminalState(current_state) == 0:
        model_has_state = False
        
        current_local_state = convertAbsToLocal(current_state)
    
        current_discrete_local_state = discretizeLocalState(current_local_state, 
                                                            distance_discretizer, 
                                                            angle_discretizer, 
                                                            speed_discretizer)
        
        # Generate model object.
        model_lookup = StateActionQN(current_discrete_local_state,'',0)
        for state_in_model in Learned_Model:
            if model_lookup == state_in_model: # same discrete local state.
                model_has_state = True
                action = state_in_model.getBestAction()
                current_state = getNewState(current_state, action)
                trajectory_states.append(current_state)
                break
        
        if not model_has_state:
            print('STATE_NOT_MODELED')
            return -1 # Path couldn't be contructed missing states in model.
    
    # What final state did we reach?
    """
        Possible final states return values: 
        DESTINATION_STATE_REWARD: Close enough to the destination(Success!)
        ABANDON_STATE_REWARD: Too far from destination (Fails)
        LODWC_REWARD: Lost of well clear (Fails)
    """
    reward = isTerminalState(current_state)
    if reward is DESTINATION_STATE_REWARD: 
        
        # Save path to csv file:
        df = pd.DataFrame.from_dict(current_state.__dict__,orient='index')
        df.to_csv(encounter_path+"/"+ "Traj.csv")
        
        return 0 # Success path.
    else:
        if reward == ABANDON_STATE_REWARD:
            print('ABANDON_STATE')
        elif reward == LODWC_REWARD:
            print('LODWC')
        return -1 # Failed path.


if __name__ == "__main__":

    assert(os.path.exists(TEST_RESULTS_PATH))

    failedTests = 0
    passedTests = 0

    # Header set to 0 because Test_Encounter_Geometries.csv contains headers on first row.
    ENCOUNTERS_GEOMETRIES = pd.read_csv('PPA/Training Encounters/Test_Encounter_Geometries.csv', header=0)
    NUMBER_OF_ENCOUNTERS = len(ENCOUNTERS_GEOMETRIES.index)  # Count the number of rows in set of encounters.

    # Retrieve discretizers:
    distance_discretizer, angle_discretizer, speed_discretizer, space_size = setUpdiscretizers()

    # Set of StateActionQN that represent the model.
    Learned_Model = []

    for encounter_index in range(NUMBER_OF_ENCOUNTERS):

        ENCOUNTER_NAME = f'ENCOUNTER_{encounter_index}'
        ENCOUNTER_PATH = TEST_RESULTS_PATH + '/' + ENCOUNTER_NAME

        init_state = getInitStateFromEncounter(ENCOUNTER_PATH)

        # Try to construct path and learn.
        outcome = constructPath(init_state, ENCOUNTER_PATH)

        if outcome == -1:  # Failed Path.
            failedTests += 1
            print("FAILED: ", ENCOUNTER_NAME)

        if outcome == 0:  # Success Path:
            print("SUCCESS: ", ENCOUNTER_NAME)
            passedTests += 1

    print("PASSED = ", passedTests)
    print("FAILED = ", failedTests)

