"""
Run the training/learning process...
Generate a file that represents our model, this file is then loaded into PPA_Test to evaluate the model.
"""
from PPA.MCTS import *
from PPA.StateActionQN import *
import pickle
from PPA.State import *
from PPA.global_constants import *


# Set of StateActionQN objects that represent the learned model.
Learned_Model = []

# Retrieve discretizers to use during training.
distance_discretizer, angle_discretizer, speed_discretizer, space_size = setUpdiscretizers()

last_model_index = 0

# Used to keep track of the training number when creating a results directory.
TRAINING_NUMBER = 0

mcts = None


def learnFromEncounter(encounter_directory, encounter_index):
    """
    Given the directory to an encounter, learn from it.
    :param encounter_directory:
    :param encounter_index:
    :return:
    """
    global last_model_index, mcts

    print("LEARNING  FROM ", encounter_directory)

    encounter_state = getInitStateFromEncounter(encounter_directory, encounter_index)

    # Sanity check -- are the two aircraft's initial positions well separated by at least the well clear?
    try:
        assert (encounter_state.get_distance() >= DWC_DIST)
    except AssertionError:
        print("(DWC_DIST ERROR): SKIPPED ENCOUNTER", encounter_index)
        return

    # Generate a Monte Carlo Tree Search with initial state at this initial encounter state.
    if mcts is None:
        mcts = MCST(encounter_state)
    
    # Perform selection, expansion, and simulation procedures MCTS_ITERATIONS times.
    """
        For each iteration of the SELECT, EXPAND, and SIMULATE procedure
        we learn something about one or more continuous state and action pairs.
    """
    for i in range(MCTS_ITERATIONS):

        # Try to construct path every MCTS_CUT iterations of MCTS: avoid over-fitting.
        if i % MCTS_CUT == 0:

            # Empty the set of (state, action, reward):
            mcts.state_action_reward = []
            # Get State,Action,Rewards for states that updated in the last 1000 iterations.
            mcts.getStateActionRewards(mcts.root)
            # Add/Update model objects.
            addModelObjects(mcts)
            print("STATES MODELED: ", len(Learned_Model))
            # Try to construct a path with the current model.
            result = constructPathWhileLearning(encounter_state, last_model_index)
            if result == 0:
                print("SUCCESS PATH")
                return
            last_model_index = len(Learned_Model)

        """
            Run Monte Carlo Tree Search:
        """
        selected_state = mcts.selection()
        mcts.expansion(selected_state)
        mcts.simulate()

    last_model_index = 0


def addModelObjects(mcts):
    """
    After a number of iterations of MCTS for a given encounter, get the set of (state,action,rewards) and add it
    to the model. If a state is already  in the model update its Q and N values for the given action by averaging.
    :param mcts: The tree with the (state,action,rewards) tuples.
    """

    # The set of state,action,rewards that agent learnt from this encounter's iteration.
    for state, action, reward in mcts.state_action_reward:

        if reward == 0:
            # This is a non expanded child - No knowledge about its Q value.
            continue

        # Assume this state,action pair in not modeled.
        already_in_model = False

        # Convert state to a local state.
        localstate = convertAbsToLocal(state)
        # Discretize the resulting local state.
        discrete_local_state = discretizeLocalState(localstate,
                                                    distance_discretizer,
                                                    angle_discretizer,
                                                    speed_discretizer)
        # Generate a discrete model object.
        stateActionQN = StateActionQN(discrete_local_state, action, reward)

        for state_in_model in Learned_Model:
            # Check the model object is currently in memory:
            if state_in_model == stateActionQN:
                """
                    This discrete local state already
                    exists in our model: Update our 
                    knowledge about its Q value by averaging.
                """
                state_in_model.update(action, reward)
                already_in_model = True
                break

        if already_in_model is True:
            continue
        # Add new Model Object.
        Learned_Model.append(stateActionQN)


def runEncounters():
    """
    Given the set of training encounters specified in globlal_constants -- TRAINING_SET, iterate over each
    encounter and run MCTS.
    :return:
    """

    global PATH, TRAINING_NUMBER

    PATH = TEST_RESULTS_PATH

    # Create a directory for the Training encounters.
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    elif os.path.exists(PATH):
        i = 1
        PATH += str(i)
        while os.path.exists(PATH):
            i += 1
            PATH = PATH[:-1] + str(i)
        os.makedirs(PATH)
        TRAINING_NUMBER = i

    # Header set to 0 because Test_Encounter_Geometries.csv contains headers on first row.
    ENCOUNTERS_GEOMETRIES = pd.read_csv(TRAINING_SET, header=0)
    
    NUMBER_OF_ENCOUNTERS = len(ENCOUNTERS_GEOMETRIES.index)     # Count the number of rows in set of encounters.

    """
        Learn from training set
    """
    for encounter_index in range(NUMBER_OF_ENCOUNTERS):

            # Create a directory for this encounter's description and resulting path after a model test.
            ENCOUNTER_NAME = f'ENCOUNTER_{encounter_index}'
            ENCOUNTER_PATH = PATH + '/' + ENCOUNTER_NAME
            os.makedirs(ENCOUNTER_PATH)
            
            # Create a .csv file to describe this encounter
            (ENCOUNTERS_GEOMETRIES.iloc[encounter_index]).to_csv(ENCOUNTER_PATH + '/desc.csv', index=False, header=False)
            # Learn
            learnFromEncounter(ENCOUNTER_PATH, encounter_index)


def constructPathWhileLearning(initial_state: State, last_model_index):
    """
    After a number of iterations of MCTS for a given encounter, try to construct a trajectory using the current model.
    :param initial_state: Initial state for the trajectory
    :param last_model_index: Used to point to the last checked model object, when new state,action pairs are added
                             to the model only check the recently added  (Avoid checking the whole model list).
    :return:
    """

    current_state = initial_state
    # Begin to construct a trajectory
    while isTerminalState(current_state) == 0: # A return of 0 means the current state is not final.
        # Assume this specific state is not modeled in memory yet.
        model_has_state = False
        # Convert to a local state.
        current_local_state = convertAbsToLocal(current_state)
        # Discretize the local state
        current_discrete_local_state = discretizeLocalState(current_local_state,
                                                            distance_discretizer,
                                                            angle_discretizer,
                                                            speed_discretizer)

        # Generate a dummy model object for comparison purposes.
        model_lookup = StateActionQN(current_discrete_local_state, '', 0)
        # Iterate over the set of modeled objects.
        for state_in_model in Learned_Model[last_model_index:]:
            if model_lookup == state_in_model:  # same discrete local state.
                model_has_state = True
                action = state_in_model.getBestAction()
                current_state = getNewState(current_state, action, TIME_INCREMENT)
                break
        if not model_has_state:
            return -1  # Valid trajectory couldn't be constructed: Missing the current state in model.
    # loop ends: Reached a final state.

    """
        What final state did the agent reached?
        Possible final states return values: 
        DESTINATION_STATE_REWARD: Close enough to the destination(Success!)
        ABANDON_STATE_REWARD: Too far from destination (Fails)
        LODWC_REWARD: Lost of well clear (Fails).
    """
    reward = isTerminalState(current_state)
    if reward is DESTINATION_STATE_REWARD:
        return 0  # Success: A valid trajectory was found.

    else:
        if reward == ABANDON_STATE_REWARD:
            print('ABANDON_STATE')
        elif reward == LODWC_REWARD:
            print('LODWC')
        return -1  # Failed: A valid trajectory couldn't be constructed.


"""
Main method.
"""
if __name__ == "__main__":

    space_size_str = "{:e}".format(space_size)
    # Print useful information about the hyper-parameters.
    info_str = f'''
        ***********PPA TRAINING PARAMETERS**********

            STATE SPACE SIZE = {space_size_str}
            # MCTS ITERATIONS = {MCTS_ITERATIONS} 
            GAMMA = {GAMMA}
            EPISODE LENGTH = {EPISODE_LENGTH}
            EXPLORATION FACTOR (C) = {UCB1_C}
            TIME INCREMENT = {TIME_INCREMENT}
            TRAINING SET = {TRAINING_SET}

        ********************************************
    '''
    print(info_str)

    # Train using the training examples.
    runEncounters()
    # What percentage of the discrete state space did we cover.
    print("Final State Space Coverage (%) = ", (len(Learned_Model) / space_size) * 100)

    """
    Open a file, where you want to store the model.
    This file can be loaded and used with PPA_Test.py to evaluate the  performance of the model.
    """

    model_str = f'model-{TRAINING_NUMBER}.pickle'
    file = open(model_str, 'wb')
    # dump model information to that file.
    pickle.dump(Learned_Model, file)

    # close the file
    file.close()
