from PPA.MCTS import *
from PPA.StateActionQN import *
import pickle
from PPA.State import *
from PPA.global_constants import *


# Set of StateActionQN obejects that represent the learned model.
Learned_Model = []

# Trajectory recommended for a given encounter.
trajectory_states = []

# Retrieve discretizers.
distance_discretizer, angle_discretizer, speed_discretizer, space_size = setUpdiscretizers()

last_model_index = 0

TRAINING_NUMBER = 0

TRAINING_SET = 'PPA/Training Encounters/Rand_Test_Encounter_Geometries.csv'

# ENCOUNTER
def learnFromEncounter(encounter_directory, encounter_index, mcts: MCST):

    global last_model_index

    print("LEARNING  FROM ", encounter_directory)

    encounter_state = getInitStateFromEncounter(encounter_directory, encounter_index)

    # TODO: Think about this...
    # Sanity check -- are the two aircraft's initial positions well separated
    # by at least the well clear?
    # assert(encounter_state.get_horizontal_distance() >=  DWC_DIST)

    """
        Run Monte Carlo Tree Search:
    """
    # Generate a Monte Carlo Tree Search with initial state
    # at the initial encounter state.
    if mcts is None:
        mcts = MCST(encounter_state)
    
    # Perform selection, expansion, and simulation procedures MCTS_ITERATIONS times.
    """
        For each iteration of the SELECT, EXPAND, and SIMULATE procedure
        we learn something about one or more continuous state and action pairs.
    """

    for i in range(MCTS_ITERATIONS):
        # Try to construct path every 1000 iterations, avoid over-fitting.
        if i != 0 and i % 1000 == 0:
            # Empty the set of (state, action, reward):
            mcts.state_action_reward = []
            # Get State,Action,Rewards for states that updated.
            mcts.getStateActionRewards(mcts.root)
            addModelObjects(mcts)
            print("MODELED: ", len(Learned_Model))
            # Try to construct a path.
            result = constructPathWhileLearning(encounter_state, last_model_index)
            if result == 0:
                print("SUCCESS PATH")
                return

            last_model_index = len(Learned_Model)

        # Run MCTS steps.
        selected_state = mcts.selection()
        mcts.expansion(selected_state)
        mcts.simulate()

    last_model_index = 0
    return mcts


def addModelObjects(mcts):
    """
    :param mcts:
    :return:
    """

    # The set of state,action,rewards that agent learnt from this encounter's iteration.
    for state, action, reward in mcts.state_action_reward:

        if reward == 0:
            # This is a non expanded child - No knowledge about its Q value.
            continue

        # Assume the state,action pair in not modeled.
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
                    knowledge about its Q value via average.
                """
                state_in_model.update(action, reward)
                already_in_model = True
                break

        if already_in_model is True:
            continue
        Learned_Model.append(stateActionQN)


def runEncounters():
    """
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
        Learn from training set:
    """
    for encounter_index in range(NUMBER_OF_ENCOUNTERS):

            # Create a directory for this encounter's description and resulting path after a model test`.
            ENCOUNTER_NAME = f'ENCOUNTER_{encounter_index}'
            ENCOUNTER_PATH = PATH + '/' + ENCOUNTER_NAME
            os.makedirs(ENCOUNTER_PATH)
            
            # Create a .csv file to describe this encounter
            (ENCOUNTERS_GEOMETRIES.iloc[encounter_index]).to_csv(ENCOUNTER_PATH + '/desc.csv', index=False, header=False)
            # Learn:

            mcts = learnFromEncounter(ENCOUNTER_PATH, encounter_index, None)


def constructPathWhileLearning(initial_state: State, last_model_index):
    """
    :param initial_state:
    :param last_model_index:
    :return:
    """
    trajectory_states = [initial_state]

    current_state = initial_state

    while isTerminalState(current_state) == 0:
        model_has_state = False

        current_local_state = convertAbsToLocal(current_state)

        current_discrete_local_state = discretizeLocalState(current_local_state,
                                                            distance_discretizer,
                                                            angle_discretizer,
                                                            speed_discretizer)
        # Generate model object.
        model_lookup = StateActionQN(current_discrete_local_state, '', 0)
        for state_in_model in Learned_Model[last_model_index:]:
            if model_lookup == state_in_model:  # same discrete local state.
                model_has_state = True
                action = state_in_model.getBestAction()
                current_state = getNewState(current_state, action)
                trajectory_states.append(current_state)
                break

        if not model_has_state:
            return -1  # Valid trajectory couldn't be constructed: missing states in model.
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


if __name__ == "__main__":

    space_size_str = "{:e}".format(space_size)

    # Print useful information about the hyper-parameters.

    info_str = f'''
        ****PPA TRAINING****
            STATE SPACE SIZE = {space_size_str}
            # MCTS ITERATIONS = {MCTS_ITERATIONS} 
            GAMMA = {GAMMA}
            EPISODE LENGTH = {EPISODE_LENGTH}
            EXPLORATION FACTOR (C) = {UCB1_C}
            TIME INCREMENT = {TIME_INCREMENT}
            TRAINING SET = {TRAINING_SET}
        ********************
    '''
    print(info_str)

    # Train with the training examples.
    runEncounters()

    print("Final State Space Coverage (%) = ", (len(Learned_Model) / space_size) * 100)

    # open a file, where you want to store the data
    model_str = f'model-{TRAINING_NUMBER}.pickle'
    file = open(model_str, 'wb')

    # dump information to that file
    pickle.dump(Learned_Model, file)

    # close the file
    file.close()
