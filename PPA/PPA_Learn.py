from PPA.MCTS import *
from PPA.StateActionQN import *
import pickle
from PPA.State import *
from PPA.global_constants import *

# Tests Failed
FAILS = 0
PASSED = 0
# Set of StateActionQN that represent the model.
Learned_Model = []

# Trajectory recommended for a given encounter.
trajectory_states = []

# Retrieve discretizers:
distance_discretizer, angle_discretizer, speed_discretizer, space_size = setUpdiscretizers()

ENCOUNTER_MCTS = []


# ENCOUNTER
def learnFromEncounter(encounter_directory, encounter_index, mcts: MCST):
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
        if i % 500 == 0:
            result = constructPathWhileLearning(encounter_state)
            if result == 0:
                print("SUCCESS PATH")
                return
        selected_state = mcts.selection()
        mcts.expansion(selected_state)
        mcts.simulate()

    """
        Iterate the (state,action,reward) tuples learned starting from
        this encounter and convert the states to local coordinates to generate 
        (local_state,action,reward) tuples.
    """

    # Empty the set of (state, action, reward):
    mcts.state_action_reward = []
    # Get State Action Rewards for states that updated.
    mcts.getStateActionRewards(mcts.root)

    # print("UPDATED/LEARNED ABOUT :", len(mcts.state_action_reward))

    # The set of state,action,rewards that it learnt from this encounter iteration.
    for state, action, reward in mcts.state_action_reward:

        if reward == 0:
            # Non expanded child.
            continue

        already_in_model = False

        # Convert to a local state.
        localstate = convertAbsToLocal(state)
        # Discretize the local state.
        discrete_local_state = discretizeLocalState(localstate, 
                                                    distance_discretizer, 
                                                    angle_discretizer, 
                                                    speed_discretizer)
        # Generate model object.
        # print("Creating stateActionQN w: ", reward)
        stateActionQN = StateActionQN(discrete_local_state, action, reward)
        
        for state_in_model in Learned_Model:
            # Check the model in memory:
            if state_in_model == stateActionQN:
                """
                    This discrete local state already
                    exists in our model: Update our 
                    knowledge about it.
                """
                state_in_model.update(action, reward)
                already_in_model = True
                break

        if already_in_model is True:
            # print("Updated:", stateActionQN)
            continue

        # print("NEW. Added to Model:", stateActionQN)
        Learned_Model.append(stateActionQN)

    return mcts


# TODO: COMMENT.
def runEncounters():

    global PATH

    PATH = TEST_RESULTS_PATH

    if not os.path.exists(PATH):
        os.makedirs(PATH)
    elif os.path.exists(PATH):
        i = 1
        PATH += str(i)
        while (os.path.exists(PATH)):
            i += 1
            PATH = PATH[:-1] + str(i)
        os.makedirs(PATH)

    # Header set to 0 because Test_Encounter_Geometries.csv contains headers on first row.
    ENCOUNTERS_GEOMETRIES = pd.read_csv('PPA/Training Encounters/Test_Encounter_Geometries2.csv', header = 0)
    
    NUMBER_OF_ENCOUNTERS = len(ENCOUNTERS_GEOMETRIES.index)     # Count the number of rows in set of encounters.

    """
        Learn from training set:
    """
    for encounter_index in range(NUMBER_OF_ENCOUNTERS):

            # Create a directory for this encounter's description and resulting path after a test.
            ENCOUNTER_NAME = f'ENCOUNTER_{encounter_index}'
            ENCOUNTER_PATH = PATH + '/' + ENCOUNTER_NAME
            os.makedirs(ENCOUNTER_PATH)
            
            # Create a .csv file to describe this encounter
            (ENCOUNTERS_GEOMETRIES.iloc[encounter_index]).to_csv(ENCOUNTER_PATH + '/desc.csv', index=False, header=False)
            # Learn:

            mcts = learnFromEncounter(ENCOUNTER_PATH, encounter_index, None)
            ENCOUNTER_MCTS.append(mcts)

            # # SAMPLE A STATE IN MODEL
            #
            # sample = random.sample(Learned_Model, 2)
            # for actionQN in sample:
            #     print(actionQN)


def constructPathWhileLearning(initial_state: State):

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
        for state_in_model in Learned_Model:
            if model_lookup == state_in_model:  # same discrete local state.
                model_has_state = True
                action = state_in_model.getBestAction()
                current_state = getNewState(current_state, action)
                trajectory_states.append(current_state)
                break

        if not model_has_state:
            print('STATE_NOT_MODELED')
            return -1  # Path couldn't be constructed missing states in model.

    # What final state did we reach?
    """
        Possible final states return values: 
        DESTINATION_STATE_REWARD: Close enough to the destination(Success!)
        ABANDON_STATE_REWARD: Too far from destination (Fails)
        LODWC_REWARD: Lost of well clear (Fails).
    """
    reward = isTerminalState(current_state)
    if reward is DESTINATION_STATE_REWARD:
        # Save path to csv file:
        return 0  # Success path.
    else:
        if reward == ABANDON_STATE_REWARD:
            print('ABANDON_STATE')
        elif reward == LODWC_REWARD:
            print('LODWC')
        return -1  # Failed path.


if __name__ == "__main__":

    space_size_str = "{:e}".format(space_size)

    # Print useful information about the hyper-paramenters.

    print("****PPA TRAINING****")
    print("STATE SPACE SIZE = ", space_size_str)
    print("MCTS ITERATIONS = : ", MCTS_ITERATIONS)
    print("GAMMA : ", GAMMA)
    print("EPISODE LENGTH : ", EPISODE_LENGTH)
    print("EXPLORATION FACTOR (C) : ", UCB1_C)
    print("TIME INCREMENT : ", TIME_INCREMENT)
    print("********************")

    # Train with the training examples.
    runEncounters()

    print("Final Space Coverage (%) = ", (len(Learned_Model) / space_size) * 100)

    # open a file, where you want to store the data
    file = open('model.pickle', 'wb')

    # dump information to that file
    pickle.dump(Learned_Model, file)

    # close the file
    file.close()
