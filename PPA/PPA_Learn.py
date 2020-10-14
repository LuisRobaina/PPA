from PPA.MCTS import *
from PPA.StateActionQN import *
import pickle
from PPA.State import *

# Set of StateActionQN that represent the model.
Learned_Model = []

# Trajectory recommended for a given encounter.
trajectory_states = []

# Retrieve discretizers:
distance_discretizer, angle_discretizer, speed_discretizer, space_size = setUpdiscretizers()


def learnFromEncounter(encounter_directory, mcts: MCST):

    print("LEARNING  FROM ", encounter_directory)

    encounter_state = getInitStateFromEncounter(encounter_directory) 

    # Sanity check -- are the two aircraft's initial positions well separated
    # by at least the well clear?
    assert(encounter_state.get_horizontal_distance() >=  DWC_DIST)

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

        #print("Coverage % = ", (len(Learned_Model) / space_size) * 100)

    return mcts


def constructPath(last_traj_state: State, encounter_path, model_index):

    global trajectory_states

    print("CONSTRUCTING TRAJ FOR:", encounter_path)

    current_state = last_traj_state
    
    # Is initial state a terminal state?
    assert(not isTerminalState(current_state))

    while isTerminalState(current_state) == 0:

        model_has_state = False
        
        current_local_state = convertAbsToLocal(current_state)
    
        current_discrete_local_state = discretizeLocalState(current_local_state, 
                                                            distance_discretizer, 
                                                            angle_discretizer, 
                                                            speed_discretizer)
        
        # Generate model object.
        model_lookup = StateActionQN(current_discrete_local_state, '', 0)
        print("Looking for: ", model_lookup)
        print("STATES IN MODEL:", len(Learned_Model))

        # Start checking learned model from last checked state.
        for state_in_model in Learned_Model[model_index:]:
            if state_in_model == model_lookup:  # Same discrete local state.

                model_has_state = True
                action = state_in_model.getBestAction()
                print("Took action: ", action)
                current_state = getNewState(current_state, action)
                trajectory_states.append(current_state)
                model_index = 0
                break

        if not model_has_state:

            # TODO: REMOVE. Taking a random action from here.
            print('STATE_NOT_MODELED. TAKING RAND ACTION.')
            action = random.choice(['LEFT', 'RIGHT', 'NO_TURN'])

            current_state = getNewState(current_state, action )
            trajectory_states.append(current_state)

            #return -1, current_state, len(Learned_Model) -1  # Path couldn't be constructed: Missing state in the model.

    """
        What final state did we reach?
        
        Possible final states return values: 
        DESTINATION_STATE_REWARD: Close enough to the destination(Success!)
        ABANDON_STATE_REWARD: Too far from destination (Fails)
        LODWC_REWARD: Lost of well clear (Fails)
        
    """
    reward = isTerminalState(current_state)

    if reward is DESTINATION_STATE_REWARD:

        # Save path to csv file:
        traj = np.array([trajectory_states])
        traj.tofile(encounter_path+"/" + "Traj.csv", sep=',')
        trajectory_states = []
        return 0, current_state, 0  # Success path.

    else:
        if reward == ABANDON_STATE_REWARD:
            print('ABANDON_STATE')
        elif reward == LODWC_REWARD:
            print('LODWC')
        trajectory_states = []
        return -1, current_state, 0  # Failed path.


def runEncounters():

    # TODO: COMMENT.
    if not os.path.exists(TEST_RESULTS_PATH):
        os.makedirs(TEST_RESULTS_PATH)
    
    # Header set to 0 because Test_Encounter_Geometries.csv contains headers on first row.
    ENCOUNTERS_GEOMETRIES = pd.read_csv('PPA/Training Encounters/Test_Encounter_Geometries.csv', header=0)
    
    NUMBER_OF_ENCOUNTERS = len(ENCOUNTERS_GEOMETRIES.index) # Count the number of rows in set of encounters.

    """
        Learn from training set:
    """
    for encounter_index in range(NUMBER_OF_ENCOUNTERS):

            # Create a directory for this encounter's description and resulting path after a test.
            ENCOUNTER_NAME = f'ENCOUNTER_{encounter_index}'
            ENCOUNTER_PATH = TEST_RESULTS_PATH + '/' + ENCOUNTER_NAME
            os.makedirs(ENCOUNTER_PATH)
            
            # Create a .csv file to describe this encounter
            (ENCOUNTERS_GEOMETRIES.iloc[0]).to_csv(ENCOUNTER_PATH + '/desc.csv', index=False, header=False)
            # Learn:
            mcts = learnFromEncounter(ENCOUNTER_PATH, None)
            init_state = getInitStateFromEncounter(ENCOUNTER_PATH)

            found_traj = False
            last_traj_state = init_state
            trajectory_states.append(init_state)
            last_model_index = 0

            while not found_traj:
                # Try to construct path and learn.
                outcome, last_traj_state, last_model_index = constructPath(last_traj_state,
                                                                           ENCOUNTER_PATH,
                                                                           last_model_index)

                if outcome == -1:   # Failed Path.
                        # Start Learning Again...
                        learnFromEncounter(ENCOUNTER_PATH, mcts)
                if outcome == 0:    # Success Path:
                    print("Optimal Trajectory Found ", ENCOUNTER_NAME)
                    found_traj = True


if __name__ == "__main__":

    space_size_str = "{:e}".format(space_size)
    print("STATE SPACE SIZE : ", space_size_str)

    runEncounters()

    with open('model.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(Learned_Model, f, pickle.HIGHEST_PROTOCOL)
