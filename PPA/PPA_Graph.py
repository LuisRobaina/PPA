import plotly
import plotly.plotly as py
import plotly.figure_factory as ff
import numpy as np
import math
from numpy import linalg as LA
import pandas as pd

import os

# Algorithm's Constants:
TIME_INCREMENT = 1.0            # Seconds that each action runs for.
DESTINATION_STATE = [0, 0]      # Coordinates of the destination.

DESTINATION_STATE_REWARD = 1.0  # Reward for reaching the destination state.
ABANDON_STATE_REWARD = -0.5     # Negative reward (a penalty) if state is too far from destination.
TIME_REWARD = -0.001            # Negative reward (penalty) for every second that passes.
LODWC_REWARD = -0.3             # Negative reward (penalty) for Lost of Well Clear.
TURN_ACTION_REWARD = -0.00001   # Negative reward (penalty) for every turn action.

# Final State Constants:
DWC_DIST = 2200                 # (ft) Well Clear distance.
DESTINATION_DIST_ERROR = 500    # (ft) Max distance from destination consider Destination reached.
ABANDON_STATE_ERROR = 51660     # (ft) Distance which if exceed results on an abandon state.

ACTIONS = {                     # Actions are in degrees per second.
    'LEFT': -5,
    'NO_TURN': 0,
    'RIGHT': 5
}

MCTS_ITERATIONS = 1000
# UCB1 Exploration term.
UCB1_C = 2
GAMMA = 0.9                     # Discount Factor.

EPISODE_LENGTH = None

# Directory Paths:
TEST_RESULTS_PATH = os.getcwd() + '/Test Results'

# Conversion Factors:
NMI_TO_FT = 6076.12
HR_TO_SEC = 3600

# For state discretization purposes:
MIN_DISTANCE = 0              # (ft).
MAX_DISTANCE = 60761          # (ft) equivalent to 10 Nautical Miles.

MIN_SPEED = 0                 # (ft/sec).
MAX_SPEED = 287               # About 170 knot (ft/sec).

MIN_ANGLE = -180              # (deg).
MAX_ANGLE = 180               # (deg).


class State:
    # Define a state object type.
    def __init__(self, ownship_pos, intruder_pos, ownship_vel, intruder_vel):
        # np.array all.
        self.ownship_pos = ownship_pos  # [x,y] (ft)
        self.intruder_pos = intruder_pos  # [x,y] (ft)
        self.ownship_vel = ownship_vel  # [v_x,v_y] (ft/sec)
        self.intruder_vel = intruder_vel  # [v_x,v_y] (ft/sec)

    def get_horizontal_distance(self):  # (ft)
        return LA.norm(self.ownship_pos - self.intruder_pos)

    def __str__(self):
        return f"""
            own_pos (ft) = ({self.ownship_pos}),
            own_vel (ft/s) = ({self.ownship_vel}),
            int_pos (ft) = ({self.intruder_pos}),
            int_vel (ft/s) = ({self.intruder_vel})        
        """


def getInitStateFromEncounter(encounter_directory, encounter_index):
    # Load an encounter description from the directory.
    ENCOUNTER_DESC = pd.read_csv(encounter_directory + '/desc.csv')

    # Convert the encounter to a dictionary.
    i = encounter_index + 1
    encounter_properties = ENCOUNTER_DESC.to_dict().get(str(i))
    """
    Note:
    encounter_properties is a dictionary with the following integer keys
    (All keys are integers):
        0: (time_to_CPA_sec), 1: (destination_time_after_CPA_sec), 2: (OIF_CPA), 3: (CPA_distance_ft),
        4: (v_o_kts), 5: (v_i_kts), 6: (int_rel_heading_deg), 7: (total_runs), 8: (depth), 9: (skip).
    """

    # Given the encounter properties, compute the initial state of the system of the two aircrafts.
    encounter_state = computeInitialState(encounter_properties)
    return encounter_state


def computeInitialState(encounter_properties: dict) -> State:
    """
        Compute the ownship and intruder's initial states based on the encounter
        design parameters:
        encounter_properties is a dictionary as follows:
            encounter_properties = {
                0: (time_to_CPA),
                1: (destination_time_after_CPA)
                2: (OIF_CPA),
                3: (CPA_distance_ft),
                4: (v_o_kts)
                5: (v_i_kts),
                6: (int_rel_heading_deg),
                7: (total_runs),
                8: (depth),
                9: (skip)
            }
    """

    """
    For the ownship:
    """
    # Velocity of the ownship.
    # Note: Ownship flights north so the x component of ownship_vel is 0.
    ownship_vel_x = 0  # (ft/s).
    speed_ownship = float(encounter_properties[4])
    ownship_vel_y = speed_ownship * NMI_TO_FT / HR_TO_SEC  # (ft/s).

    # Velocity vector for the ownship.
    ownship_vel = np.array([ownship_vel_x, ownship_vel_y])

    # Position of ownship.
    # Ownship is placed south of the destination point [0,0].
    ownship_x = 0

    # Using time to CPA and time to destination after CPA we can compute the initial y coordinates for the ownship.
    time_to_CPA = float(encounter_properties[0])
    destination_time_after_CPA = float(encounter_properties[1])
    # Place the ownship at the correct y coordinate such that at the given velocity it will take
    # (time_to_CPA + destination_time_after_CPA) seconds to reach the destination [0,0].
    ownship_y = -(time_to_CPA + destination_time_after_CPA) * ownship_vel_y  # (ft).

    # position vector (ownship).
    ownship_pos = np.array([ownship_x, ownship_y])

    """
    For the intruder:
    """
    # Velocity of the intruder:

    intruder_velocity_magnitud = float(encounter_properties[5]) * NMI_TO_FT / HR_TO_SEC  # (ft/s).
    intruder_heading_angle = float(encounter_properties[6])  # degrees.

    intruder_vel_x = intruder_velocity_magnitud * math.sin(math.radians(intruder_heading_angle))
    intruder_vel_y = intruder_velocity_magnitud * math.cos(math.radians(intruder_heading_angle))

    # velocity vector (intruder).
    intruder_vel = np.array([intruder_vel_x, intruder_vel_y])

    # Position of the intruder:

    # Solve for initial position difference vector, delta_pos_t0:
    delta_vel_t0 = intruder_vel - ownship_vel
    delta_vel_magnitud = LA.norm(delta_vel_t0)

    # Get the horizontal distance at CPA.
    S = float(encounter_properties[3])
    # Initial distance btw aircrafts:
    delta_pos_magnitud = math.sqrt((S ** 2) + (time_to_CPA ** 2 * delta_vel_magnitud ** 2))

    # Let the angle between delta_vel_t0 and delta_post_t0 be theta.
    cos_theta = -time_to_CPA * math.sqrt(delta_vel_magnitud ** 2) / delta_pos_magnitud;
    sin_theta_2 = 1 - cos_theta ** 2;

    if (sin_theta_2 < 0 and sin_theta_2 > -1e-15):
        sin_theta_2 = 0;  # ignore numerical impresion.

    sin_theta = math.sqrt(sin_theta_2);

    # Given the angle we can rotate delta_vel_t0 to get delta_post_t0:
    # Clock-wise rotation.
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    delta_post_t0 = (rotation_matrix @ delta_vel_t0) * delta_pos_magnitud / delta_vel_magnitud

    # Assume position of ownship is [0,0] then position of intruder is delta_post_t0.
    # if x position of intruder at CPA is < 0 then ownship passes in front.

    # position vector (intruder).
    intruder_pos = ownship_pos + delta_post_t0
    intruder_pos_at_CPA = intruder_pos + (time_to_CPA * intruder_vel)

    OIF_CPA = bool(encounter_properties[2])

    if intruder_pos_at_CPA[0] <= 0 and OIF_CPA is True or intruder_pos_at_CPA[1] >= 0 and OIF_CPA is False:
        # Rotation performed was the correct roation.
        pass
    else:
        # Perform Counter-clock wise rotation.
        rotation_matrix = np.array([
            [cos_theta, sin_theta],
            [-sin_theta, cos_theta]
        ])
        delta_post_t0 = (rotation_matrix @ delta_vel_t0) * delta_pos_magnitud / delta_vel_magnitud
        intruder_pos = ownship_pos + delta_post_t0

    # Create State object
    encounter_state = State(ownship_pos, intruder_pos, ownship_vel, intruder_vel)
    return encounter_state


def getNewState(state: State, action):
    """
        Returns an instance of State which
        represents a new state after taking an
        action: (q,a) -> q'
    """
    ownship_vel = np.array(state.ownship_vel)
    ownship_pos = np.array(state.ownship_pos)

    # Velocity:
    if action is 'NO_TURN':
        new_vel_own = ownship_vel  # [v_x,v_y] (ft/sec).

        # new_own_pos = ownship_pos # [x,y] (ft)

    else:
        theta = 5  # degs.
        cos_theta = math.cos(math.radians(theta))
        sin_theta = math.sin(math.radians(theta))

        if action is 'LEFT':
            # Perform Counter-clock wise rotation.
            rotation_matrix = np.array([
                [cos_theta, sin_theta],
                [-sin_theta, cos_theta]
            ])
            new_vel_own = rotation_matrix @ ownship_vel

        elif action is 'RIGHT':
            # Perform clock-wise rotation.
            rotation_matrix = np.array([
                [cos_theta, -sin_theta],
                [sin_theta, cos_theta]
            ])
            new_vel_own = rotation_matrix @ ownship_vel

    # Position:
    # For Ownship:
    avg_disp = 0.5 * (new_vel_own + ownship_vel) * TIME_INCREMENT
    new_own_pos = state.ownship_pos + avg_disp  # [x_o,y_o] (ft).

    # For Intruder: Intruder flights at a constant velocity.
    intr_vel = np.array(state.intruder_vel)
    new_vel_intr = intr_vel
    intr_disp = 0.5 * (new_vel_intr + intr_vel) * TIME_INCREMENT
    new_intr_pos = state.intruder_pos + intr_disp

    # New state after the action.
    new_state = State(new_own_pos, new_intr_pos, new_vel_own, new_vel_intr)
    return new_state


class LocalState:

    def __init__(self, r_do, theta_do, v_do, r_io, theta_io, psi_io_nr_io, v_i):
        self.distance_ownship_destination = r_do
        self.theta_destintation_ownship = theta_do
        self.ownship_vel = v_do
        self.intruder_vel = v_i
        self.distance_int_own = r_io
        self.theta_int_own_track = theta_io
        self.angle_rel_vel_neg_rel_pos = psi_io_nr_io

    def __str__(self):
        return f"""
            distance ownship destination (r_do) = {self.distance_ownship_destination},
            angle destintation ownship (theta_do) = {self.theta_destintation_ownship},
            ownship speed (v_do) = ({self.ownship_vel}),
            distance intruder ownship (r_io) = ({self.distance_int_own}),
            angle intruder ownship track (theta_io) = {self.theta_int_own_track},
            angle of relative velocity w.r.t -(relative position) (psi_io_nr_io) = {self.angle_rel_vel_neg_rel_pos},
            intruder speed (v_i) = {self.intruder_vel}
        """

    # For testing:
    def return_as_array(self):
        return np.array([self.distance_ownship_destination,
                         self.theta_destintation_ownship,
                         self.ownship_vel,
                         self.intruder_vel,
                         self.distance_int_own,
                         self.theta_int_own_track,
                         self.angle_rel_vel_neg_rel_pos
                         ])


def convertAbsToLocal(absolute_encounter):
    """
    Given an absolute state convert it to a local state.
    """
    ownship_pos = np.array(absolute_encounter.ownship_pos)
    intruder_pos = np.array(absolute_encounter.intruder_pos)

    ownship_vel = np.array(absolute_encounter.ownship_vel)
    intruder_vel = np.array(absolute_encounter.intruder_vel)

    # Distance to the destination (ownship)
    destination = np.array(DESTINATION_STATE)
    dest_ownship_vector = destination - ownship_pos  # [0,0] - [ownship_x, ownship_y].
    distance_ownship_destination = LA.norm(dest_ownship_vector)  # distance to the destination at [0,0].

    theta_destintation_ownship = math.degrees(math.atan2(dest_ownship_vector[1], dest_ownship_vector[1]))

    psi_o = math.degrees(math.atan2(ownship_vel[0], ownship_vel[1]))  # ownship vel w.r.t y axis.

    speed_destination_ownship = LA.norm(destination - ownship_vel)  # speed of ownship.

    intruder_pos_relative_ownship = intruder_pos - ownship_pos
    distance_intruder_ownship = LA.norm(intruder_pos_relative_ownship)

    # ntruder angle w.r.t y axis.
    theta_int_own_orig = math.degrees(math.atan2(intruder_pos_relative_ownship[0], intruder_pos_relative_ownship[1]))
    theta_intruder_own_track = theta_int_own_orig - psi_o  # angle of the intruder pos w.r.t ownship's ground track.

    # tan^-1 (-180,180]

    if theta_intruder_own_track < -180:
        theta_intruder_own_track += 360

    elif theta_intruder_own_track > 180:
        theta_intruder_own_track -= 360

    # speed of the intruder.
    speed_intruder = LA.norm(intruder_vel)

    # Compute the angle between -intruder_pos_relative_ownship and intruder_vel_relative_ownship.
    # This angle is 0 for a straight-to-collision geometry and increases in a clockwise fashion.
    intruder_vel_relative_ownship = intruder_vel - ownship_vel
    # w.r.t. the y-axis (-180, 180]
    psi_io_rel_vel = math.degrees(math.atan2(intruder_vel_relative_ownship[0], intruder_vel_relative_ownship[1]))

    # angle of the psi_io_rel_vel vector w.r.t. the -intruder_pos_relative_ownship vector.
    # This angle is 0 for a straight-to-collision geometry.
    angle_rel_vel_neg_rel_pos = psi_io_rel_vel - (theta_int_own_orig - 180)

    if angle_rel_vel_neg_rel_pos <= -180:
        angle_rel_vel_neg_rel_pos += 360
    elif angle_rel_vel_neg_rel_pos > 180:
        angle_rel_vel_neg_rel_pos -= 360

    # Create local state.
    local_state = LocalState(distance_ownship_destination,
                             theta_destintation_ownship,
                             speed_destination_ownship,
                             distance_intruder_ownship,
                             theta_intruder_own_track,
                             angle_rel_vel_neg_rel_pos,
                             speed_intruder)
    return local_state


def isTerminalState(state: State):
    """
    Returns a non-zero reward for a final state:

        DESTINATION_STATE_REWARD = 1.
        ABANDON_STATE_REWARD = -0.5.
        LODWC_REWARD = -0.3.

    Otherwise return 0 for non-final states.
    """
    local_state = convertAbsToLocal(state)

    if local_state.distance_ownship_destination <= DESTINATION_DIST_ERROR:
        return DESTINATION_STATE_REWARD  # Close enough to the destination, reward it.
    if local_state.distance_ownship_destination > ABANDON_STATE_ERROR:
        return ABANDON_STATE_REWARD  # Too far from destination, penalty.
    if local_state.distance_int_own < DWC_DIST:
        return LODWC_REWARD  # Lost of well clear.

    return 0


import random
import math

class MCST_State:

    def __init__(self, state: State):
        # State properties
        self.state = state
        self.Q = 0
        self.N = 0

        # Dirty == 1 if this state was updated during simulations.
        self.dirty_bit = 0

        # Child states based on the available actions.
        self.turn_left = None
        self.turn_right = None
        self.no_turn = None

        self.visited_child_count = 0

    def updateQN(self, New_Q):

        if self.N == 0:
            self.Q = New_Q
        else:  # Average.
            current_avg = self.Q
            new_avg = current_avg + ((New_Q - current_avg) / (self.N + 1))
            self.Q = new_avg

    def __str__(self):
        return f'''
            STATE: {str(self.state)}
            Q: {self.Q}
            N: {self.N}
        '''

    def clean(self):
        self.dirty_bit = 0


class MCST:

    def __init__(self, state):
        # Set the MCST initial state.
        self.root = MCST_State(state)
        self.root.N = 1
        self.visitedStatesPath = [self.root]  # Keep track of (state, action) pairs along the path to a final state.
        self.lastExpandedState = self.root  # Reference to the last expanded node where simulation starts from.
        self.state_action_reward = []  # List of 3 elements tuples (state,action,reward).

    def clearStatesPath(self):
        self.visitedStatesPath = [self.root]

    def getBestAction(self):

        # The best action to take from this state is the one with the most simulations.
        simulations_count = [self.root.turn_left.N, self.root.no_turn.N, self.root.turn_right.N]
        action_type = ['LEFT', 'NO_TURN', 'RIGHT']
        action = action_type[simulations_count.index(max(simulations_count))]

        return action

    def selection(self):

        mcst_node = self.root

        # While a given state node has been expanded, select a child using UCB1.
        while mcst_node.visited_child_count == 3:  # LEFT, NO_TURN, RIGHT child states have been expanded.

            # Exploration term:
            c = UCB1_C

            # Explore or exploit? UCB1 formula.
            UCB1_left = mcst_node.turn_left.Q + c * math.sqrt((math.log(mcst_node.N) / mcst_node.turn_left.N))

            UCB1_right = mcst_node.turn_right.Q + c * math.sqrt((math.log(mcst_node.N) / mcst_node.turn_right.N))

            UCB1_no_turn = mcst_node.no_turn.Q + c * math.sqrt((math.log(mcst_node.N) / mcst_node.no_turn.N))

            # TODO: REMOVE...
            # selectStr = f'''
            #         PARENT_N = {mcst_node.N},
            #         LEFT Q: {mcst_node.turn_left.Q}, LN: {mcst_node.turn_left.N},
            #         RIGTH Q: {mcst_node.turn_right.Q}, RN: {mcst_node.turn_right.N},
            #         NO_TURN Q: {mcst_node.no_turn.Q}, NN: {mcst_node.no_turn.N}
            #
            # '''
            # print(selectStr)

            values = [UCB1_no_turn, UCB1_left, UCB1_right]

            nextChildIndex = values.index(max(UCB1_no_turn, UCB1_left, UCB1_right))

            if nextChildIndex is 0:
                mcst_node = mcst_node.no_turn
                # Keep track of the state actions pair along the path to a final state.
            elif nextChildIndex is 1:
                mcst_node = mcst_node.turn_left
                # Keep track of the state actions pair along the path to a final state.
            else:
                # Keep track of the state actions pair along the path to a final state.
                mcst_node = mcst_node.turn_right

            # Add selected node to the Visited States Path.
            self.visitedStatesPath.append(mcst_node)

        return mcst_node

    def expansion(self, mcst_node):

        while True:
            # TODO: REFACTOR USING random.choice([list]).
            rand_num = random.random()
            if rand_num < 0.33 and mcst_node.no_turn is None:
                # Expand to the no_turn state.
                new_state = getNewState(mcst_node.state, 'NO_TURN')
                mcst_node.no_turn = MCST_State(new_state)
                self.lastExpandedState = mcst_node.no_turn
                break
            elif rand_num < 0.66 and mcst_node.turn_left is None:
                # Expand to the turn_left state.
                new_state = getNewState(mcst_node.state, 'LEFT')
                mcst_node.turn_left = MCST_State(new_state)
                self.lastExpandedState = mcst_node.turn_left
                break
            elif rand_num < 0.99 and mcst_node.turn_right is None:
                # Expand to the turn_right state.
                new_state = getNewState(mcst_node.state, 'RIGHT')
                mcst_node.turn_right = MCST_State(new_state)
                self.lastExpandedState = mcst_node.turn_right
                break

        mcst_node.visited_child_count += 1

    def simulate(self):
        # Initial reward for this MCTS node.
        Q = 0
        # Total Discount.
        discount_factor = GAMMA
        # Last state in the MCTS path.
        simState = self.lastExpandedState.state

        steps = 0
        while True:

            # TODO: REFACTOR USING random.choice([list]).
            # TODO: Add discount factor?
            rand_num = random.random()

            # Select a random action from this state.
            if rand_num < 0.33:
                simState = getNewState(simState, 'NO_TURN')
                # No penalty for NO_TURN action.
            elif rand_num < 0.66:
                # TURN_LEFT.
                simState = getNewState(simState, 'LEFT')
                Q += TURN_ACTION_REWARD
            else:
                # TURN_RIGHT.
                simState = getNewState(simState, 'RIGHT')
                Q += TURN_ACTION_REWARD

            Q += TIME_REWARD  # Time penalty for every action.

            # Check if this state is final.
            state_Q = isTerminalState(simState)
            if state_Q is not 0:  # Non-zero means simState is terminal (refer to isTerminalState).
                # Compute Reward/Score and backpropagate.
                Q += state_Q
                break  # End simulation.

            # TODO: NOT SURE ABOUT THIS...
            if EPISODE_LENGTH is not None and steps >= EPISODE_LENGTH:
                break

            discount_factor *= GAMMA

        # Back-Propagate the reward.
        self.backpropagate(discount_factor * Q)

    def backpropagate(self, Q):

        # Update Last Expanded state and mark it as dirty.
        self.lastExpandedState.Q += Q
        self.lastExpandedState.N += 1
        self.lastExpandedState.dirty_bit = 1

        for mcst_state in self.visitedStatesPath:
            # Update Q values and Number of Simulations.
            mcst_state.updateQN(Q)
            mcst_state.N += 1
            # Mark it as dirty.
            mcst_state.dirty_bit = 1
            np.savetxt(outfile, [mcst_state.state.ownship_pos], delimiter=','
                       , fmt='%-7.2f')
            # outfile.write(mcst_state.state.ownship_pos)
        # Empty statesPath for next selection round.
        self.clearStatesPath()

    def getStateActionRewards(self, current_state):

        # Only iterate over nodes that changed, if node did not change
        # its sub-tree will not change...

        # Recursive base case.
        if current_state is None:
            return 0

        # Avoid branches that did not update.
        if current_state.dirty_bit == 0:
            return current_state.Q

        self.state_action_reward.append((current_state.state,
                                         'LEFT',
                                         self.getStateActionRewards(current_state.turn_left)))
        self.state_action_reward.append((current_state.state,
                                         'RIGHT',
                                         self.getStateActionRewards(current_state.turn_right)))
        self.state_action_reward.append((current_state.state,
                                         'NO_TURN',
                                         self.getStateActionRewards(current_state.no_turn)))

        current_state.clean()
        return current_state.Q

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
            selected_state = mcts.selection()
            mcts.expansion(selected_state)
            mcts.simulate()


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
            PATH += str(i)
        os.makedirs(PATH)

    # Header set to 0 because Test_Encounter_Geometries.csv contains headers on first row.
    ENCOUNTERS_GEOMETRIES = pd.read_csv('PPA/Training Encounters/Test_Encounter_Geometries.csv', header=0)

    NUMBER_OF_ENCOUNTERS = len(ENCOUNTERS_GEOMETRIES.index)  # Count the number of rows in set of encounters.

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
        mcts = learnFromEncounter(ENCOUNTER_PATH, encounter_index, None)


outfile = open('2D-1.txt', 'w')

print("****PPA GRAPH****")
print("MCTS ITERATIONS = : ", MCTS_ITERATIONS)
print("GAMMA : ", GAMMA)
print("EPISODE LENGTH : ", EPISODE_LENGTH)
print("EXPLORATION FACTOR (C) : ", UCB1_C)
print("*****************")

runEncounters()
