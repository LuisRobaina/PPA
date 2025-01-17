import os

# Algorithm's Constants:
# Refer to README for more details about rewards/penalties.

# Seconds that each action will runs for (During training)
TIME_INCREMENT = 10.0
# Seconds that each action will run for (During testing)
TEST_TIME_INCREMENT = 10.0

DESTINATION_STATE = [0, 0]      # Coordinates of the destination.

DESTINATION_STATE_REWARD = 1.0  # Reward for reaching the destination.
# Negative reward (a penalty) if ownship reaches state too far from destination.
ABANDON_STATE_REWARD = -0.5
# We define Lost of Well Clear if the distance between aircraft is less than 2200 ft.
# Negative reward (penalty) for Lost of Well Clear.
LODWC_REWARD = -0.3
# Negative reward (penalty) for every turn action.
TURN_ACTION_REWARD = -0.00001

# Final State Constants:
DWC_DIST = 2200                 # (ft) Well Clear distance.

# (ft) Max distance from the destination which is considered Destination Reached.
DESTINATION_DIST_ERROR = 500

# (ft) Distance which if exceed results in an abandon state.
ABANDON_STATE_ERROR = 50000

# Set of actions the ownship can take.
ACTIONS = {                     # Actions are in degrees per second.
    'LEFT': -5,
    'NO_TURN': 0,
    'RIGHT': 5
}

"""
Maximum number of MCTS iterations that can run for a given encounter. 
Each iteration of MCTS includes: selection, expansion, simulation.
Refer to MCTS.py for more details.
"""
MCTS_ITERATIONS = 10000
# Every MCTS_CUT iterations try to construct a trajectory. If it is successful then move to the next training encounter.
# If a cut is not desired set MCTS_CUT = 1. Then every MCTS will go for MCTS_ITERATIONS.
MCTS_CUT = 500

UCB1_C = 3                      # UCB1 Exploration term.
GAMMA = 0.9                     # Discount Factor.

# Max number of actions that can be taken when simulating for Performance.
EPISODE_LENGTH = None

# Directory Paths:
# Set of Training Encounters.
TRAINING_SET = 'PPA/Training Encounters/Test_Encounter_Geometries2.csv'
# Directory where each encounter description/trajectory will be placed.
TEST_RESULTS_PATH = os.getcwd() + '/Test Results'

# Conversion Factors
NMI_TO_FT = 6076.12
HR_TO_SEC = 3600

# For state discretization purposes:
MIN_DISTANCE = 0              # (ft).
MAX_DISTANCE = 60761          # (ft) equivalent to 10 Nautical Miles.

MIN_SPEED = 0                 # (ft/sec).
MAX_SPEED = 287               # About 170 knot (ft/sec).

MIN_ANGLE = -180              # (deg).
MAX_ANGLE = 180               # (deg).

"""
The number of bins used for every feature type directly influences the performance of the algorithm
both in training time (larger state space) and quality of maneuvers.
"""
DISTANCE_BINS = 60  # 121
ANGLE_BINS = 36     # 72
SPEED_BINS = 20     # 57
