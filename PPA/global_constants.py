import os

# Algorithm's Constants:

TIME_INCREMENT = 1.0          # Seconds that each action runs for.
GAMMA = 1.0                   # Discount Factor.
DESTINATION_STATE = [0,0]     # Coordinates of the destination.

DESTINATION_STATE_REWARD = 1  # Reward for reaching the destination state.
ABANDON_STATE_REWARD = -0.5   # Negative reward (a penalty) if state is too far from destination.
TIME_REWARD = -0.001          # Negative reward (penalty) for every second that passes.
LODWC_REWARD = -0.3           # Negative reward (penalty) for Lost of Well Clear.
TURN_ACTION_REWARD = -0.00001 # Negative reward (penalty) for every turn action.

# Final State Constants:
DWC_DIST = 2200;              # (ft) Well Clear distance. 
DESTINATION_DIST_ERROR = 500  # (ft) Max distance from destination consider Destination reached.
ABANDON_STATE_ERROR = 51660   # (ft) Distance which if exceed results on an abandon state.

ACTIONS = {                   # Actions are in degrees per second.
    'LEFT': -5,
    'NO_TURN': 0,
    'RIGHT': 5
}

MCTS_ITERATIONS = 1000

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
