from PPA.MCTS import *
from PPA.StateActionQN import *
import pickle
from PPA.State import *
from PPA.global_constants import *

TRAINING_NUMBER = 0
TRAINING_SET =


def getInitStateFromEncounter(encounter_directory, encounter_index):
    # Load an encounter description from the directory.
    ENCOUNTER_DESC = pd.read_csv(encounter_directory + '/desc.csv')

    # Convert the encounter to a dictionary.
    encounter_properties = ENCOUNTER_DESC.to_dict().get(str(encounter_index))

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
        sin_theta_2 = 0  # ignore numerical impresion.

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


# ENCOUNTER
def learnFromEncounter(encounter_directory, encounter_index, mcts: MCST):
    global last_model_index

    print("LEARNING  FROM ", encounter_directory)

    encounter_state = getInitStateFromEncounter(encounter_directory, encounter_index)

    # TODO: Think about this...
    # Sanity check -- are the two aircraft's initial positions well separated
    # by at least the well clear?
    # assert(encounter_state.get_horizontal_distance() >=  DWC_DIST)

def runEncounters():
    """
    :return:
    """

    # Header set to 0 because Test_Encounter_Geometries.csv contains headers on first row.
    ENCOUNTERS_GEOMETRIES = pd.read_csv(TRAINING_SET, header=0)
    NUMBER_OF_ENCOUNTERS = len(ENCOUNTERS_GEOMETRIES.index)  # Count the number of rows in set of encounters.

    """
        Learn from training set:
    """
    for encounter_index in range(NUMBER_OF_ENCOUNTERS):
        # Create a directory for this encounter's description and resulting path after a model test`.
        ENCOUNTER_NAME = f'ENCOUNTER_{encounter_index}'
        ENCOUNTER_PATH = PATH + '/' + ENCOUNTER_NAME
        os.makedirs(ENCOUNTER_PATH)

        # Create a .csv file to describe this encounter
        print((ENCOUNTERS_GEOMETRIES.iloc[encounter_index]))
        # Learn:

        # encounter_properties = ENCOUNTER_DESC.to_dict().get(str(encounter_index))

        learnFromEncounter(ENCOUNTER_PATH, encounter_index, None)



if __name__ == "__main__":
    # Train with the training examples.
    runEncounters()

