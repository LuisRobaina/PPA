from PPA.State import *
from PPA.Global_constants import *

LODWC_ENCOUNTERS = 0


def evaluateEncounters(SET):

    global PATH, TRAINING, LODWC_ENCOUNTERS
    """
    Given the set of training encounters specified in globlal_constants -- TRAINING_SET, iterate over each
    encounter and run MCTS.
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
    ENCOUNTERS_GEOMETRIES = pd.read_csv(SET, header=0)

    # Count the number of rows in set of encounters.
    NUMBER_OF_ENCOUNTERS = len(ENCOUNTERS_GEOMETRIES.index)

    """
        Learn from training set
    """
    for encounter_index in range(NUMBER_OF_ENCOUNTERS):

        # Create a directory for this encounter's description and resulting path after a model test.
        ENCOUNTER_NAME = f'ENCOUNTER_{encounter_index}'
        ENCOUNTER_PATH = PATH + '/' + ENCOUNTER_NAME
        os.makedirs(ENCOUNTER_PATH)

        # Create a .csv file to describe this encounter
        (ENCOUNTERS_GEOMETRIES.iloc[encounter_index]).to_csv(
            ENCOUNTER_PATH + '/desc.csv', index=False, header=False)
        # Learn
        encounter_state = getInitStateFromEncounter(
            ENCOUNTER_PATH, encounter_index)
        # Sanity check -- are the two aircraft's initial positions well separated by at least the well clear?
        try:
            assert (encounter_state.get_distance() >= DWC_DIST)
        except AssertionError:
            log_str = f'''
                        (DWC_DIST ERROR): ENCOUNTER {encounter_index}
                        The two aircraft's initial positions are not well separated by at least the well clear.
                    '''
            print(log_str)
            LODWC_ENCOUNTERS += 1


"""
Main method: User prompts.
"""
if __name__ == "__main__":

    options_prompt = f"""
    ************************************************************************************************
                                            EVALUATE TRAINING SET
                        Evaluate encounters where the two aircraft's initial positions are 
                                not well separated by at least the well clear.
    
    ENCOUNTER_DIR = Path to the set of Encounters to evaluate (csv file).
    ************************************************************************************************
    """

    print(options_prompt)
    ENCOUNTER_DIR = input("ENCOUNTER_DIR: ")
    while True:
        try:
            assert (os.path.exists(ENCOUNTER_DIR))
            break
        except AssertionError as e:
            print("INVALID ENCOUNTER_DIR.")
            ENCOUNTER_DIR = input("ENCOUNTER_DIR: ")

    print("Verify: ", ENCOUNTER_DIR)
    evaluateEncounters(ENCOUNTER_DIR)
    print("CREATED DIRECTORY: ", PATH)
    print("LODWC_ENCOUNTERS = ", LODWC_ENCOUNTERS)
