# libraries
import matplotlib.pyplot as plt
import pickle
from PPA.Global_constants import *

options_prompt = f"""
    ************************************************************************************************
    MODEL_DIR: Path to the model pickle file.
    ************************************************************************************************
    """
print(options_prompt)
MODEL_DIR = input("MODEL_DIR: ")

while True:
    try:
        assert(os.path.exists(MODEL_DIR))
        break
    except AssertionError as e:
        print("INVALID MODEL_DIR.")
        MODEL_DIR = input("MODEL_DIR: ")

x = ['D-O-D', 'T-D-O', 'O-V', 'I-V', 'D-I-O', 'T-I-O', 'A-R-N-P']

D_O_Dbin = []
T_D_Obin = []
O_Vbin = []
I_Vbin = []
D_I_Obin = []
T_I_Obin = []
A_R_N_Pbin = []

with open(MODEL_DIR, 'rb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    Learned_Model = pickle.load(f)

for states_in_model_list in Learned_Model:
    for state_in_model in states_in_model_list:
        D_O_Dbin.append(state_in_model.discrete_state.dis_ownship_destBIN)
        T_D_Obin.append(state_in_model.discrete_state.theta_destintation_ownshipBIN)
        O_Vbin.append(state_in_model.discrete_state.ownship_velBIN)
        I_Vbin.append(state_in_model.discrete_state.intruder_velBIN)
        D_I_Obin.append(state_in_model.discrete_state.dis_int_ownBIN)
        T_I_Obin.append(state_in_model.discrete_state.theta_int_own_trackBIN)
        A_R_N_Pbin.append(state_in_model.discrete_state.angle_rel_vel_neg_rel_posBIN)

# multiple line plot
plt.plot([x[0]]*len(D_I_Obin), D_O_Dbin, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.plot([x[1]]*len(T_D_Obin), T_D_Obin, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.plot([x[2]]*len(O_Vbin), O_Vbin, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.plot([x[3]]*len(I_Vbin), I_Vbin, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.plot([x[4]]*len(D_I_Obin), D_I_Obin, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.plot([x[5]]*len(T_I_Obin), T_I_Obin, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.plot([x[6]]*len(A_R_N_Pbin), A_R_N_Pbin, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)

plt.legend()
plt.savefig('discrete_state.png')