# libraries
import matplotlib.pyplot as plt
import pickle

# For state discretization purposes:
MIN_DISTANCE = 0  # (ft).
MAX_DISTANCE = 60761  # (ft) equivalent to 10 Nautical Miles.

MIN_SPEED = 0  # (ft/sec).
MAX_SPEED = 287  # About 170 knot (ft/sec).

MIN_ANGLE = -180  # (deg).
MAX_ANGLE = 180  # (deg).

x = ['D-O-D', 'T-D-O', 'O-V', 'I-V', 'D-I-O', 'T-I-O', 'A-R-N-P']

D_O_Dbin = []
T_D_Obin = []
O_Vbin = []
I_Vbin = []
D_I_Obin = []
T_I_Obin = []
A_R_N_Pbin = []

with open('model.pickle', 'rb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    Learned_Model = pickle.load(f)

for state_in_model in Learned_Model:
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