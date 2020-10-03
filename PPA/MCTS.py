from random import random
from PPA.State import *
from PPA.LocalState import *
import math


class MCST_State:
    
    def __init__(self, state: State):
        # State properties
        self.state = state
        self.Q = 0
        self.N = 0
        
        # Child states based on the available actions. 
        self.turn_left = None
        self.turn_right = None
        self.no_turn = None

        self.visited_child_count = 0
        
    def __str__(self):
        return f'''
            STATE: {str(self.state)}
            Q: {self.Q}
            N: {self.N}
        '''


class MCST:
    
    def __init__(self, state):
        # Set the MCST initial state.
        self.root = MCST_State(state)
        self.root.N = 1
        self.visitedStatesPath = []             # Keep track of (state, action) pairs along the path to a final state.
        self.lastExpandedState = self.root      # Reference to the last expanded node where simulation starts from.
        self.state_action_reward = []           # List of 3 elements tuples (state,action,reward).
        
    def getBestAction(self):

        # The best action to take from this state is the one with the most simulations.
        simulations_count = [self.root.turn_left.N, self.root.no_turn.N, self.root.turn_right.N]
        action_type = ['LEFT','NO_TURN','RIGHT']
        action = action_type[simulations_count.index(max(simulations_count))]
        
        return action 
    
    def selection(self):

        mcst_node = self.root
        
        # While a given state node has been expanded, select a child using UCB1.
        while mcst_node.visited_child_count == 3: # LEFT, NO_TURN, RIGHT child states have been visited.
                        
            c = math.sqrt(2)
            
            # Explore or exploit...    
            UCB1_left = (mcst_node.turn_left.Q)/(mcst_node.turn_left.N) + c * math.sqrt((math.log(mcst_node.N)/mcst_node.turn_left.N))

            UCB1_right = (mcst_node.turn_right.Q)/(mcst_node.turn_right.N) + c * math.sqrt((math.log(mcst_node.N)/mcst_node.turn_right.N))

            UCB1_no_turn = (mcst_node.no_turn.Q)/(mcst_node.no_turn.N) + c * math.sqrt((math.log(mcst_node.N)/mcst_node.no_turn.N))
            
            values = [UCB1_no_turn, UCB1_left, UCB1_right]
            
            nextChildIndex = values.index(max(UCB1_no_turn,UCB1_left,UCB1_right))

            if nextChildIndex is 0:
                mcst_node = mcst_node.no_turn
                # Keep track of the state actions pair along the path to a final state.
                self.visitedStatesPath.append( (mcst_node, 'NO_TURN') )
            elif nextChildIndex is 1:
                mcst_node = mcst_node.turn_left
                # Keep track of the state actions pair along the path to a final state.
                self.visitedStatesPath.append( (mcst_node, 'LEFT') )
            else:
                # Keep track of the state actions pair along the path to a final state.
                mcst_node = mcst_node.turn_right
                self.visitedStatesPath.append( (mcst_node, 'RIGHT') )
                                              
        return mcst_node
    
    def expansion(self, mcst_node):

        while True:
            rand_num = random()
            if rand_num < 0.33 and mcst_node.no_turn is None:
                
                # Expand to the no_turn state.                              
                new_state = getNewState(mcst_node.state,'NO_TURN')                        
                mcst_node.no_turn = MCST_State(new_state)
                self.lastExpandedState = mcst_node.no_turn                     
                break
            elif rand_num < 0.66 and mcst_node.turn_left is None:
                
                # Expand to the turn_left state.
                new_state = getNewState(mcst_node.state,'LEFT')                        
                mcst_node.turn_left = MCST_State(new_state)
                self.lastExpandedState = mcst_node.turn_left     
                break
            elif rand_num < 0.99 and mcst_node.turn_right is None:
                # Expand to the turn_right state.
                                              
                new_state = getNewState(mcst_node.state,'RIGHT')                        
                mcst_node.turn_right = MCST_State(new_state)
                self.lastExpandedState = mcst_node.turn_right
                break
                
        mcst_node.visited_child_count += 1
        
    def simulate(self):
        # Initial reward for this MCTS node.
        Q = 0  
        # Last state in the MCTS path
        simState = self.lastExpandedState.state

        while True:
            rand_num = random()
            # Select a random action from this state.
            if rand_num < 0.33:
                simState = getNewState(simState,'NO_TURN')
                # No penalty for NO_TURN action.
            elif rand_num < 0.66:
                # TURN_LEFT.
                simState = getNewState(simState,'LEFT')
                Q += TURN_ACTION_REWARD
            else:
                # TURN_RIGHT.
                simState = getNewState(simState,'RIGHT')
                Q += TURN_ACTION_REWARD
            
            Q += TIME_REWARD # Time penalty for every action.
            
            # Check if this state is final.
            state_Q = isTerminalState(simState)
            if state_Q is not 0: # Non-zero means simState is terminal (refer to isTerminalState).
                # Compute Reward/Score and backpropagate.
                Q += state_Q
                break  # End simulation.
            
        # Back-Propagate the reward.
        self.backpropagate(Q)

    def backpropagate(self, Q):

        self.lastExpandedState.Q += Q
        self.lastExpandedState.N += 1
        
        for state_action in self.visitedStatesPath:
            state_action[0].Q += Q
            state_action[0].N += 1

        for state_action in self.visitedStatesPath:
            
            state = state_action[0]
            action = state_action[1]
            
            Q_state = state.Q
            Q_after_action = 0
            
            if action is 'NO_TURN' and state.no_turn is not None:
                Q_after_action = state.no_turn.Q 
            elif action is 'LEFT' and state.turn_left is not None:
                Q_after_action = state.turn_left.Q 
            elif action is 'RIGHT' and state.turn_right is not None:
                Q_after_action = state.turn_right.Q
            
            state_action_Q = Q_after_action - Q_state
            
            self.state_action_reward.append((state.state, action, state_action_Q))
    
        # Empty statesPath for next selection round.
        self.visitedStatesPath.clear()
