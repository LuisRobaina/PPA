"""
MCTS = Monte Carlo Tree Search.

MCTS.py implements the methods necessary to run the MCTS algorithm. This implementation is based on
the details explained here: https://towardsdatascience.com/monte-carlo-tree-search-in-reinforcement-learning-b97d3e743d0f
"""

import random
from PPA.LocalState import *
import math


class MCST_State:
    """
        A MCTS State represents a node on the Monte Carlo Tree (MCT).
        MCTS contains:
            1. state: A state in continuous 2D.
            2. Q: The average expected discounted sum of rewards from this node down the tree.
            3. N: The number of times this node has been selected.
            4. dirty_bit: Whether or not this node was updated on a given iteration of MCTS.
    """

    def __init__(self, state: State):
        # State properties
        self.state = state
        self.Q = 0
        self.N = 0

        # Dirty == 1 if this state was updated during simulations.
        self.dirty_bit = 0

        # Pointers to children states based on the available actions.
        self.turn_left = None
        self.turn_right = None
        self.no_turn = None
        # How many child states of this node have been expanded.
        self.visited_child_count = 0

    def updateQN(self, New_Q):
        """
        When a node on the MCT is part of a path to a new expanded node the Q value of the new
        expanded node via simulation is back-propagated to all the nodes that lead to it on the tree. This
        propagated Q value is averaged with the current Q value of the node.
        The N value is also increased by 1.
        :param New_Q: The new Q value resulting from simulation.
        """
        # First Time Update.
        if self.N == 0:
            self.Q = New_Q
        else:  # Average with current Q value.
            current_avg = self.Q
            new_avg = current_avg + ((New_Q - current_avg) / (self.N + 1))
            self.Q = new_avg

    def __str__(self):
        """
        Return a string represenation of a MCTS object.
        """
        return f'''
            STATE: {str(self.state)}
            Q: {self.Q}
            N: {self.N}
        '''

    def clean(self):
        """
        Once we process an updated node, mark it as clean.
        """
        self.dirty_bit = 0


class MCST:
    """
    Implementation of the procedures needed to construct a MCTS: Selection, Expansion
    Simulation, and Back-propagation.
    """

    def __init__(self, state):
        # Set the MCST initial state.
        self.root = MCST_State(state)
        self.root.N = 1
        # Every iteration there is a sequence of selections that lead to an unknown state to be expanded. Keep track
        # Keep track of (state, action) pairs along the path to a final state.
        self.visitedStatesPath = [self.root]
        # Points to the last expanded node on a given iteration.
        # Reference to the last expanded node where simulation starts from.
        self.lastExpandedState = self.root
        # Sequence of  stats,action,rewards tuples to be used for the model: Refer to README for more details.
        # List of 3 elements tuples (state,action,reward).
        self.state_action_reward = []

    def clearStatesPath(self):
        """
        After processing the nodes, empty it for the next iteration of MCTS.
        """
        self.visitedStatesPath = [self.root]

    def getBestAction(self):
        """
        The best action to take from this node is the one with the most simulations based on UCB1.
        """
        simulations_count = [self.root.turn_left.N,
                             self.root.no_turn.N, self.root.turn_right.N]
        action_type = ['LEFT', 'NO_TURN', 'RIGHT']
        action = action_type[simulations_count.index(max(simulations_count))]

        return action

    def selection(self):
        """
        Selection step on the MCTS iteration.
        Starting  from the root, select the node with the highest UCB1 value.
        """
        mcst_node = self.root

        # We only run selection on nodes that have the 3 children expanded.
        # While a given state node has been expanded, select a child using UCB1.
        # LEFT, NO_TURN and RIGHT child states have been expanded.
        while mcst_node.visited_child_count == 3:

            # Exploration term:
            c = UCB1_C

            # Explore or exploit? UCB1 formula.
            UCB1_left = mcst_node.turn_left.Q + c * \
                math.sqrt((math.log(mcst_node.N) / mcst_node.turn_left.N))

            UCB1_right = mcst_node.turn_right.Q + c * \
                math.sqrt((math.log(mcst_node.N) / mcst_node.turn_right.N))

            UCB1_no_turn = mcst_node.no_turn.Q + c * \
                math.sqrt((math.log(mcst_node.N) / mcst_node.no_turn.N))

            values = [UCB1_no_turn, UCB1_left, UCB1_right]

            nextChildIndex = values.index(
                max(UCB1_no_turn, UCB1_left, UCB1_right))

            if nextChildIndex is 0:
                # Select no_turn child.
                mcst_node = mcst_node.no_turn
            elif nextChildIndex is 1:
                # Select left child.
                mcst_node = mcst_node.turn_left
            else:
                # Select right child.
                mcst_node = mcst_node.turn_right

            # Add selected node to the Visited States Path.
            self.visitedStatesPath.append(mcst_node)

        # Return a selected node that does not have all 3 children node expanded: Used for expansion().
        return mcst_node

    def expansion(self, mcst_node):
        """
        Randomly pick a non expanded child node to run simulation on.
        Node picked to expand is set as lastExpandedNode.
        :param mcst_node: A selected node that does not have all 3 children expanded.
        """

        while True:     # Randomly pick a non expanded node.
            rand_num = random.random()
            if rand_num < 0.33 and mcst_node.no_turn is None:
                # Expand to the no_turn state.
                new_state = getNewState(
                    mcst_node.state, 'NO_TURN', TIME_INCREMENT)
                mcst_node.no_turn = MCST_State(new_state)
                self.lastExpandedState = mcst_node.no_turn
                break
            elif rand_num < 0.66 and mcst_node.turn_left is None:
                # Expand to the turn_left state.
                new_state = getNewState(
                    mcst_node.state, 'LEFT', TIME_INCREMENT)
                mcst_node.turn_left = MCST_State(new_state)
                self.lastExpandedState = mcst_node.turn_left
                break
            elif rand_num < 0.99 and mcst_node.turn_right is None:
                # Expand to the turn_right state.
                new_state = getNewState(
                    mcst_node.state, 'RIGHT', TIME_INCREMENT)
                mcst_node.turn_right = MCST_State(new_state)
                self.lastExpandedState = mcst_node.turn_right
                break

        mcst_node.visited_child_count += 1

    def simulate(self):
        """
        Run simulations on the last expanded node.
        """

        # Initial reward for this MCTS node.
        Q = 0
        # Total Discount.
        discount_factor = GAMMA
        # Last state in the MCTS path where simulation will start.
        simState = self.lastExpandedState.state

        # Number of steps (actions) taken.
        steps = 0
        while True:
            rand_num = random.random()

            # Select a random action from this state.
            if rand_num < 0.33:
                simState = getNewState(simState, 'NO_TURN', TIME_INCREMENT)
                # No penalty for NO_TURN action.
            elif rand_num < 0.66:
                # TURN_LEFT.
                simState = getNewState(simState, 'LEFT', TIME_INCREMENT)
                # penalize for turning.
                Q += TURN_ACTION_REWARD
            else:
                # TURN_RIGHT.
                simState = getNewState(simState, 'RIGHT', TIME_INCREMENT)
                # penalize for turning.
                Q += TURN_ACTION_REWARD

            # Check if this state is final.
            state_Q = isTerminalState(simState)
            # Non-zero means simState is terminal (refer to isTerminalState).
            if state_Q is not 0:
                # Compute Reward/Score and back-propagate.
                Q += state_Q
                break   # End simulation.

            # If there is a limit in the number of actions that can be taken enforce it.
            if EPISODE_LENGTH is not None and steps >= EPISODE_LENGTH:
                break

            discount_factor *= GAMMA

        # Back-Propagate the reward.
        self.backpropagate(discount_factor*Q)

    def backpropagate(self, Q):
        """
        Back-propagate the new Q value up the tree.
        :param Q: Q value to  back-propagate.
        """
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

        # Empty statesPath for next selection round.
        self.clearStatesPath()

    def getStateActionRewards(self, current_state):
        """
        Generate a set of tuples  (state,action,reward) as follows:
        For every node that was updated during this iteration, the Q value of its left action is the Q value
        of its left-child, the Q value of its right action is the Q value of its right-child etc.

        Rational: These tuples answer the question: What is the expected reward I should get if I take action a from
        state S. This expected reward is the expected reward of the state this  action a leads to.

        These tuples are then discretized and added to the model.
        """
        # Only iterate over nodes that changed, if node did not change
        # its sub-tree did not change.

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
