from PPA.Discretizers import *


class StateActionQN:
    """
    Defines the object type that our model will store:
    The model refers to the file that will store the results of
    our training and will be used to test.
    The model answers the following question: I am at this discrete state,
    what is the best action to take?

    The model is created during training with PPA_Learn.py and loaded
    during testing with PPA_Test.py
    """
    def __init__(self, d_state: DiscreteLocalState, action, Q):
        # Discrete State to be modeled by this object.
        self.discrete_state = d_state
        # The expected reward for taking a left turn.
        self.LEFT_Q = 0
        # The expected reward for going straight turn.
        self.NO_TURN_Q = 0
        # The expected reward for taking a right turn.
        self.RIGHT_Q = 0

        # Number of times I have visited this
        self.NO_TURN_N = 0
        self.LEFT_N = 0
        self.RIGHT_N = 0

        # Initial call to update the Q value of an action.
        self.update(action, Q)
    
    def getBestAction(self):
        """
        Return the action with the highest expected reward.
        :return:
        """
        actions = ['LEFT','NO_TURN','RIGHT']
        # The set of Q values for the actions.
        actionsQ = [self.LEFT_Q, self.NO_TURN_Q, self.RIGHT_Q]
        
        action = actions[actionsQ.index(max(actionsQ))]
        return action
    
    def update(self, action: str, New_Q):
        """
            Update our knowledge of this action
            from this discrete state by averaging with its previous
            Q value.
        """
        if action is '':
            return  
        
        if action is "LEFT":
            
            if self.LEFT_N == 0:
                # First Q value for this action.
                self.LEFT_Q = New_Q
        
            else:   # Average.
                current_avg = self.LEFT_Q
                new_avg = current_avg + ((New_Q - current_avg)/(self.LEFT_N+1))
                self.LEFT_Q = new_avg

            # Update number of visits to this action.
            self.LEFT_N += 1
        elif action is "RIGHT":

            if self.RIGHT_N == 0:
                # First Q value for this action.
                self.RIGHT_Q = New_Q
                
            else:   # Average.
                current_avg = self.RIGHT_Q
                new_avg = current_avg + ((New_Q - current_avg)/(self.RIGHT_N+1))
                self.RIGHT_Q = new_avg

            # Update number of visits to this action.
            self.RIGHT_N += 1

        elif action is "NO_TURN":

            if self.NO_TURN_N == 0:
                # First Q value for this action.
                self.NO_TURN_Q = New_Q
                
            else:   # Average.
                current_avg = self.NO_TURN_Q
                new_avg = current_avg + ((New_Q - current_avg)/(self.NO_TURN_N+1))
                self.NO_TURN_Q = new_avg

            # Update number of visits to this action.
            self.NO_TURN_N += 1

    def __str__(self):
        return f'''
            [discrete_state = {self.discrete_state}]
            LEFT_Q = {"{:e}".format(self.LEFT_Q)},
            RIGHT_Q = {"{:e}".format(self.RIGHT_Q)},
            NO_TURN_Q = {"{:e}".format(self.NO_TURN_Q)}
        '''

    def __eq__(self, obj):
        return isinstance(obj, StateActionQN) and obj.discrete_state == self.discrete_state
