from PPA.Discretizers import *


class StateActionQN:
    
    def __init__(self, d_state: DiscreteLocalState, action, Q):

        self.discrete_state = d_state
        self.LEFT_Q = 0
        self.NO_TURN_Q = 0
        self.RIGHT_Q = 0
        self.N = 0

        self.update(action, Q)
    
    def getBestAction(self):
        actions = ['LEFT','NO_TURN','RIGHT']
        actionsQ = [self.LEFT_Q, self.NO_TURN_Q, self.RIGHT_Q]
        
        action = actions[actionsQ.index(max(actionsQ))]
        return action
    
    def update(self, action: str, New_Q):
        """
            Update our knowledge of this action
            from this state.
        """
        if action is '':
            return  
        
        if action is "LEFT":
            
            if self.N == 0:
                self.LEFT_Q = New_Q
        
            else:   # Average.
                current_avg = self.LEFT_Q
                new_avg = current_avg + ((New_Q - current_avg)/(self.N+1))
                self.LEFT_Q = new_avg
        
        elif action is "RIGHT":

            if self.N == 0:
                self.RIGHT_Q = New_Q
                
            else:   # Average.
                current_avg = self.RIGHT_Q
                new_avg = current_avg + ((New_Q - current_avg)/(self.N+1))
                self.RIGHT_Q = new_avg
                
        elif action is "NO_TURN":

            if self.N == 0:
                self.NO_TURN_Q = New_Q
                
            else:   # Average.
                current_avg = self.NO_TURN_Q
                new_avg = current_avg + ((New_Q - current_avg)/(self.N+1))
                self.NO_TURN_Q = new_avg
        # Update number of visits to this StateActionQN object.
        self.N += 1
    
    def __str__(self):
        return f'''
            [discrete_state = {self.discrete_state}]
            N(Updates) = {self.N}
            LEFT_Q = {"{:e}".format(self.LEFT_Q)},
            RIGHT_Q = {"{:e}".format(self.RIGHT_Q)},
            NO_TURN_Q = {"{:e}".format(self.NO_TURN_Q)}
        '''

    def __eq__(self, obj):
        return isinstance(obj, StateActionQN) and obj.discrete_state == self.discrete_state
