import numpy as np


class HMM(object):
    """
    Training for Hidden Markov Model (HMM).

    Methods:
    --------------------------------------------------------
    learn_initial: learn the initial probality for the sequence
    learn_transition: learn the transition probality matrix for the sequence
    """
    
    
    def __init__(self, sequence, state):

        """
        Parameters:
        -------------------------------
        sequence: np.array, a time series
        state: int, the number of state
        """
        self.sequence = sequence
        self.state = state
                
    def learn_initial(self):

        """
        learn the initial probality for the sequence
        """
        
        ini_pro = np.zeros((self.state))
        
        unique, counts = np.unique(self.sequence, return_counts=True)
        unique = unique.astype('int64')
        ini_pro[unique] = counts
        
        return ini_pro/ini_pro.sum()
    
        
    def learn_transfer(self):

        """
        learn the transition probality matrix for the sequence
        """
    
        trans_pro = np.zeros((self.state, self.state))
        
        for time in range(len(self.sequence)-1):            
            trans_pro[int(self.sequence[time])][int(self.sequence[time+1])]+=1
            
        for column in xrange(self.state):
            if trans_pro[:,column].any()==0:
                trans_pro[:,column] = np.ones((self.state))
            
        return trans_pro.T/np.sum(trans_pro, axis =1)
