import numpy as np
import pandas as pd


class RestrictActions(object):
    """
    Class RestrictActions. Given a series of episodes consisting of
        (state, action, sprime), build a list of viable actions
        for a given state. Viable is defined as the list of actions
        that have been previously attempted sorted in descending order.
        If only a single action is viable, return a single element list.
    """
    def __init__(self,n_states,n_actions):
        self.n_states = n_states
        self.n_actions = n_actions

    def load_sas_as_episodes(self, episodes, action_col):
        """
        episodes is expected to a pandas dataframe that has already been
        sorted by id, and then bloc (or timestamp). It is expected to
        have at least two columns: state and action.
        """
        self.sas_count = np.zeros((self.n_states, self.n_actions, self.n_states))
        i =0
        for name, transitions in episodes:
            state = transitions['state'].tolist()
            action = transitions[action_col].tolist()
            for i in range(len(state)-2):
                self.sas_count[state[i],action[i],state[i+1]] += 1
        self.build_action_set()

    def load_sas_as_array(self, array):
        self.sas_count = np.zeros((self.n_states, self.n_actions, self.n_states))
        for i in range(len(array)):
            state, action, sprime = array[i]
            self.sas_count[state,action,sprime] += 1
        self.build_action_set()


    def build_action_set(self):
        """
        Assuming the sas_count has been populated, create the list of
        actions for a given state, sorted by frequency
        """
        self.action_freq = np.zeros((self.n_states,self.n_actions))

        for (s,a,sprime), value in np.ndenumerate(self.sas_count):
            if value > 0:
                self.action_freq[int(s)][a] += value


    def get_actions(self, state, min_freq=30):
        """
        For a given state, return the list of appropriate actions sorted
        in descending order by most often seen in the observations
        """
        low_data = np.where(self.action_freq[state] < min_freq)[0].tolist()
        sorted_actions = np.argsort(self.action_freq[state])[::-1].tolist()
        # Remove actions with a low count
        for i in low_data:
            sorted_actions.remove(i)
        return sorted_actions

    def get_consensus(self, state, min_percent=0.9):
        """
        For a given state, the state chosen greater than min_percent of the 
	time.
        """
        low_data = np.where(self.action_freq[state] < 1)[0].tolist()
        sorted_actions = np.argsort(self.action_freq[state])[::-1].tolist()
        # Remove actions with a low count
        for i in low_data:
            sorted_actions.remove(i)
        total_actions = np.sum(self.action_freq[state])
        action = sorted_actions[0] # top choice
        if (self.action_freq[state][action] / total_actions > min_percent):
            return action
        else:
            return None


    def get_actions_per_state(self, state):
        """
        For a given state, return the list of number of times each action
        was taken, ordered by action.
        """
        return self.action_freq[state].tolist()
