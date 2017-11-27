'''
Data organized as sequences of states action and observations:
    states_sequence, actions_sequence, rewards_sequence
Additinaly, some functions use a list of 'fence_posts', marking the beginings
    of new episodes within sequences of states.
    
'''

import numpy as np
import matplotlib.pyplot as plt
import warnings

def turn_policy_to_stochastic_policy(
        pi,
        num_of_states = None,
        num_of_actions = None ):
    if len(np.shape(pi)) == 1:
        if num_of_states is None:
            num_of_states = len(pi)
        if num_of_actions is None:
            num_of_actions = np.max(pi) + 1
            warnings.warn('number of actions not given. calculated number of actions : ' + str(num_of_actions))
        pi_original = np.copy(pi)
        pi = np.zeros( ( num_of_states, num_of_actions ) )
        for si in range(num_of_states):
            pi[si, pi_original[si] ] = 1
    return pi
                
            
def generate_trials(
    T, R, initialize_prob, terminal_states, pi, num_of_trials ):
    # No action (nan) or reward (0) taken in termination state

    num_of_states = np.shape( T )[ 0 ]
    num_of_actions = np.shape( T )[ 1 ]
    pi = turn_policy_to_stochastic_policy( pi, num_of_states, num_of_actions )
    fence_posts = [ ]
    states_list = [ ]
    actions_list = [ ]
    rewards_list = [ ]
    time_i = 0
    for trial_i in range( num_of_trials ):
        fence_posts += [ time_i ]
        states_list += [ np.random.choice( num_of_states , p = initialize_prob ) ]
        terminated = False
        while not terminated:
            actions_list += [ np.random.choice( num_of_actions , p = pi[ 
                states_list[ -1 ], : ] ) ]
            states_list += [ np.random.choice( num_of_states , p = T[
                states_list[ -1 ], actions_list[ -1 ], : ] ) ]
            rewards_list += [ R[ 
                states_list[ -2 ], actions_list[ -1 ], states_list[ -1 ] ] ]
            terminated = states_list[-1] in terminal_states
            time_i += 1
        actions_list += [ np.random.choice( num_of_actions, p = pi[
            states_list[ -1], :] ) ]
        fake_state = np.random.choice( num_of_states , p = T[
            states_list[ -1 ], actions_list[ -1 ], : ] ) 
        rewards_list += [ R[ 
            states_list[ -1 ], actions_list[ -1 ], fake_state ] ]
        time_i += 1

    return states_list, actions_list, rewards_list, fence_posts

def learn_transition_function(
    states_list,
    actions_list,
    terminal_states = None,
    method = 'uniform' ):
    '''
    method : 'uniform' - unobserved state action pairs transition to all 
                         non-terminal states with uniform probability.
    '''

    num_of_states = np.max(states_list) + 1 # includes an absorbing state
    num_of_actions = np.max(actions_list) + 1
    sas_obs = np.zeros((num_of_states, num_of_actions, num_of_states)) # state-action-state-observations
    states_actions_not_observed = np.ones( (num_of_states, num_of_actions) , dtype = bool )
    for oi in range(len(states_list)-1): # Observation index
        sas_obs[states_list[oi], actions_list[oi], states_list[oi+1]] += 1
        states_actions_not_observed[ states_list[oi], actions_list[oi] ] = False
    print( "%5.2f of the states\\actions combinations were not observed" \
           % (np.sum(states_actions_not_observed) / (num_of_actions*num_of_states)) )
    sa_obs = np.sum( sas_obs , axis = 2 ) # state-action-observations
    T = np.zeros((num_of_states, num_of_actions, num_of_states))
    for si in range(num_of_states - 1):
        for ai in range(num_of_actions):
            for stagi in range(num_of_states):
                if sa_obs[ si, ai ] > 0:
                    T[ si, ai, stagi ] = sas_obs[ si, ai, stagi] / sa_obs[ si, ai ]
    if method == 'uniform':
        T_elements_with_unobserved_transitions = \
            np.tile(np.reshape(states_actions_not_observed, \
            ( num_of_states, num_of_actions, 1)), ( 1, 1, num_of_states ))
        T_elements_with_unobserved_transitions[ :, :, -1 ] = False
        T[T_elements_with_unobserved_transitions] = 1 / (num_of_states-1)
    if terminal_states is not None:
        for terminal_state__current in terminal_states:
            T[ terminal_state__current, :, terminal_state__current ] = 1
    
#TODO : make more efficient than three for loops
    return T

def learn_rewards_function(
    states_list,
    actions_list,
    rewards_list,
    default_reward = 0,
    method = None,
    terminal_states = None ):
    
    if method == 'optimistic':
        default_reward = np.max(rewards_list)
    num_of_states = np.max(states_list) + 1 # includes an absorbing state
    num_of_actions = np.max(actions_list) + 1
    sas_obs = np.zeros((num_of_states, num_of_actions, num_of_states)) # state-action-state-observations
    R = np.zeros((num_of_states, num_of_actions, num_of_states))
    for oi in range(len(states_list)-1): # Observation index
        sas_obs[ states_list[oi], actions_list[oi], states_list[oi+1] ] += 1
        R[ states_list[oi], actions_list[oi], states_list[oi+1] ] += \
            rewards_list[oi]
    R /= sas_obs
    R[ sas_obs == 0 ] = default_reward
    if terminal_states is not None:
        for terminal_state__current in terminal_states:
            R[ terminal_state__current, :, terminal_state__current ] = 0
#TODO : deal with cases where some state actions pairs are not observed
    return R

def policy_evaluation( T , R , pi , gamma , theta = 0.1 ):
    
    num_of_states = np.shape( T )[ 0 ]
    num_of_actions = np.shape( T )[ 1 ]
    # Evaluate V
    V = np.zeros( num_of_states )
    converged = False
    while not converged:
        delta = 0
        for si in range( num_of_states ):
            v = V[si]
            if pi.ndim == 2:
                V[si] = np.sum( np.tile( np.reshape( pi[si, :], (1, num_of_actions)), 
                    (num_of_states, 1)) * T[ si, :, :].T * (R[si, :, :].T + 
                    gamma * np.tile( np.reshape( V, ( num_of_states, 1) ), ( 1, num_of_actions) ) ) ) 
            elif pi.ndim == 1:
                V[si] = np.sum( T[ si, pi[si], : ] * ( R[ si, pi[si], : ] + gamma * V ) ) 
            delta = max( delta , abs( v - V[si] ) )
        converged = delta < theta
    # Evaluate Q
    Q = np.zeros( ( num_of_states, num_of_actions ) )
    for state_ind in range( num_of_states ):
        for action_ind in range( num_of_actions ):
            Q[ state_ind, action_ind ] = np.sum( T[ state_ind, action_ind, : ] * \
                ( R[ state_ind, action_ind, : ] + gamma * V ) )
    
    return V, Q


def policy_evaluation_slow( T , R , pi , gamma , theta = 1 ):
# a slower but more readable version
    
    num_of_states = np.shape( T )[ 0 ]
    num_of_actions = np.shape( T )[ 1 ]
    pi = turn_policy_to_stochastic_policy( pi, num_of_states, num_of_actions )
    # Evaluate V
    V = np.zeros( num_of_states )
#    V = 100 * np.ones( num_of_states )
    converged = False
    while not converged:
        delta = 0
        for si in range( num_of_states ):
            v = V[si]
            temp = 0
            for ai in range(num_of_actions):
                for sit in range(num_of_states):
                    temp += pi[si, ai] * T[si,ai,sit] *(R[si,ai,sit]+gamma*V[si])
            V[si] = temp
            delta = max( delta , abs( v - V[si] ) )
        converged = delta < theta
    # Evaluate Q
    Q = np.zeros( ( num_of_states, num_of_actions ) )
    for state_ind in range( num_of_states ):
        for action_ind in range( num_of_actions ):
            Q[ state_ind, action_ind ] = np.sum( T[ state_ind, action_ind, : ] * \
                ( R[ state_ind, action_ind, : ] + gamma * V ) )
    
    return V, Q

def value_iteration( T , R , gamma , theta = 0.1 ):
    
    num_of_states = np.shape( T )[ 0 ]
    num_of_actions = np.shape( T )[ 1 ]
    V = np.zeros( num_of_states )
    converged = False
    while not converged:
        delta = 0
        for si in range( num_of_states ):
            v = np.copy(V[si])
            
            possible_V_values = T[ si, :, : ] * \
                ( R[ si, :, : ] + gamma * np.tile( np.reshape( V, (1, num_of_states ) ),\
                ( num_of_actions, 1 ) ) )
            possible_V_values = np.sum( possible_V_values, axis = 1 )
            V[si] = np.max( possible_V_values )
            
            delta = max( delta, abs(v-V[si]) )
        converged = delta < theta  
#        print(delta)
    
    pi = np.zeros( num_of_states )
    for si in range( num_of_states ):
        possible_V_values = T[ si, :, : ] * \
            ( R[ si, :, : ] + gamma * np.tile( np.reshape( V, (1, num_of_states ) ),\
            ( num_of_actions, 1 ) ) )
        possible_V_values = np.sum( possible_V_values, axis = 1 )
        pi[si] = np.argmax(possible_V_values)
    pi = pi.astype(int)
    
    return pi , V

################################################################################
#####  Off-policy evaluation functions   #######################################
################################################################################

def off_policy_importance_sampling(
    states_sequence, actions_sequence, rewards_sequence, fence_posts, gamma,
    pi_evaluation, pi_behavior, num_of_states = None, num_of_actions = None ):

    num_of_trials = len( fence_posts )
    individual_trial_estimators = []
    pi_evaluation = turn_policy_to_stochastic_policy( \
        pi_evaluation, num_of_states = num_of_states, num_of_actions = num_of_actions )
    pi_behavior = turn_policy_to_stochastic_policy( \
        pi_behavior, num_of_states = num_of_states, num_of_actions = num_of_actions )
    for trial_i in range( num_of_trials ):
        rho = 1
        discount = 1/gamma
        trial_return = 0
        if trial_i < num_of_trials - 1:
            steps_in_trial = fence_posts[ trial_i+1 ] -  fence_posts[ trial_i ]
        else:
            steps_in_trial = len( states_sequence) - fence_posts[-1]
        for t in range(
                fence_posts[ trial_i], fence_posts[ trial_i] + steps_in_trial ):
            rho *= pi_evaluation[ states_sequence[ t], actions_sequence[ t]] / \
                pi_behavior[ states_sequence[ t], actions_sequence[ t]]
            discount *= gamma
            trial_return += discount * rewards_sequence[ t ]
        individual_trial_estimators += [ trial_return * rho ]
    estimator = np.mean( individual_trial_estimators )
    
    return estimator, individual_trial_estimators

def off_policy_per_decision_importance_sampling(
    states_sequence, actions_sequence, rewards_sequence, fence_posts, gamma,
    pi_evaluation, pi_behavior, num_of_states = None, num_of_actions = None ):

    num_of_trials = len( fence_posts )
    individual_trial_estimators = []
    pi_evaluation = turn_policy_to_stochastic_policy( \
        pi_evaluation, num_of_states = num_of_states, num_of_actions = num_of_actions )
    pi_behavior = turn_policy_to_stochastic_policy( \
        pi_behavior, num_of_states = num_of_states, num_of_actions = num_of_actions )
    for trial_i in range( num_of_trials ):
        current_trial_estimator = 0
        rho = 1
        discount = 1/gamma
        if trial_i < num_of_trials - 1:
            steps_in_trial = fence_posts[ trial_i+1 ] -  fence_posts[ trial_i ]
        else:
            steps_in_trial = len( states_sequence) - fence_posts[-1]
        for t in range(
                fence_posts[ trial_i], fence_posts[ trial_i] + steps_in_trial ):
            rho *= pi_evaluation[ states_sequence[ t], actions_sequence[ t]] / \
                pi_behavior[ states_sequence[ t], actions_sequence[ t]]
            discount *= gamma
            current_trial_estimator += rho * discount * rewards_sequence[ t ]
        individual_trial_estimators += [ current_trial_estimator ]
    estimator = np.mean( individual_trial_estimators )

    return estimator, individual_trial_estimators

##### NOT SURE IT'S RIGHT YET ##############
def off_policy_per_decision_weighted_importance_sampling(
    states_sequence, actions_sequence, rewards_sequence, fence_posts, gamma,
    pi_evaluation, pi_behavior, num_of_states = None, num_of_actions = None ):

    num_of_trials = len( fence_posts )
    individual_trial_estimators = []
    pi_evaluation = turn_policy_to_stochastic_policy( \
        pi_evaluation, num_of_states = num_of_states, num_of_actions = num_of_actions )
    pi_behavior = turn_policy_to_stochastic_policy( \
        pi_behavior, num_of_states = num_of_states, num_of_actions = num_of_actions )
    fence_posts_with_length_appended = fence_posts + [ len( states_sequence ) ]
    single_patient_sequences_length = [ fence_posts_with_length_appended[i+1] - \
        fence_posts_with_length_appended[i] for i in range(len(fence_posts)) ]
    length_of_longest_patient_sequence = max( single_patient_sequences_length )
    rho_array = np.nan * np.zeros( ( num_of_trials, length_of_longest_patient_sequence ) )
#    rho_array = np.ones( ( num_of_trials, length_of_longest_patient_sequence ) )
    for trial_i in range( num_of_trials ):
        rho = 1
        if trial_i < num_of_trials - 1:
            steps_in_trial = fence_posts[ trial_i+1 ] -  fence_posts[ trial_i ]
        else:
            steps_in_trial = len( states_sequence) - fence_posts[-1]
        t_within_trial = 0
        for t in range(
                fence_posts[ trial_i], fence_posts[ trial_i] + steps_in_trial ):
            rho *= pi_evaluation[ states_sequence[ t], actions_sequence[ t]] / \
                pi_behavior[ states_sequence[ t], actions_sequence[ t]]
            rho_array[ trial_i, t_within_trial ] = rho
            t_within_trial += 1
        rho_array[ trial_i, t_within_trial: ] = rho
    weights_normalization = np.sum( rho_array, axis = 0 )
    for trial_i in range( num_of_trials ):
        current_trial_estimator = 0
        rho = 1
        discount = 1/gamma
        if trial_i < num_of_trials - 1:
            steps_in_trial = fence_posts[ trial_i+1 ] -  fence_posts[ trial_i ]
        else:
            steps_in_trial = len( states_sequence) - fence_posts[-1]
        t_within_trial = 0
        for t in range(
                fence_posts[ trial_i], fence_posts[ trial_i] + steps_in_trial ):
            rho *= pi_evaluation[ states_sequence[ t], actions_sequence[ t]] / \
                pi_behavior[ states_sequence[ t], actions_sequence[ t]]
            w = rho / weights_normalization[ t_within_trial ]
            discount *= gamma
            current_trial_estimator += w * discount * rewards_sequence[ t ]
            t_within_trial += 1
        individual_trial_estimators += [ current_trial_estimator ]
    estimator = np.nansum( individual_trial_estimators )
    ### Don't like the nansum

    return estimator, individual_trial_estimators

def off_policy_weighted_importance_sampling(
    states_sequence, actions_sequence, rewards_sequence, fence_posts, gamma,
    pi_evaluation, pi_behavior, num_of_states = None, num_of_actions = None ):

    num_of_trials = len( fence_posts )
    individual_trial_estimators = []
    pi_evaluation = turn_policy_to_stochastic_policy( \
        pi_evaluation, num_of_states = num_of_states, num_of_actions = num_of_actions )
    pi_behavior = turn_policy_to_stochastic_policy( \
        pi_behavior, num_of_states = num_of_states, num_of_actions = num_of_actions )
    rho_array = np.nan * np.zeros( num_of_trials )
    for trial_i in range( num_of_trials ):
        rho = 1
        if trial_i < num_of_trials - 1:
            steps_in_trial = fence_posts[ trial_i+1 ] -  fence_posts[ trial_i ]
        else:
            steps_in_trial = len( states_sequence) - fence_posts[-1]
        for t in range(
                fence_posts[ trial_i], fence_posts[ trial_i] + steps_in_trial ):
            rho *= pi_evaluation[ states_sequence[ t], actions_sequence[ t]] / \
                pi_behavior[ states_sequence[ t], actions_sequence[ t]]
        rho_array[ trial_i ] = rho
    normalization = np.nansum( rho_array )
    for trial_i in range( num_of_trials ):
        rho = 1
        discount = 1/gamma
        trial_return = 0
        if trial_i < num_of_trials - 1:
            steps_in_trial = fence_posts[ trial_i+1 ] -  fence_posts[ trial_i ]
        else:
            steps_in_trial = len( states_sequence) - fence_posts[-1]
        for t in range(
                fence_posts[ trial_i], fence_posts[ trial_i] + steps_in_trial ):
            rho *= pi_evaluation[ states_sequence[ t], actions_sequence[ t]] / \
                pi_behavior[ states_sequence[ t], actions_sequence[ t]]
            discount *= gamma
            trial_return += discount * rewards_sequence[ t ]
        individual_trial_estimators += [ trial_return * rho ]
    estimator = np.sum( individual_trial_estimators ) / normalization

    return estimator, individual_trial_estimators

def off_policy_per_decision_doubly_robust(
    states_sequence, actions_sequence, rewards_sequence, fence_posts, gamma,
    pi_evaluation, pi_behavior, V = None, Q = None, num_of_states = None, num_of_actions = None ):

    num_of_trials = len( fence_posts )
    individual_trial_estimators = []
    pi_evaluation = turn_policy_to_stochastic_policy( \
        pi_evaluation, num_of_states = num_of_states, num_of_actions = num_of_actions )
    pi_behavior = turn_policy_to_stochastic_policy( \
        pi_behavior, num_of_states = num_of_states, num_of_actions = num_of_actions )
    # estimate V and Q if they are not passed as parameters
    if V is None or Q is None:
        # TODO : add part which calculate R and T from data if they are not given
        V, Q = policy_evaluation( T , R , pi_evaluation , gamma )
    # calculate the doubly robust estimator of the policy
    for trial_i in range( num_of_trials ):
        current_trial_estimator = 0
        rho = 1
        discount = 1/gamma
        if trial_i < num_of_trials - 1:
            steps_in_trial = fence_posts[ trial_i+1 ] -  fence_posts[ trial_i ]
        else:
            steps_in_trial = len( states_sequence) - fence_posts[-1]
        for t in range(
                fence_posts[ trial_i], fence_posts[ trial_i] + steps_in_trial ):
            previous_rho = rho
            rho *= pi_evaluation[ states_sequence[ t], actions_sequence[ t]] / \
                pi_behavior[ states_sequence[ t], actions_sequence[ t]]
            discount *= gamma
            current_trial_estimator += rho * discount * rewards_sequence[ t ] - \
                discount * ( rho * Q[ states_sequence[ t ], actions_sequence[ t ] ] - \
                             previous_rho * V[ states_sequence[ t ] ] )
        individual_trial_estimators += [ current_trial_estimator ]
    estimator = np.mean( individual_trial_estimators )

    return estimator, individual_trial_estimators

def off_policy_per_decision_weighted_doubly_robust(
    states_sequence, actions_sequence, rewards_sequence, fence_posts, gamma,
    pi_evaluation, pi_behavior, V = None, Q = None, num_of_states = None, num_of_actions = None ):

    num_of_trials = len( fence_posts )
    individual_trial_estimators = []
    pi_evaluation = turn_policy_to_stochastic_policy( \
        pi_evaluation, num_of_states = num_of_states, num_of_actions = num_of_actions )
    pi_behavior = turn_policy_to_stochastic_policy( \
        pi_behavior, num_of_states = num_of_states, num_of_actions = num_of_actions )
    # estimate V and Q if they are not passed as parameters
    if V is None or Q is None:
        # TODO : add part which calculate R and T from data if they are not given
        V, Q = policy_evaluation( T , R , pi_evaluation , gamma )
    # calculate the doubly robust estimator of the policy
    fence_posts_with_length_appended = fence_posts + [ len( states_sequence ) ]
    single_patient_sequences_length = [ fence_posts_with_length_appended[i+1] - \
        fence_posts_with_length_appended[i] for i in range(len(fence_posts)) ]
    length_of_longest_patient_sequence = max( single_patient_sequences_length )
    rho_array = np.nan * np.zeros( ( num_of_trials, length_of_longest_patient_sequence ) )
#    rho_array = np.ones( ( num_of_trials, length_of_longest_patient_sequence ) )
    for trial_i in range( num_of_trials ):
        rho = 1
        if trial_i < num_of_trials - 1:
            steps_in_trial = fence_posts[ trial_i+1 ] -  fence_posts[ trial_i ]
        else:
            steps_in_trial = len( states_sequence) - fence_posts[-1]
        t_within_trial = 0
        for t in range(
                fence_posts[ trial_i], fence_posts[ trial_i] + steps_in_trial ):
            rho *= pi_evaluation[ states_sequence[ t], actions_sequence[ t]] / \
                pi_behavior[ states_sequence[ t], actions_sequence[ t]]
            rho_array[ trial_i, t_within_trial ] = rho
            t_within_trial += 1
        rho_array[ trial_i, t_within_trial: ] = rho
    weights_normalization = np.sum( rho_array, axis = 0 )
    for trial_i in range( num_of_trials ):
        current_trial_estimator = 0
        rho = 1
        w = 1 / num_of_trials
        discount = 1/gamma
        if trial_i < num_of_trials - 1:
            steps_in_trial = fence_posts[ trial_i+1 ] -  fence_posts[ trial_i ]
        else:
            steps_in_trial = len( states_sequence) - fence_posts[-1]
        t_within_trial = 0
        for t in range(
                fence_posts[ trial_i], fence_posts[ trial_i] + steps_in_trial ):
            previous_w = w
            rho *= pi_evaluation[ states_sequence[ t], actions_sequence[ t]] / \
                pi_behavior[ states_sequence[ t], actions_sequence[ t]]
            w = rho / weights_normalization[ t_within_trial ]
            discount *= gamma
            current_trial_estimator += w * discount * rewards_sequence[ t ] - \
                discount * ( w * Q[ states_sequence[ t ], actions_sequence[ t ] ] - \
                             previous_w * V[ states_sequence[ t ] ] )
            t_within_trial += 1
        individual_trial_estimators += [ current_trial_estimator ]
    estimator = np.sum( individual_trial_estimators )

    return estimator, individual_trial_estimators