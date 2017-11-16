# general imports 
import matplotlib.pyplot as plt
import matplotlib 
import numpy as np

# created by us 
import gridworld 
    

def gridworld_true_MDP( task_name, action_error_prob, pit_reward, gamma ):
    # note that for correct calculating of T the task is given action_error_prob=0
    task = gridworld.GridWorld( task_name ,
                                action_error_prob=0, 
                                rewards={'*': 50, 'moved': -1, 'hit-wall': -1,'X':pit_reward} ,
                                terminal_markers='*' )        
    state_count = task.num_states  
    action_count = task.num_actions
    
    T = np.zeros( ( state_count, action_count, state_count ) )
    R = np.zeros( ( state_count, action_count, state_count ) )
    
    for state_idx in range(state_count):
        unflat_idx = task.maze.unflatten_index(state_idx)
        for action_idx in range(action_count):
            if task.maze.get_unflat( unflat_idx ) == '#':
                T[ state_idx, action_idx, state_idx ] = 1
            elif task.maze.get_unflat( unflat_idx ) == '*':
                T[ state_idx, action_idx, state_idx ] = 1
            else:
                task.state = task.maze.flatten_index(unflat_idx)
                new_state, reward = task.perform_action( action_idx )
                T[ state_idx, action_idx, new_state ] += 1 - action_error_prob
                R[ state_idx, action_idx, new_state ] = reward
                if action_idx == 0 or action_idx == 1:
                    task.state = task.maze.flatten_index(unflat_idx)
                    new_state, reward = task.perform_action( 2 )
                    T[ state_idx, action_idx, new_state ] += action_error_prob / 2
                    R[ state_idx, action_idx, new_state ] = reward
                    task.state = task.maze.flatten_index(unflat_idx)
                    new_state, reward = task.perform_action( 3 )
                    T[ state_idx, action_idx, new_state ] += action_error_prob / 2
                    R[ state_idx, action_idx, new_state ] = reward
                else:
                    task.state = task.maze.flatten_index(unflat_idx)
                    new_state, reward = task.perform_action( 0 )
                    T[ state_idx, action_idx, new_state ] += action_error_prob / 2
                    R[ state_idx, action_idx, new_state ] = reward
                    task.state = task.maze.flatten_index(unflat_idx)
                    new_state, reward = task.perform_action( 1 )
                    T[ state_idx, action_idx, new_state ] += action_error_prob / 2
                    R[ state_idx, action_idx, new_state ] = reward
                    
    MDP = {
        'T' : T,
        'R' : R,
        'gamma' : gamma,
        'state_count' : state_count,
        'action_count' : action_count}
    
    return MDP
            