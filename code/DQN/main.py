import sys
sys.path.append('../utils')

import PatientRecordProcessor
import DQN as dqn
import numpy as np
import tensorflow as tf


def train(model, histories, num_epoches=10):
    # shuffle histories
    np.random.shuffle(histories)
    # insert memory
    model.store_transition(histories[:model.memory_size,:])
    # learn
    for epoch in range(num_epoches):
        loss = model.learn()
        #if epoch % 10 == 0:
        print ( 'epoch:{}, loss:{}'.format(epoch, loss) )


if __name__ == "__main__":

    ######## build training set ########
    prefix = '../../data/'
    prp = PatientRecordProcessor.PatientRecordProcessor(prefix + 'Sepsis_imp.csv', prefix + 'states_list.pkl')

    histories = prp.build_training_history()

    ######## train ########
    tf.reset_default_graph()

    MEMORY_SIZE = 1280
    ACTION_SPACE = 25
    FEATURES = 51
    NUM_EPOCHES = 100

    ## speicify output_path while tunning parameters
    with tf.variable_scope('dueling'):
        dueling_DQN = dqn.DQN(
            num_actions=ACTION_SPACE, num_features=FEATURES, mem_size=MEMORY_SIZE,
            dueling=True, output_graph=True)

    train(dueling_DQN, histories, num_epoches=NUM_EPOCHES)
