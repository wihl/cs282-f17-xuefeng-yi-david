import numpy as np
import tensorflow as tf

np.random.seed(3)
tf.set_random_seed(3)

'''
TODOs: 
1. implement prioritized experience replay
2. drop-out
3. or batch normalization (prevent over-fitting)
4. early_stopping
5. split train, val, and test
'''
class DQN:
    
    def __init__(
        self,
        num_actions,
        num_features,
        hidden_units = 128,
        learning_rate=0.001,
        reward_discount = 0.95,
        epsilon_greedy = 0.9,
        e_increment = None,
        replace_target_iter = 200,
        mem_size = 2000,
        batch_size = 128,
        dueling = False,
        output_graph = False,
        output_path = None,
        sess = None
    ):
        # network layer sizes
        self.num_actions, self.num_features = num_actions, num_features
        self.hidden_units = hidden_units
        # training 
        self.lr = learning_rate
        self.gamma = reward_discount
        self.replace_target_iter = replace_target_iter
        self.memory_size = mem_size
        self.batch_size = batch_size
        # epsilon
        self.e_increment = e_increment
        self.e_max = epsilon_greedy
        self.epsilon = 0 if e_increment else (1 - self.e_max)
        
        self.dueling = dueling
        self.learn_step_counter = 0
        # s, a, r ,s_
        self.memory = np.zeros((self.memory_size, 2 * self.num_features + self.num_actions + 1))
        
        #build net
        self.__build_net__()

        # sync target network
        self.sync_target_op = [tf.assign(t, e) for t, e in zip(tf.get_collection('target_net_params'), \
                                                               tf.get_collection('eval_net_params'))]
        
        if not sess:
            self.sess = tf.Session()
        else:
            self.sess = sess
        
        if output_graph:
            if not output_path: output_path = ''
            else: output_path += '/'
            self.writer = tf.summary.FileWriter("logs/" + output_path, self.sess.graph)
        
        self.sess.run(tf.global_variables_initializer())
        
    def __build_net__(self):
        def build_layers(s, c_names, summary=True):
            # use xavier_initializer with normal distribution
            w_init, b_init = tf.contrib.layers.xavier_initializer(uniform=False), tf.constant_initializer(.1)
            # inpyt layer + relu
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.num_features, self.hidden_units], initializer=w_init, collections=c_names)
                b1 = tf.get_variable('b1', [1, self.hidden_units], initializer=b_init, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)
            
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [self.hidden_units, self.hidden_units], initializer=w_init, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.hidden_units], initializer=b_init, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
            
            if self.dueling:
                # state value
                with tf.variable_scope('Value'):
                    w2 = tf.get_variable('w3', [self.hidden_units, 1], initializer=w_init, collections=c_names)
                    b2 = tf.get_variable('b3', [1, 1], initializer=b_init, collections=c_names)
                    self.V = tf.matmul(l2, w2) + b2
                # action value
                with tf.variable_scope('Advantage'):
                    w2 = tf.get_variable('w3', [self.hidden_units, self.num_actions], initializer=w_init, collections=c_names)
                    b2 = tf.get_variable('b3', [1, self.num_actions], initializer=b_init, collections=c_names)
                    self.A = tf.matmul(l2, w2) + b2
                # output Q value layer
                with tf.variable_scope('Q'):
                    # Q = V(s) + A(s,a)
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))
            else:
                # output Q value layer
                with tf.variable_scope('Q'):
                    w2 = tf.get_variable('w3', [self.hidden_unitshidden_size, self.num_actions], initializer=w_init, collections=c_names)
                    b2 = tf.get_variable('b3', [1, self.num_actions], initializer=b_init, collections=c_names)
                    out = tf.matmul(l2, w2) + b2
            
            if summary:
                tf.summary.histogram('V', self.V)
                tf.summary.histogram('A', self.A)
                tf.summary.histogram('Q', out)
            
            return out
        
        ## ------------------ build evaluate_net ------------------
        # input, i.e state
        self.s = tf.placeholder(tf.float32, [None, self.num_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.num_actions], name='q_target')
        
        with tf.variable_scope('eval_net'):
            self.q_eval = build_layers(self.s, ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES])
        
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        
        ## ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.num_features], name='s_')
        
        with tf.variable_scope('target_net'):
            self.q_next = build_layers(self.s_, ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES], summary=False)
        
        ## ------------------ summary ------------------
        tf.summary.scalar('loss', self.loss)
        self.merged = tf.summary.merge_all()
        
    def store_transition(self, histories):
        self.memory = histories

            #####################
            # if with simulator #
            #####################
            #     def store_transition(self, s, a, r, s_):
            #         if not hasattr(self, 'memory_counter'):
            #             self.memory_counter = 0
            #         transition = np.hstack((s, a, r, s_))
            #         # start to replace when full
            #         index = self.memory_counter % self.memory_size
            #         self.memory[index, :] = transition
            #         self.memory_counter += 1

            #     def choose_action(self, observation):
            #         observation = observation[np.newaxis, :]
            #         if np.random.uniform() > self.epsilon:
            #             actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            #             action = np.argmax(actions_value)
            #         else:
            #             action = np.random.randint(0, self.num_actions)
            #         return action
    
    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0 and self.learn_step_counter != 0:
            self.sess.run(self.sync_target_op)
            print('\ntarget_params_synced\n')
        
        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        
        # next observation
        q_next = self.sess.run(self.q_next, feed_dict={self.s_: batch_memory[:, -self.num_features:]})
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.num_features]})
        
        q_target = q_eval.copy()
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.num_features].astype(int)
        reward = batch_memory[:, self.num_features + 1]
        
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        
        _, cost, summary = self.sess.run([self._train_op, self.loss, self.merged],
                                     feed_dict={self.s: batch_memory[:, :self.num_features],
                                                self.q_target: q_target})
        
        self.writer.add_summary(summary, self.learn_step_counter)
        
        if self.e_increment:
            if self.epsilon < self.e_max:
                self.epsilon += self.e_increment 
            else:
                self.epsilon = self.e_max
            
        self.learn_step_counter += 1
        
        return cost