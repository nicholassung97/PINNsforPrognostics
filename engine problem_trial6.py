import sys
# Import Libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import pandas as pd

# Set Random
np.random.seed(1234)
tf.set_random_seed(1234) # New version of tensorflow uses random.set_seed instead

# Create Neural Network
class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, feature_b_train, cycle_b_train, hi_b_train, feature_f_train, cycle_f_train, layers, lb, ub):
        self.lb = lb
        self.ub = ub
        self.feature_b_train = feature_b_train
        self.cycle_b_train = cycle_b_train
        self.hi_b_train = hi_b_train
        self.feature_f_train = feature_f_train
        self.cycle_f_train = cycle_f_train
        self.layers = layers
        self.max_cycle = int(max(cycle_f_train))

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.feature_b_train_tf = tf.placeholder(tf.float32, shape=[None, self.feature_b_train.shape[1]])
        self.cycle_b_train_tf = tf.placeholder(tf.float32, shape=[None, self.cycle_b_train.shape[1]])
        self.hi_b_train_tf = tf.placeholder(tf.float32, shape=[None, self.hi_b_train.shape[1]])
        self.feature_f_train_tf = tf.placeholder(tf.float32, shape=[None, self.feature_f_train.shape[1]])
        self.cycle_f_train_tf = tf.placeholder(tf.float32, shape=[None, self.cycle_f_train.shape[1]])

        self.hi_pred = self.net_hi(self.feature_b_train_tf, self.cycle_b_train_tf)
        self.cost1_pred = self.net_cost1(self.feature_f_train_tf, self.cycle_f_train_tf)
        self.cost2_pred = self.net_cost2(self.feature_f_train_tf, self.cycle_f_train_tf)
        self.cost3_pred = self.net_cost3(self.feature_f_train_tf, self.cycle_f_train_tf)

        self.loss0 = tf.reduce_mean(tf.square(self.hi_b_train_tf - self.hi_pred))
        self.loss1 = tf.reduce_mean(tf.square(self.cost1_pred))
        self.loss2 = tf.reduce_mean(tf.square(self.cost2_pred))
        self.loss3 = tf.reduce_mean(tf.square(self.cost3_pred))
        self.loss = self.loss0 + self.loss1 + self.loss2 + self.loss3
        # self.loss = tf.reduce_mean(tf.square(self.hi_b_train_tf - self.hi_pred)) + tf.reduce_mean(tf.square(self.cost1_pred)) + \
                    # tf.reduce_mean(tf.square(self.cost2_pred)) + tf.reduce_mean(tf.square(self.cost3_pred))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        init = tf.global_variables_initializer()
        self.sess.run(init)


"""Function to create weights and biases based on layers"""

def initialize_NN(self, layers):          # Creates weights and biases   
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            # creates rows of zeros (len of layers)
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
PhysicsInformedNN.initialize_NN = initialize_NN

"""Function that genrates range of values based on the normal distribution"""

def xavier_init(self, size):            # Generates values with following a normal distribution 
        in_dim = size[0]
        out_dim = size[1] 
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        # Initializer that generates a truncated normal distribution.
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
PhysicsInformedNN.xavier_init = xavier_init

"""Not sure how this works"""

def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        # below for loop to calculate for hidden layers
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))    # tf.tanh(tf.add(tf.matmul(H, W), b))
        # below to calculate for output layer
        W = weights[-1]
        b = biases[-1]
        # Y = tf.add(tf.matmul(H, W), b)
        # this is just a linear combination, no activation function applied
        Y = tf.math.sigmoid(tf.add(tf.matmul(H, W), b))
        return Y 
PhysicsInformedNN.neural_net = neural_net

"""Function that returns x when t is entered"""


def net_hi(self, feature, cycle):
    hi = self.neural_net(tf.concat([feature, cycle], 1), self.weights, self.biases)
    return hi
PhysicsInformedNN.net_hi = net_hi


"""Function that inputs features and cycles to return the cost function"""


def net_cost1(self, feature, cycle):
    total_change = []
    for i in range(1, self.max_cycle):
        feature_cycle1 = feature[cycle_f_train[:,0] == i]
        cycle_cycle1 = cycle[cycle_f_train[:, 0] == i]
        feature_cycle2 = feature[cycle_f_train[:,0] == i+1]
        cycle_cycle2 = cycle[cycle_f_train[:, 0] == i+1]
        hi_cycle1 = self.net_hi(feature_cycle1, cycle_cycle1)
        hi_cycle2 = self.net_hi(feature_cycle2, cycle_cycle2)
        change = tf.math.reduce_mean(hi_cycle2) - tf.math.reduce_mean(hi_cycle1)
        total_change.append(change)
    total_change = tf.stack(total_change)
    # cost1 = tf.keras.activations.hard_sigmoid(total_change)
    cost1 = tf.keras.activations.relu(total_change)
    # cost1 = tf.math.sigmoid(total_change)
    return cost1
PhysicsInformedNN.net_cost1 = net_cost1
# # Use std to reduce the deviation within each cycle
# def net_cost2(self, feature, cycle):
#     total_std = []
#     for i in range(1, self.max_cycle+1):
#         feature_cycle1 = feature[cycle_f_train[:, 0] == i]
#         cycle_cycle1 = cycle[cycle_f_train[:, 0] == i]
#         hi_per_cycle = self.net_hi(feature_cycle1, cycle_cycle1)
#         std = tf.math.reduce_std(hi_per_cycle)
#         total_std.append(std)
#     total_std = tf.stack(total_std)
#     # Set a sharper sigmoid function
#     cost2 = tf.keras.activations.relu(total_std - 0.1)
#     # cost2 = tf.math.divide(1, (1+ tf.keras.activations.exponential(-100*(total_std-0.1))))
#     return cost2
# PhysicsInformedNN.net_cost2 = net_cost2

# Use MAPE to reduce the deviations within each cycle
def net_cost2(self, feature, cycle):
    total_mape = []
    for i in range(1, self.max_cycle+1):
        feature_cycle1 = feature[cycle_f_train[:, 0] == i]
        cycle_cycle1 = cycle[cycle_f_train[:, 0] == i]
        hi_pred = self.net_hi(feature_cycle1, cycle_cycle1)
        hi_true = tf.math.reduce_mean(hi_pred)
        mape = tf.stack(tf.keras.metrics.mean_absolute_percentage_error(hi_true, hi_pred))
        total_mape = tf.concat([total_mape, mape],0)
    cost2 = tf.stack(total_mape)
    return cost2
PhysicsInformedNN.net_cost2 = net_cost2


def net_cost3(self, feature, cycle):
    total_drop = []
    for i in range(1, self.max_cycle):
        feature_cycle1 = feature[cycle_f_train[:, 0] == i]
        cycle_cycle1 = cycle[cycle_f_train[:, 0] == i]
        feature_cycle2 = feature[cycle_f_train[:, 0] == i + 1]
        cycle_cycle2 = cycle[cycle_f_train[:, 0] == i + 1]
        hi_cycle1 = self.net_hi(feature_cycle1, cycle_cycle1)
        hi_cycle2 = self.net_hi(feature_cycle2, cycle_cycle2)
        drop = 0.8 * tf.math.reduce_mean(hi_cycle1) - tf.math.reduce_mean(hi_cycle2)
        total_drop.append(drop)
    total_drop = tf.stack(total_drop)
    cost3 = tf.keras.activations.relu(total_drop)
    # cost3 = tf.math.sigmoid(total_drop)
    return cost3
PhysicsInformedNN.net_cost3 = net_cost3

def callback(self, loss):
        print('Loss:', loss)
PhysicsInformedNN.callback = callback

"""Function that outputs iterations, loss and time given no of iterations"""


def train(self):
    tf_dict = {self.feature_b_train_tf: self.feature_b_train,
               self.cycle_b_train_tf: self.cycle_b_train,
               self.hi_b_train_tf: self.hi_b_train,
               self.feature_f_train_tf: self.feature_f_train,
               self.cycle_f_train_tf: self.cycle_f_train}

    self.optimizer.minimize(self.sess,
                            feed_dict=tf_dict,
                            fetches=[self.loss],
                            loss_callback=self.callback)
PhysicsInformedNN.train = train

"""Functions that produces the predicted values for x and f"""


def predict(self, feature, cycle):
    # self.sess.run(init)

    hi_star = self.sess.run(self.hi_pred, {self.feature_b_train_tf: feature, self.cycle_b_train_tf: cycle})
    cost1_star = self.sess.run(self.cost1_pred, {self.feature_f_train_tf: feature, self.cycle_f_train_tf: cycle})
    cost2_star = self.sess.run(self.cost2_pred, {self.feature_f_train_tf: feature, self.cycle_f_train_tf: cycle})
    cost3_star = self.sess.run(self.cost3_pred, {self.feature_f_train_tf: feature, self.cycle_f_train_tf: cycle})
    return hi_star, cost1_star, cost2_star, cost3_star
PhysicsInformedNN.predict = predict

"""# Input values into PhysicsInformedNN"""

data_mat = scipy.io.loadmat('DataTennessee_FE(Time_Statistics,0.5s,overlap=0).mat')
data_all = data_mat['DATA']
data_all_cycle = data_mat['InfoMat']

data_feature = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11' ]
data_signal = ['Current1', 'Current2', 'Current3', 'Voltage1', 'Voltage2', 'Voltage3', 'Accelerometer1', 'Accelerometer2', 'Microphone', 'Tachometer', 'Temperature']
"""Labelling Plus Converting to Pandas Data Frame"""
data_column = []
for i in data_signal:
  for j in data_feature:
    data_column.append(i + "_" + j)
df = pd.DataFrame(data_all, columns = data_column)
df1 = pd.DataFrame(data_all_cycle, columns = ['motor_id', 'cycle', 'ignored'])
df_total = pd.concat([df, df1], axis=1, join="inner")
"""Engine Data"""
eng_healthy = df_total[(df_total.cycle.isin([1,2,3,4]))]
eng_healthy.pop('ignored')
eng_healthy_mean = []
eng_healthy_var = []
for feature_name in data_column:
  eng_healthy_mean.append(eng_healthy[feature_name].mean())
  eng_healthy_var.append(eng_healthy[feature_name].var())
eng_healthy_mean = np.array(eng_healthy_mean)
eng_healthy_var = np.array(eng_healthy_var)
eng_faulty = df_total[((df_total.motor_id == 1) & (df_total.cycle >=15))| ((df_total.motor_id == 2) & (df_total.cycle >=25))| ((df_total.motor_id == 3) & (df_total.cycle >=23))| ((df_total.motor_id == 4) & (df_total.cycle >=26))| ((df_total.motor_id == 5) & (df_total.cycle >=25))| ((df_total.motor_id == 6) & (df_total.cycle >=24))| ((df_total.motor_id == 7) & (df_total.cycle >=24))| ((df_total.motor_id == 9) & (df_total.cycle >=22))]
eng_faulty.pop('ignored')
eng_faulty_mean = []
eng_faulty_var = []
for feature_name in data_column:
  eng_faulty_mean.append(eng_faulty[feature_name].mean())
  eng_faulty_var.append(eng_faulty[feature_name].var())
eng_faulty_mean = np.array(eng_faulty_mean)
eng_faulty_var = np.array(eng_faulty_var)
fisher_ratio = np.square(eng_healthy_mean - eng_faulty_mean)/(eng_healthy_var + eng_faulty_var)
###### Select the top number of features to choose
select_top_feature = 20
fisher_ratio_index = fisher_ratio.argsort()[(-1*select_top_feature):][::-1]


###### Select cycle to choose
select_engine = 9
select_engine_idx = np.where(data_all_cycle[:,0] == select_engine)[0]

# extracts the top 20 features for all engines
data_feature = data_all[:, fisher_ratio_index]
# extracts the top 20 features for the selected engine
data_eng_top20 = data_feature[select_engine_idx]
# extracts only the cycles for all engines
data_cycle = data_all_cycle[:,1:2]
# extracts only the cycles for the selected engine
data_eng_cycle = data_cycle[select_engine_idx]

"""Boundary Conditions - Includes cycle, features and hi"""

# First cycle
min_cycle = int(min(data_eng_cycle))
select_min_cycle = np.where(data_eng_cycle[:,0] == min_cycle)[0]
# Last cycle
max_cycle = int(max(data_eng_cycle))
select_max_cycle = np.where(data_eng_cycle[:,0] == max_cycle)[0]

# Features for the first cycle
feature_first = data_eng_top20[select_min_cycle]
# Features for the last cycle
feature_last = data_eng_top20[select_max_cycle]
# Features at the boundary conditions
feature_b_train = np.vstack((feature_first, feature_last))

# Cycle no for the first cycle
cycle_first = data_eng_cycle[select_min_cycle]
# Cycle no for the last cycle
cycle_last = data_eng_cycle[select_max_cycle]
# Cycle no at the boundary conditions
cycle_b_train = np.vstack((cycle_first, cycle_last))

# Ground state of HI for the boundary conditions
hi_first = np.ones((len(select_min_cycle), 1))*0.95
hi_last = np.ones((len(select_max_cycle), 1))*0.1
# HI values at the boundary conditions
hi_b_train = np.vstack((hi_first, hi_last))

# Boundaries of the maximum and minimum features
feature_lb = data_eng_top20.min(0)
feature_ub = data_eng_top20.max(0)
# Boundaries of the maximum and minimum cycles
cycle_lb = data_eng_cycle.min(0)
cycle_ub = data_eng_cycle.max(0)

lb = np.concatenate((feature_lb, cycle_lb), axis=0)
ub = np.concatenate((feature_ub, cycle_ub), axis=0)

# Represent collocation points with every cycle
feature_f_train = data_eng_top20
cycle_f_train = data_eng_cycle
# N_f = 10000
# feature_f_train = feature_lb + (feature_ub-feature_lb)*lhs(20, N_f)
# array_1_to_18 = np.arange(start = cycle_lb, stop = cycle_ub+1)
# cycle_f_train = np.random.choice(array_1_to_18, (N_f,1), replace=True)

layers = [(int(len(fisher_ratio_index))+1), 50, 1]


model = PhysicsInformedNN(feature_b_train, cycle_b_train, hi_b_train, feature_f_train, cycle_f_train, layers, lb, ub)
start_time = time.time()
model.train()
elapsed = time.time() - start_time
print('Training time: %.4f' % (elapsed))

hi_pred, cost1_pred, cost2_pred, cost3_pred = model.predict(feature_f_train, cycle_f_train)

# Creating Mean values
hi_pred_mean = []
cycle_plot = []
for i in range(min_cycle, max_cycle+1):
    hi_mean_row = np.mean(hi_pred[np.where(cycle_f_train[:,0] == i)[0]])
    cycle_row = i
    hi_pred_mean.append(hi_mean_row)
    cycle_plot.append(cycle_row)

# Plot the curve
plt.title('Engine '+ str(select_engine) + ' - Health Index against No of Cycles')
plt.plot(cycle_f_train, hi_pred, 'x', label='Predicted Value')
plt.plot(cycle_plot, hi_pred_mean, '-r', label='Mean Predicted Value')
plt.locator_params(axis='x', integer=True, tight=True)
plt.legend()
plt.xlabel('cycle')
plt.ylabel('health index value')
plt.show()
