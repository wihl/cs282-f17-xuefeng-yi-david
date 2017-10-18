import numpy as np
from sklearn.cluster import KMeans
import pickle as pkl

print ('data preparing ...')
# read the src

datapath = '../../data/'
data = np.genfromtxt(datapath + 'Sepsis_imp.csv', dtype=float, delimiter=',', skip_header=1)
# remove intervention, but include ventilation, sedation, RRT
interventions = np.setdiff1d(np.arange(47, 57), [51,52,54,55,56]) #[52,53,55,57]
data = np.delete(data, interventions, axis=1)
data = np.delete(data, [0,1,2], axis=1)

data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

print ('clustering ...')
kmeans = KMeans(n_clusters=2000, random_state=0).fit(data)
states_list = kmeans.labels_
centers = kmeans.cluster_centers_

pkl.dump(states_list, open(datapath + 'states_list.pkl', 'wb'))
pkl.dump(centers, open(datapath + 'centers.pkl', 'wb'))
