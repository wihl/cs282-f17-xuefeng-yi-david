import numpy as np
import discretize_sepsis_actions as discretizer
import pickle as pkl

class PatientRecordProcessor:
    
    def __init__(self, raw_path, cluster_path):
        
        print ( 'loading dataset ...' )
        self.data = np.genfromtxt(raw_path, dtype=float, delimiter=',', skip_header=1)
        print ( 'loading clustered states ...' )
        self.clusters = pkl.load(open(cluster_path, 'rb'), encoding='latin1')
        print ( 'discretizing actions ...' )
        self.discretize_actions()
        self.patient_map = None
        print ( 'initialization succeed' )
        
    def discretize_actions(self):
        
        self.action_sequence, self.vaso_bins, self.iv_bins = \
        discretizer.discretize_actions(self.data[:,50], self.data[:,47])
    
    def build_patient_map(self):
        
        self.patient_map = {}
        interventions = np.setdiff1d(np.arange(47, 57), [51,52,54,55,56])
        turcated_data = np.delete(self.data, interventions, axis=1)
        turcated_data = np.delete(turcated_data, [0,1,2], axis=1)
        for i, row in enumerate(self.data):
            icuid = str(row[1])
            state_action_outcome = [self.clusters[i], self.action_sequence[i], row[50], row[47], row[-2]]
            if icuid not in self.patient_map:
                # state_id, action, outcome
                self.patient_map[icuid] = {
                    'age':row[4], 'gender':row[3], 'sa':[state_action_outcome],
                    'obser':[turcated_data[i,:]]}
            else:
                self.patient_map[icuid]['sa'].append(state_action_outcome)
                self.patient_map[icuid]['obser'].append(turcated_data[i,:])
        
        return self.patient_map
    
    def build_training_history(self):
        memory = []
        if not self.patient_map:
            print ( 'building patient map ...' )
            self.patient_map = self.build_patient_map()
        
        for _, patient in self.patient_map.items():
            
            if len(patient['sa']) <= 5:
                continue

            for i, patient_icu_stay in enumerate(patient['sa']):
                _, action, _, _, outcome = patient_icu_stay
                s_obser = patient['obser'][i]
                next_s_obser = patient['obser'][i + 1]
                
                reward = 0
                if (i + 1) == len(patient['sa']) - 1:
                    # last stay, check the outcome
                    if patient['sa'][i + 1][-1] == 0: 
                        # survived
                        reward = 15
                    else:
                        reward = -15
                
                memory.append(np.hstack((s_obser, action, reward, next_s_obser)))
                
                if reward != 0:
                    break
        return np.array(memory)