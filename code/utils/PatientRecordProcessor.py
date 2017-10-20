import numpy as np
import discretize_sepsis_actions as discretizer
import pickle as pkl
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


import sys
sys.path.append('../sim')

import icu as icu


class PatientRecordProcessor:

    columns = ['bloc','icustayid','charttime', 'gender', 'age', 'elixhauser',
                're_admission', 'SOFA', 'SIRS', 'Weight_kg', 'GCS', 'HR',
                'SysBP', 'MeanBP', 'DiaBP', 'Shock_Index', 'RR', 'SpO2',
                'Temp_C', 'FiO2_1', 'Potassium', 'Sodium', 'Chloride',
                'Glucose', 'BUN', 'Creatinine', 'Magnesium', 'Calcium',
                'Ionised_Ca', 'CO2_mEqL', 'SGOT', 'SGPT', 'Total_bili',
                'Albumin', 'Hb', 'WBC_count', 'Platelets_count', 'PTT',
                'PT', 'INR', 'Arterial_pH', 'paO2', 'paCO2',
                'Arterial_BE', 'Arterial_lactate', 'HCO3', 'PaO2_FiO2',
                'median_dose_vaso', 'max_dose_vaso', 'input_total_tev',
                'input_4hourly_tev', 'output_total', 'output_4hourly',
                'cumulated_balance_tev', 'sedation', 'mechvent', 'rrt',
                'died_in_hosp', 'mortality_90d']

    observ_cols = ['elixhauser','re_admission', 'SOFA', 'SIRS', 'Weight_kg', 'GCS', 'HR',
                'SysBP', 'MeanBP', 'DiaBP', 'RR', 'SpO2',
                'Temp_C', 'FiO2_1', 'Potassium', 'Sodium', 'Chloride',
                'Glucose', 'BUN', 'Creatinine', 'Magnesium', 'Calcium',
                'Ionised_Ca', 'CO2_mEqL', 'SGOT', 'SGPT', 'Total_bili',
                'Albumin', 'Hb', 'WBC_count', 'Platelets_count', 'PTT',
                'PT', 'INR', 'Arterial_pH', 'paO2', 'paCO2',
                'Arterial_BE', 'Arterial_lactate', 'HCO3', 'PaO2_FiO2',
                'output_total', 'output_4hourly',
                'sedation', 'mechvent', 'rrt']

    n_clusters = 2000

    def __init__(self):
        self.patient_map = None


    def load_sim(self,n_patients,config_path):
        print ('generating episodes...')
        icusim = icu.ICUsim(n_patients)
        icusim.load_config(config_path)
        self.df = icusim.patients
        observations = self.df[self.observ_cols]
        print ('generating clusters...')
        mbk = MiniBatchKMeans(init='k-means++', n_clusters=self.n_clusters,
                                batch_size=100,n_init=10, max_no_improvement=10,
                                verbose=0, init_size=3*self.n_clusters,random_state=0)
        mbk.fit(observations)
        self.clusters = mbk.cluster_centers_
        print ( 'discretizing actions ...' )
        self.discretize_actions()
        print ( 'initialization succeeded' )


    def load_csv(self, raw_path, cluster_path):
        print ( 'loading dataset ...' )
        self.df = pd.read_csv(raw_path)
        print ( 'loading clustered states ...' )
        self.clusters = pkl.load(open(cluster_path, 'rb'), encoding='latin1')
        print ( 'discretizing actions ...' )
        self.discretize_actions()
        print ( 'initialization succeeded' )

    def discretize_actions(self):

        self.action_sequence, self.vaso_bins, self.iv_bins = \
        discretizer.discretize_actions(self.df.loc[:,'input_4hourly_tev'],
                                       self.df.loc[:,'median_dose_vaso'])

    def build_patient_map(self):

        self.patient_map = {}
        for i, row in self.df.iterrows():
            icuid = str(row['icustayid'])
            state_action_outcome = [self.clusters[i], self.action_sequence[i],
                                    row['input_4hourly_tev'],
                                    row['median_dose_vaso'],
                                    row['died_in_hosp']]
            if icuid not in self.patient_map:
                # state_id, action, outcome
                self.patient_map[icuid] = {
                    'age':row['age'], 'gender':row['gender'], 'sa':[state_action_outcome],
                    'obser':[row[self.observ_cols].as_matrix()]}
            else:
                self.patient_map[icuid]['sa'].append(state_action_outcome)
                self.patient_map[icuid]['obser'].append(row[self.observ_cols].as_matrix())

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
