import numpy as np
import pandas as pd


class icusim(object):

    def __init__(self, num_patients):
        self.num_patients = num_patients
        self.columns = ['bloc','icustayid','charttime', 'gender', 'age', 'elixhauser',
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
        self.df = pd.DataFrame(columns=self.columns)
        for i in range(num_patients):
            self.add_episode()


    def add_episode(self):
        custid = self.df['icustayid'].max()
        if np.isnan(custid):
            custid = 1
        else:
            custid += 1
        n_blocs = np.random.randint(2,20)
        values = {}
        values['bloc'] =  list(range(n_blocs))
        values['icustayid'] = [custid]*n_blocs
        values['gender'] = [np.random.choice([0,1])] * n_blocs
        values['charttime'] = list(range(4570416000,4570416000 + (n_blocs *14400), 14400))
        values['age'] = [np.random.randint(13,85)] * n_blocs
        newdf = pd.DataFrame(values)
        self.df = pd.concat([self.df,newdf])


    def write_episodes(self, filename):
        self.df.to_csv(filename,index=False)
