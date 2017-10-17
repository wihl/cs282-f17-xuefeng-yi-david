import numpy as np
import pandas as pd
import generator as g
import json


class ICUsim(object):

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

    @property
    def patients(self):
        return self.df

    def load_config(self,filename):
        values = {}
        custid = self.df['icustayid'].max()
        if np.isnan(custid):
            custid = 1
        else:
            custid += 1

        with open(filename) as config_file:
            self.config = json.load(config_file)
            gen = g.Generator()
            for i in range(self.num_patients):
                #    self.add_episode()
                n_bloc = np.random.randint(2,20)
                values['icustayid'] = [custid] * n_bloc
                values['bloc'] =  list(range(n_bloc))
                for var in self.config['Sepsis-ICU']['variables']:
                    values[var] = gen.gen(n_bloc,self.config['Sepsis-ICU']['variables'][var])
                newdf = pd.DataFrame(values)
                self.df = pd.concat([self.df,newdf])

    def write_episodes(self, filename):
        self.df.to_csv(filename,index=False)
