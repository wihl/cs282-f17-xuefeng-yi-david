import numpy as np
import pandas as pd


class Generator(object):

    def gen(self,n,var):
        if var['type'] == 'integer':
            return self.gen_int(n,var)
        elif var['type'] == 'sequence':
            return self.gen_seq(n,var)
        elif var['type'] == 'categorical':
            return self.gen_categorical(n,var)
        elif var['type'] == 'float':
            return self.gen_float(n,var)
        else:
            raise(ValueError("unknown type to generate: '%s'" % var['type']))

    def gen_int(self,n,var):
        i_min = var['min']
        i_max = var['max']
        return np.random.randint(i_min,i_max,n)

    def gen_categorical(self,n,var):
        return np.random.choice(var['values'],n)

    def gen_seq(self,n,var):
        if 'end' in var:
            end = int(var['end'])
        else:
            end = var['start'] + n * var['step']
        return list(range(var['start'],end, var['step']))

    def gen_float(self,n,var):
        f_min = float(var['min'])
        f_max = float(var['max'])
        return np.random.uniform(f_min,f_max,n)
