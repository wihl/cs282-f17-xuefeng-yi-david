import icu as icu

n_patients = 10

if __name__ == "__main__":
    icusim = icu.ICUsim(n_patients)
    icusim.load_config('config.json')
    patients = icusim.patients
    icusim.write_episodes('test.csv')
