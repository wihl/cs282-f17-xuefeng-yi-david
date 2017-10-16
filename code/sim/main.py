import icu as icu

n_patients = 10

if __name__ == "__main__":
    patients = icu.icusim(n_patients)
    patients.write_episodes("test.csv")
