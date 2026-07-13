def patients_dir(Data_Path):
    patient_dir = []
    for grade in ["HGG", "LGG"]:
        grade_dir = os.path.join(Data_Path, grade)
        patients = sorted(os.listdir(grade_dir))
        patient_paths = [os.path.join(grade_dir, p) for p in patients]

        for path in patient_paths:
            patient_dir.append(path)

    return patient_dir
