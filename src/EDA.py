def EDA(patients_dir):
    patient_dir = random.choice(patients_dir)
    img = []
    seg = []
    for modality in ["t1", "t1ce", "t2", "flair"]:
        v = nib.load(os.path.join(patient_dir, f"{os.path.basename(patient_dir)}_{modality}.nii")).get_fdata().astype(np.float32)
        v = (v - v.min()) / (v.max() - v.min() + 1e-8)
        img.append(v)

    seg = nib.load(os.path.join(patient_dir, f"{os.path.basename(patient_dir)}_seg.nii")).get_fdata().astype(np.int32)

    def view_slice(z):
        plt.figure(figsize=(12, 6))
        for i in range(4):
            plt.subplot(2, 4, i+1)
            plt.imshow(img[i][:, :, z], cmap="gray")
            plt.axis("off")

            plt.subplot(2, 4, 5+i)
            plt.imshow(img[i][:, :, z], cmap="gray")
            plt.axis("off")

            slice_seg = seg[:, :, z]
            masked_seg = np.ma.masked_where(slice_seg == 0, slice_seg)
            plt.imshow(masked_seg, cmap="jet", alpha=0.7, vmin=0, vmax=4)

        plt.show()

    interact(view_slice, z=IntSlider(min=0, max=154, step=1, value=75))