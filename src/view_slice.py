   def view_slice(z):
        plt.figure(figsize=(8, 6))
        titles = ["T1","T1CE","T2","FLAIR"]

        for i in range(4):

            plt.subplot(3,4,i+1)
            plt.imshow(image[:,:,z,i], cmap="gray")
            plt.title(titles[i])
            plt.axis("off")

            plt.subplot(3,4,5+i)
            plt.imshow(image[:,:,z,i], cmap="gray")
            plt.imshow(np.ma.masked_where(mask[:,:,z]==0, mask[:,:,z]),
                       cmap="jet", alpha=0.6, vmin=0, vmax=3)
            plt.title("Ground Truth")
            plt.axis("off")

            plt.subplot(3,4,9+i)
            plt.imshow(image[:,:,z,i], cmap="gray")
            plt.imshow(np.ma.masked_where(prediction[:,:,z]==0, prediction[:,:,z]),
                       cmap="jet", alpha=0.6, vmin=0, vmax=3)
            plt.title("Prediction")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    interact(view_slice, z=IntSlider(min=0, max=154, value=64))

MRI_Results(patients_dir, model)