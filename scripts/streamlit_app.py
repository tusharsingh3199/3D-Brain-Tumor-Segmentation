import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import requests
import streamlit as st

API_URL = "http://localhost:8000/segment"
MODALITIES = ["t1", "t1ce", "t2", "flair"]

st.set_page_config(layout="wide")
st.title(" Brain Tumor Segmentation")
st.write("Upload the 4 MRI modalities (.nii files), run segmentation, and scroll through slices.")

uploads = {}
cols = st.columns(4)
for col, name in zip(cols, MODALITIES):
    uploads[name] = col.file_uploader(name.upper(), type=["nii", "nii.gz"])

if st.button("Run Segmentation"):
    if not all(uploads.values()):
        st.error("Please upload all 4 scans.")
    else:
        with st.spinner("Running model inference... this can take a while"):
            files = {name: (f.name, f.getvalue()) for name, f in uploads.items()}
            response = requests.post(API_URL, files=files)

        if response.status_code != 200:
            st.error(f"Server error: {response.text}")
        else:
            st.session_state["seg_bytes"] = response.content
            st.session_state["scan_bytes"] = {name: f.getvalue() for name, f in uploads.items()}
            st.success("Segmentation complete!")

if "seg_bytes" in st.session_state:
    # write temp files so nibabel can load them
    scans = {}
    for name, data in st.session_state["scan_bytes"].items():
        path = f"_tmp_{name}.nii"
        with open(path, "wb") as f:
            f.write(data)
        v = nib.load(path).get_fdata().astype(np.float32)
        scans[name] = (v - v.min()) / (v.max() - v.min() + 1e-8)

    with open("_tmp_seg.nii.gz", "wb") as f:
        f.write(st.session_state["seg_bytes"])
    seg = nib.load("_tmp_seg.nii.gz").get_fdata()

    depth = scans["t1"].shape[2]
    z = st.slider("Slice", 0, depth - 1, depth // 2)

    fig, ax = plt.subplots(2, 4, figsize=(14, 7))
    for i, name in enumerate(MODALITIES):
        ax[0, i].imshow(scans[name][:, :, z], cmap="gray")
        ax[0, i].set_title(name.upper())
        ax[0, i].axis("off")

        ax[1, i].imshow(scans[name][:, :, z], cmap="gray")
        ax[1, i].imshow(np.ma.masked_where(seg[:, :, z] == 0, seg[:, :, z]),
                         cmap="jet", alpha=0.7, vmin=1, vmax=4)
        ax[1, i].axis("off")

    ax[0, 0].set_ylabel("Scan", fontsize=12)
    ax[1, 0].set_ylabel("+ Segmentation", fontsize=12)
    st.pyplot(fig)

    st.download_button(
        "⬇Download seg.nii.gz",
        data=st.session_state["seg_bytes"],
        file_name="seg.nii.gz",
        mime="application/gzip",
    )