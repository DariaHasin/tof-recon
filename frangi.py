import nibabel as nib
import numpy as np
from skimage.filters import frangi

# Load the NIfTI scan
nifti_path = '/home/daria/shared/data/tof-project/nesvor-output/Daria/nesvor_recon_20240307_1831.nii.gz'
img = nib.load(nifti_path)
data = img.get_fdata()

# Define custom parameters for the frangi filter
scale_range = (1, 10)  # range of scales to use
scale_step = 2  # step size between scales
alpha = 0.5  # sensitivity to plate-like structures
beta = 0.5  # sensitivity to blob-like structures
gamma = 35  # sensitivity to line-like structures
black_ridges = False  # detect white ridges

# Apply the Frangi filter slice by slice
filtered_data = np.zeros_like(data)
for i in range(data.shape[2]):
    filtered_data[:, :, i] = frangi(
        data[:, :, i],
        scale_range=scale_range,
        scale_step=scale_step,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=black_ridges
    )

# Save the filtered image
filtered_img = nib.Nifti1Image(filtered_data, img.affine, img.header)
filtered_nifti_path = '/home/daria/shared/data/tof-project/nesvor-output/Daria/nesvor_recon_20240307_1831_frangi_35.nii.gz'
nib.save(filtered_img, filtered_nifti_path)

print(f"Filtered NIfTI file saved as {filtered_nifti_path}")