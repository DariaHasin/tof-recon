import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def get_indices(num_slices, num_display=10):
    indices = np.linspace(0, num_slices - 1, num_display, dtype=int)
    return indices

def get_slice_planes(img, slice_indices, color):
    affine = img.affine
    data_shape = img.shape
    planes = []

    for i in slice_indices:
        if i >= data_shape[2]:
            continue

        # Define voxel coordinates for the corners of the plane
        x_coords = [0, data_shape[0], data_shape[0], 0]
        y_coords = [0, 0, data_shape[1], data_shape[1]]
        z_coords = [i, i, i, i]

        # Create vertices for the plane
        vertices = np.array([x_coords, y_coords, z_coords]).T
        vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))  # Add homogeneous coordinate
        vertices = np.dot(vertices, affine.T)[:, :3]  # Apply affine transformation

        # Create polygon for the plane
        verts = [vertices]
        plane = Poly3DCollection(verts, alpha=0.5, facecolors=color, linewidths=1, edgecolors='k')
        planes.append(plane)
    return planes

def plot_slice_planes(axial_planes, coronal_planes, sagittal_planes):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for plane in axial_planes:
        ax.add_collection3d(plane)
    for plane in coronal_planes:
        ax.add_collection3d(plane)
    for plane in sagittal_planes:
        ax.add_collection3d(plane)

    labels = {'axial': 'Axial Plane', 'coronal': 'Coronal Plane', 'sagittal': 'Sagittal Plane'}
    ax.legend([axial_planes[0], coronal_planes[0], sagittal_planes[0]], 
              [labels['axial'], labels['coronal'], labels['sagittal']], 
              loc='upper right', fontsize=10)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()


if __name__ == "__main__":
    subject = 'Daria'
    img_axial = nib.load(f'~/shared/data/tof-project/scans_stripped/{subject}/ax_stripped.nii.gz')
    img_coronal = nib.load(f'~/shared/data/tof-project/scans_stripped/{subject}/cor_stripped.nii.gz')
    img_sagittal = nib.load(f'~/shared/data/tof-project/scans_stripped/{subject}/sag_stripped.nii.gz')

    axial_slices = get_indices(img_axial.shape[2])
    coronal_slices = get_indices(img_coronal.shape[2])
    sagittal_slices = get_indices(img_sagittal.shape[2])

    colors = {'axial': 'red', 'coronal': 'green', 'sagittal': 'blue'}

    axial_planes = get_slice_planes(img_axial, axial_slices, color=colors['axial'])
    coronal_planes = get_slice_planes(img_coronal, coronal_slices, color=colors['coronal'])
    sagittal_planes = get_slice_planes(img_sagittal, sagittal_slices, color=colors['sagittal'])

    plot_slice_planes(axial_planes, coronal_planes, sagittal_planes)
