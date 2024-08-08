import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

def extract_data(img):
    affine = img.affine
    data = img.get_fdata()
    
    coords = np.indices(data.shape).reshape(3, -1).T
    real_coords = np.dot(coords, affine[:3, :3].T) + affine[:3, 3]
    intensities = data.ravel()
    return real_coords, intensities


def get_normal(img):  
    plane_dict = {'axial': 2, 'coronal': 1, 'sagittal': 0}
    affine = img.affine
    norm_affine = affine.copy()
    norm_affine[:3, :3] = np.linalg.qr(affine[:3, :3])[0]  # Orthogonalize
    z_axis = np.argmax(np.abs(norm_affine[:3, 2]))

    if z_axis == plane_dict['axial']:
        return [0, 0, 1]
    elif z_axis == plane_dict['coronal']:
        return [0, 1, 0]
    elif z_axis == plane_dict['sagittal']:
        return [1, 0, 0]
    else:
        raise ValueError("Unrecognized orientation or affine matrix.")
    

class NIfTIDataset(Dataset):
    def __init__(self, img_list):
        self.coords = []
        self.intensities = []
        self.normals = []
        
        for img in img_list:
            real_coords, intensities = extract_data(img)
            normal = get_normal(img)
            
            self.coords.append(real_coords)
            self.intensities.append(intensities)
            self.normals.append([normal] * len(intensities))  # Normal vector is the same for all intensities
        
        self.coords = np.concatenate(self.coords, axis=0)
        self.intensities = np.concatenate(self.intensities, axis=0)
        self.normals = np.concatenate(self.normals, axis=0)
        
    def __len__(self):
        return len(self.intensities)
    
    def __getitem__(self, idx):
        coord = self.coords[idx]
        intensity = self.intensities[idx]
        normal = self.normals[idx]
        
        sample = {
            'coords': torch.tensor(coord, dtype=torch.float32),
            'normal': torch.tensor(normal, dtype=torch.float32),
            'intensity': torch.tensor(intensity, dtype=torch.float32)
        }
        return sample
    

if __name__ == "__main__":
    subject = 'Daria'
    img_axial = nib.load(f'~/shared/data/tof-project/scans_stripped/{subject}/ax_stripped.nii.gz')
    img_coronal = nib.load(f'~/shared/data/tof-project/scans_stripped/{subject}/cor_stripped.nii.gz')
    img_sagittal = nib.load(f'~/shared/data/tof-project/scans_stripped/{subject}/sag_stripped.nii.gz')
    
    img_list = [img_axial, img_coronal, img_sagittal]

    dataset = NIfTIDataset(img_list)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    for batch in dataloader:
        coords = batch['coords']
        normal = batch['normal']
        intensity = batch['intensity']
        
        print(coords, normal, intensity)
        break