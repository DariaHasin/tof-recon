import SimpleITK as sitk
import numpy as np
from config import *


def find_nonzero_indices_3d(array):
    nonzero = np.nonzero(array)
    indices_axis0 = nonzero[0]
    indices_axis1 = nonzero[1]
    indices_axis2 = nonzero[2]
    return indices_axis0, indices_axis1, indices_axis2


def find_first_last_nonzero_indices(indices):
    first_nonzero_index = indices.min()
    last_nonzero_index = indices.max()

    return first_nonzero_index, last_nonzero_index


def find_first_last_nonzero_indices_3d(array):
    indices_axis0, indices_axis1, indices_axis2 = find_nonzero_indices_3d(array)

    first_nonzero_index_axis0, last_nonzero_index_axis0 = find_first_last_nonzero_indices(indices_axis0)
    first_nonzero_index_axis1, last_nonzero_index_axis1 = find_first_last_nonzero_indices(indices_axis1)
    first_nonzero_index_axis2, last_nonzero_index_axis2 = find_first_last_nonzero_indices(indices_axis2)

    return [first_nonzero_index_axis0, last_nonzero_index_axis0], [first_nonzero_index_axis1, last_nonzero_index_axis1], [first_nonzero_index_axis2, last_nonzero_index_axis2]

def trim_image(orig_image, indices_axis0, indices_axis1, indices_axis2):
  image_trimed = orig_image[indices_axis2[0]:indices_axis2[1], indices_axis1[0]:indices_axis1[1], indices_axis0[0]:indices_axis0[1]]
  image_trimed.SetOrigin(orig_image.GetOrigin())
  image_trimed.SetSpacing(orig_image.GetSpacing())
  image_trimed.SetDirection(orig_image.GetDirection())
  return image_trimed

if __name__ == "__main__":
    for key in PLANE_REPITITION:
        image = sitk.ReadImage(f'{SCANS}/{key}{SCANS_SUFFIX_STRIP}')
        image_arr = sitk.GetArrayFromImage(image)
        indices_axis0, indices_axis1, indices_axis2 = find_first_last_nonzero_indices_3d(image_arr)
        image_trimed = trim_image(image, indices_axis0, indices_axis1, indices_axis2)
        sitk.WriteImage(image_trimed, f'{SCANS}/{key}{SCANS_SUFFIX_CUT_STRIP}')
