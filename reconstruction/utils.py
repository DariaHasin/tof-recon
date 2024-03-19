from datetime import datetime
from natsort import natsorted
import SimpleITK as sitk
import pandas as pd
import numpy as np
import pytz
import glob
import os
from config import *


def get_input_stacks():
    input_stacks = ''
    for key, value in PLANE_REPITITION.items():
        for n in range(value):
            input_stacks = input_stacks + f'{SCANS}/{key}{SCANS_SUFFIX} '
    return input_stacks


def append_row_to_excel(filename, row_data):
    try:
        df = pd.read_excel(filename)
    except FileNotFoundError:
        df = pd.DataFrame()
    df = df.append(pd.Series(row_data), ignore_index=True)
    df.to_excel(filename, index=False)


def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))


''' Return a vol with values 0-255 and after histogram matching between slices '''
def normlize_vol(vol):
    vol_arr = sitk.GetArrayFromImage(vol)
    vol_arr[vol_arr!=0] = (vol_arr[vol_arr!=0] + abs(np.min(vol_arr)))/abs(np.min(vol_arr)) * 255
    vol_bbg = sitk.GetImageFromArray(vol_arr)
    ref_slice = vol_bbg[:, :, round(len(vol_arr)/2)]
    size = vol_bbg.GetSize()
    vol_img = sitk.Image(size[0], size[1], size[2],sitk.sitkFloat32)

    for z in range(size[2]):
        slice_2d = vol_bbg[:, :, z]
        equalized_slice = sitk.HistogramMatching(slice_2d, ref_slice)
        vol_img = sitk.Paste(vol_img, equalized_slice, equalized_slice.GetSize(), destinationIndex=[0, 0, z])
    return vol_img


def convert_slices_to_volume(slices, size, output_path):
    vol_img = sitk.Image(size[0], size[1], len(slices), sitk.sitkFloat32)

    for i, slice_filename in enumerate(slices):
        slice_image = sitk.ReadImage(slice_filename)
        vol_img = sitk.Paste(vol_img, slice_image, slice_image.GetSize(), destinationIndex=[0, 0, i])

    vol_img = normlize_vol(vol_img)
    vol_img.SetSpacing(slice_image.GetSpacing())
    vol_img.SetOrigin(slice_image.GetOrigin())
    vol_img.SetDirection(slice_image.GetDirection())

    sitk.WriteImage(vol_img, output_path)


def get_output_title(today_date, current_time):
    today_date_str = str(today_date).replace('-', '')
    output_title = f'nesvor_recon_{today_date_str}_{current_time}{NIFTI_SUFFIX}'
    return output_title


def get_today_date_time():
    today_date = datetime.today().date()
    israel_timezone = pytz.timezone('Israel')
    current_time = datetime.now(israel_timezone).strftime('%H%M')
    return today_date, current_time

def add_row_of_current_recon(row_data, today_date, current_time):
    
    append_row_to_excel(RUNS_TABLE, row_data)
    slices = listdir_nohidden(SIM_SLICES)
    slices = natsorted(slices)

    scan_path = f'{SCANS}/{UNCRT_VOL_PLANE}{SCANS_SUFFIX}'
    scan = sitk.ReadImage(scan_path)
    scan_size = scan.GetSize()

    slices_ax = slices[:scan.GetSize()[2]-2]
    convert_slices_to_volume(slices_ax, (scan.GetSize()[0],scan.GetSize()[1]),
                    os.path.join(UNCRT_VOL,
                                f'uncertainty_volume_{today_date}_{current_time}{NIFTI_SUFFIX}'))


