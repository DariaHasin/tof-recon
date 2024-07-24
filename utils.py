import SimpleITK as sitk
import nibabel as nib
import numpy as np
import imageio
import glob
import os

def list_dir(d):
    """
    Lists all files in a given directory with their full paths.

    Parameters:
    d (str): Path to the directory.

    Returns:
    list: List of file paths in the directory.
    """
    return [os.path.join(d, f) for f in os.listdir(d)]

def listdir_nohidden(path):
    """
    Lists all non-hidden files and directories in the specified directory.

    Parameters:
    path (str): The path to the directory.

    Returns:
    list of str: A list of paths to the non-hidden files and directories in the specified directory.
    """
    return glob.glob(os.path.join(path, '*'))

def dcm2nii(dcm_path, nii_path, slices_th):
    """
    Converts a DICOM series to a NIfTI file if the number of slices is greater than a given threshold.

    Parameters:
    dcm_path (str): Path to the directory containing DICOM files.
    nii_path (str): Path to the output directory for the NIfTI file.
    slices_th (int): Threshold for the number of slices.

    Returns:
    None
    """
    scan_path = os.path.join(nii_path, os.path.basename(dcm_path) + '.nii.gz')
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    size = image.GetSize()
    
    if size[2] > slices_th:
        sitk.WriteImage(image, scan_path)
        print(f'{os.path.basename(scan_path)} converted')
    else:
        print(f'{os.path.basename(scan_path)} less than {slices_th} slices')

def convert_nifti_to_png_images(nifti_file, output_folder):
    """
    Converts a NIfTI file to a series of PNG images.

    Parameters:
    nifti_file (str): Path to the NIfTI file.
    output_folder (str): Path to the output directory for the PNG images.

    Returns:
    None
    """
    os.makedirs(output_folder, exist_ok=True)

    img = nib.load(nifti_file)
    data = img.get_fdata()
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
    data = data.astype(np.uint8)
    for i in range(data.shape[2]):
        im = data[:, :, i]
        print(im.shape)
        imageio.imwrite(os.path.join(output_folder, f'frame_{i:04d}.png'), data[:, :, i])

def histogram_matching(fixed_image, moving_image):
    """
    Performs histogram matching to align the intensity distributions of two images.

    Parameters:
    fixed_image (sitk.Image): The reference image.
    moving_image (sitk.Image): The image to be matched.

    Returns:
    sitk.Image: The histogram-matched image.
    """
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(256)
    matcher.SetNumberOfMatchPoints(10)
    matcher.ThresholdAtMeanIntensityOn()
    return matcher.Execute(moving_image, fixed_image)

def registration_set_up(initial_transform):
    """
    Sets up the image registration method with the specified initial transform and parameters.

    Parameters:
    initial_transform (sitk.Transform): The initial transform for registration.

    Returns:
    sitk.ImageRegistrationMethod: The configured registration method.
    """
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, 
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return registration_method

def register_images(fixed_image_path, moving_image_path):
    """
    Registers a moving image to a fixed image using a multi-step process including histogram matching and registration.

    Parameters:
    fixed_image_path (str): Path to the fixed image.
    moving_image_path (str): Path to the moving image.

    Returns:
    sitk.Image: The registered (resampled) moving image.
    """
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
    
    matched_moving_image = histogram_matching(fixed_image, moving_image)
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, matched_moving_image, sitk.Euler3DTransform(), 
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = registration_set_up(initial_transform)
    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                                  sitk.Cast(matched_moving_image, sitk.sitkFloat32))
    resampled_image = sitk.Resample(matched_moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    return resampled_image

def normalize_image_to_range_0_1(image):
    """
    Normalizes the intensity values of an image to the range [0, 1].

    Parameters:
    image (sitk.Image): The input image.

    Returns:
    sitk.Image: The normalized image.
    """
    image_float = sitk.Cast(image, sitk.sitkFloat32)
    min_value = sitk.GetArrayFromImage(image_float).min()
    max_value = sitk.GetArrayFromImage(image_float).max()
    
    if min_value == max_value:
        return image_float 
    normalized_image = (image_float - min_value) / (max_value - min_value)
    normalized_image = sitk.Cast(normalized_image, image.GetPixelID())
    return normalized_image

def normalize_image_to_range_0_255(vol):
    """
    Normalize a volume's intensity values to the range 0-255 and perform histogram matching between slices.

    Parameters:
    vol (sitk.Image): The input volume image.

    Returns:
    sitk.Image: The normalized and histogram-matched volume image.
    """
    vol_arr = sitk.GetArrayFromImage(vol)
    vol_arr[vol_arr != 0] = (vol_arr[vol_arr != 0] + abs(np.min(vol_arr))) / abs(np.min(vol_arr)) * 255
    vol_bbg = sitk.GetImageFromArray(vol_arr)
    ref_slice = vol_bbg[:, :, round(len(vol_arr) / 2)]
    size = vol_bbg.GetSize()
    vol_img = sitk.Image(size[0], size[1], size[2], sitk.sitkFloat32)

    for z in range(size[2]):
        slice_2d = vol_bbg[:, :, z]
        equalized_slice = histogram_matching(slice_2d, ref_slice)
        vol_img = sitk.Paste(vol_img, equalized_slice, equalized_slice.GetSize(), destinationIndex=[0, 0, z])
    return vol_img

def convert_slices_to_volume(slices, size, output_path, normalize_range_0_1=True):
    """
    Convert a list of 2D slice images to a 3D volume image, normalize it, and save to a file.

    Parameters:
    slices (list of str): List of file paths to the 2D slice images.
    size (tuple of int): Size of the volume in (width, height).
    output_path (str): Path to save the output volume image.
    normalize_range_0_1 (bool): Flag indicating whether to normalize the volume to the range [0, 1]. 
                                If False, the volume will be normalized to the range [0, 255].

    Returns:
    None
    """
    vol_img = sitk.Image(size[0], size[1], len(slices), sitk.sitkFloat32)

    for i, slice_filename in enumerate(slices):
        slice_image = sitk.ReadImage(slice_filename)
        vol_img = sitk.Paste(vol_img, slice_image, slice_image.GetSize(), destinationIndex=[0, 0, i])

    if normalize_range_0_1:
        vol_img = normalize_image_to_range_0_1(vol_img)
    else:
        vol_img = normalize_image_to_range_0_255(vol_img)
    vol_img.SetSpacing(slice_image.GetSpacing())
    vol_img.SetOrigin(slice_image.GetOrigin())
    vol_img.SetDirection(slice_image.GetDirection())
    sitk.WriteImage(vol_img, output_path)