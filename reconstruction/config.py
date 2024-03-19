import os

SUBJECT = 'Michal'
PLANE_REPITITION = {'ax': 1, 'cor':1, 'sag': 1}
UNCRT_VOL_PLANE = 'ax'

NIFTI_SUFFIX = '.nii.gz'
SCANS_SUFFIX_STRIP = f'_stripped{NIFTI_SUFFIX}'
SCANS_SUFFIX_CUT_STRIP = f'_cutted_stripped{NIFTI_SUFFIX}'
SCANS_SUFFIX = SCANS_SUFFIX_STRIP # choose which scan to run - strip or cut-trip

MAIN_FOLDER = '/content/drive/MyDrive/TAU/tof-project/'
RUNS_TABLE = os.path.join(MAIN_FOLDER, 'nesvor_runs.xlsx')
SCANS = os.path.join(MAIN_FOLDER, 'scans_stripped', SUBJECT)
SIM_SLICES = os.path.join(MAIN_FOLDER, 'simulated_slices', SUBJECT)
UNCRT_VOL = os.path.join(MAIN_FOLDER, 'uncertainty_volume', SUBJECT)
RECON_OUTPUT = os.path.join(MAIN_FOLDER, 'nesvor-output', SUBJECT)
# MODEL = os.path.join(RECON_OUTPUT, 'model.pth')
# OUTPUT_SLICES = os.path.join(MAIN_FOLDER, 'output_slices', SUBJECT)

if not os.path.exists(RECON_OUTPUT):
  os.makedirs(RECON_OUTPUT)
if not os.path.exists(SIM_SLICES):
  os.makedirs(SIM_SLICES)
if not os.path.exists(UNCRT_VOL):
  os.makedirs(UNCRT_VOL)
# if not os.path.exists(OUTPUT_SLICES):
#   os.makedirs(OUTPUT_SLICES)


