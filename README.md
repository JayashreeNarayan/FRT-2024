## Details:
1. The Data folder has all the raw data -- subfolders of Data_SimEnd, Data_1tff and the main FLASH file
2. the cfp.sh file navigates to the right directory and then runs a file `start.py` which plots moment maps from .npy files. You need to send in file name as `-i "<file name>"`, possible file names: FMM_0.0_0.0.npy; FMM_90.0_0.0.npy; etc.
