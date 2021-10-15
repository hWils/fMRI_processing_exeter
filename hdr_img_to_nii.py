import nibabel as nb
from zipfile import ZipFile 
  
# importing required modules 
import zipfile
import os 
import glob

main_filepath = 'C://Users//hlw69//Documents//fMRI_Exeter//Imagery_Raw_and_processed_data//Imagery_Raw_and_processed_data//'

 # only for files that have wr, but not swr
for filename in os.listdir(main_filepath):
  pptnum = int(''.join(filter(str.isdigit, filename)))
  if pptnum != 39:
    print(pptnum)
    subject = main_filepath + filename + '//'
    for condition in os.listdir(subject):
        print(condition)
        if condition == 'Places': # condition == 'Faces' or 
      #    print(condition)
          fmriFiles = subject + condition + "//Segmented//Motion_corrected//"
          for f in os.listdir(fmriFiles):
            #  print(f)
            if f.startswith('wr') and '.img' in f:
               # print(f)
              img = fmriFiles + f
              img_format = nb.load(img) 
              nb.save(img_format, img.replace('.img', '.nii'))


          



           #     if fmriFile.startswith('wr'):
            #        ('True :', img)
           # for fmri in os.listdir(fmriFiles):
             #   if fmri.startswith('wr'):
              #      print('True :', fmri)
         #   file_names = [fn for fn in os.listdir(fmriFiles)
          #    if (fn.startswith('wr'))]
         #   print(file_names)
         #   for img in glob.glob("C://Users//hlw69//Documents//fMRI_Exeter//experimentData//mask//*.img"):


"""
import glob
for img in glob.glob("C://Users//hlw69//Documents//fMRI_Exeter//experimentData//mask//*.img"):
    if
    img_format = nb.load(img) 
    nb.save(img_format, img.replace('.img', '.nii'))
"""
"""

# create one nii.gz file


import gzip
import shutil
with open('C://Users//hlw69//Documents//fMRI_Exeter//experimentData//wrImagine_30_7_1x001.nii', 'rb') as f_in:
    with gzip.open('C://Users//hlw69//Documents//fMRI_Exeter//experimentData//wrImagine_30_7_1x001.nii.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)




#fname = '<file name>.img'  
#img = nb.load(fname)
#nb.save(img, fname.replace('.img', '.nii'))"""


#for f in zf.infolist():
 #   print f.filename
# C:\Users\hlw69\Documents\fMRI_Exeter\experimentData
"""
fname = '<file name>.img'  
img = nb.load(fname)
nb.save(img, fname.replace('.img', '.nii'))"""