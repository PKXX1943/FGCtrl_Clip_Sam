import os
import glob

data_dir = 'data/brain_mri_kaggle3m'  
caption_file = os.path.join(data_dir, 'caption.txt')

with open(caption_file, 'r') as f:
    caption = f.readline().strip()  
    
image_files = []
mask_files = []

paths = glob.glob(os.path.join(data_dir, '**', '*.tif'), recursive=True)
    
for file in sorted(paths):
    if '_mask' in file:
        mask_files.append(file)
    else:
        image_files.append(file)
            
annotation_file = os.path.join(data_dir, 'annotation.txt')
with open(annotation_file, 'w') as f:
    for img_file, mask_file in zip(sorted(image_files), sorted(mask_files)):
        img_path = os.path.join(data_dir, img_file)
        mask_path = os.path.join(data_dir, mask_file)
        f.write(f"{img_path} | {mask_path} | {caption}\n")

print("annotation.txt done。")
