import os

data_dir = 'data/kvasir_seg/sessile-main'  
image_dir = os.path.join(data_dir, 'image')
mask_dir = os.path.join(data_dir, 'mask')
caption_file = os.path.join(data_dir, 'caption.txt')

with open(caption_file, 'r') as f:
    caption = f.readline().strip()  

image_files = sorted(os.listdir(image_dir))
mask_files = sorted(os.listdir(mask_dir))

assert len(image_files) == len(mask_files), "Image and mask counts do not match."
for img_file, mask_file in zip(image_files, mask_files):
    assert os.path.splitext(img_file)[0] == os.path.splitext(mask_file)[0], f"Mismatch between {img_file} and {mask_file}."
    
annotation_file = os.path.join(data_dir, 'annotation.txt')
with open(annotation_file, 'w') as f:
    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)
        f.write(f"{img_path} | {mask_path} | {caption}\n")

print("annotation.txt done。")
