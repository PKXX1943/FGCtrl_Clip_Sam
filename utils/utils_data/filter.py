import numpy as np
import os
import glob
import cv2

def filter_annotations(annotation_path, output_path):
    with open(annotation_path, 'r') as file:
        lines = file.readlines()

    filtered_lines = []
    
    for line in lines[1:]:
        image_file, mask_file, caption = line.strip().split('|')
        mask_file = mask_file.strip()
        
        # 读取mask图像
        if mask_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            # 检查mask是否全为零
            if mask is not None and np.any(mask):
                filtered_lines.append(line.strip())
            else:
                print(f"Filtered out: {mask_file} is all zeros or could not be read.")
        else:
            print(f"Invalid mask file format: {mask_file}")

    # 写入新的annotation文件
    with open(output_path, 'w') as output_file:
        output_file.write(lines[0])
        for filtered_line in filtered_lines:
            output_file.write(filtered_line + '\n')

    print(f"Filtered annotations saved to: {output_path}")
            
anns = glob.glob('data/*/annotations/*.txt')
for file in anns:
    filter_annotations(file, file.replace('.txt', '_clear.txt'))