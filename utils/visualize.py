import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def show_anns(masks, gt_mask, filename, image, ious, boundary_ious):
    if len(masks) == 0:
        return
    for i, (mask, iou, biou) in enumerate(zip(masks, ious, boundary_ious)):
        image = image.resize((1024, 1024), Image.LANCZOS)
        image = cv2.cvtColor((np.array(image)), cv2.COLOR_BGR2GRAY)

        plt.figure(figsize=(10,10))
        plt.imshow(image, cmap='gray')
        show_mask(mask, plt.gca(), 1)
        show_mask(gt_mask, plt.gca(), -1)

        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_mask(mask, ax, random_color):
    if random_color == 0:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif random_color > 0:
        color = np.array([30/255, 144/255, 255/255, 0.4])
    else:
        color = np.array([255/255, 144/255, 30/255, 0.4])
    mask_image = mask.reshape(1024, 1024, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)