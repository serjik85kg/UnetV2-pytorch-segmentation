import cv2
import numpy as np

from matplotlib import pyplot as plt
import torch
# required and recommended pytorch 0.4.0
print(torch.__version__)

def apply_mask(image, mask, out_shape=None):
    applied_mask = np.zeros_like(image)
    mask = np.argmax(mask, axis=2)
    _, contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    color = [255,0,0]
    if contours is not None:
        cv2.fillPoly(applied_mask, contours, color)
    image = cv2.addWeighted(image, 1, applied_mask, 0.8, 0)
    return image
    # plt.imshow(image)
    # plt.show()

# show outputs
def single_show(model_out):
    plt.imshow(np.argmax(model_out, axis=2))
    plt.show()

# test image path
img_path = 'examples/47f.jpg'
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

from plugin_unet.inference import UnetV2SimpleInferencer

unet_module = UnetV2SimpleInferencer()
output = unet_module.inference_single_image(image)

image_c = image
image_c = apply_mask(image_c, output)
single_show(output)
plt.imshow(image_c)
plt.show()

