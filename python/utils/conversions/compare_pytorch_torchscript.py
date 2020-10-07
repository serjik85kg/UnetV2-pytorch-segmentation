import torch

from plugin_unet.common import create_model_cpu_for_inference
from sd_lib.task.paths import TaskPaths

"""
This script make the weights conversion
from Torch format to TorchScript format
and compare them using dummy input.
Max diff must be -> 0.0
"""

model_dir = TaskPaths.MODEL_DIR

example = torch.rand(1, 3, 256, 256).cpu()

model_pytorch = create_model_cpu_for_inference(n_cls=2, model_dir=model_dir)
#model_pytorch.cuda()
output_pytorch = model_pytorch(example)
# print(output_pytorch.shape)
# print(output_pytorch[0, 0:10])
print('Original:')
print('min abs value:{}'.format(torch.min(torch.abs(output_pytorch))))
print('Torchscript:')
traced_script_module = torch.jit.trace(model_pytorch, example)
ts_output = traced_script_module(example)
# print(ts_output.shape)
# print(ts_output[0, 0:10])
print('min abs value:{}'.format(torch.min(torch.abs(ts_output))))
print('Dif sum:')
abs_diff = torch.abs(output_pytorch-ts_output) # cuda:0 ??
print(torch.sum(abs_diff))
print('max dif:{}'.format(torch.max(abs_diff)))


# Test Torchscript inference
# import cv2
# from NN.plugins.nn.unet_v2.src.dataset import input_image_normalizer
# import matplotlib.pyplot as plt
#
# img = cv2.imread('47f.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (256,256))
# raw_input = input_image_normalizer(img)
# model_input = torch.stack([raw_input], 0)
# output = traced_script_module(model_input)
# output = output.data.cpu().numpy()[0]
# plt.imshow(np.argmax(output, axis=0))
# plt.show()