import torch
import torchvision
print('Pytorch version: ', torch.__version__)
print('Torchvision version: ', torchvision.__version__)

from torch.nn import DataParallel
from sd_lib import logger
from sd_lib.nn.pytorch.weights import WeightsRW
from plugin_unet.unet import construct_unet
from sd_lib.task.paths import TaskPaths
from plugin_unet.common import create_model_for_inference

"""
This script converts Pytorch weights to Torchscript format and save it.
"""

model_dir = TaskPaths.MODEL_DIR
#print(model_dir)
model_output_dir = '../../test_outputs/'
model_output_name = 'traced_model_unet.pt'

# TEMPORARILY!
# special for model CPU tracing
def create_model_cpu(n_cls):
    logger.info('Will construct model-CPU')
    model = construct_unet(n_cls=n_cls)
    logger.info('Model-CPU has been constructed')
    model = DataParallel(model).cpu()                # this is the only difference for tracing specialization but it matters ->cpu()
    return model

def create_model_cpu_for_inference(n_cls, model_dir):
    model = create_model_cpu(n_cls=n_cls)
    model = WeightsRW(model_dir).load_safe(model)
    model.eval()
    return model

# dummy input
# default configuration may be .cuda() or .cpu(). So it must be specialised later
example = torch.rand(1, 3, 256, 256)

TORCH_CUDA = torch.cuda.is_available()
TORCH_CUDA = False # HARDCODE for CPU only

# model creation cpu/gpu and specialization of example input.
if TORCH_CUDA:
    model = create_model_for_inference(n_cls=2, device_ids=[0], model_dir=model_dir)
    example.cuda()
else:
    model = create_model_cpu_for_inference(n_cls=2, model_dir=model_dir)
    example.cpu()

# make the torchscript unet model for inferencing on iOS without describing model architecture in code
traced_script_module = torch.jit.trace(model.module, example)

#traced_script_module.save("model_traced.pt")
traced_script_module.save(model_output_dir + model_output_name)