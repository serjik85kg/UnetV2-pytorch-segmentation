import torch
from torch.nn import DataParallel
from plugins.nn.unet_v2.src.common import construct_unet
from sd_lib.nn.pytorch.weights import WeightsRW

from sd_lib.task.paths import TaskPaths
model_dir = TaskPaths.MODEL_DIR
model_output_dir = '../../test_outputs/'
model_output_name = 'model.onnx'

state_dict = torch.load(model_dir + '/model.pt', map_location='cpu')
model = construct_unet(n_cls=2)
model = DataParallel(model).cpu() # cpu matters. cpu()
model.load_state_dict(state_dict)
model.eval()
# model = construct_unet(n_cls=2)
# model = DataParallel(model)
# model = WeightsRW(model_dir).load_strictly(model)

dummy_input = torch.rand(1, 3, 256, 256)

input_names = ["my_input"]
output_names = ["my_output"]

torch.onnx.export(model.module,
                  dummy_input,
                  model_output_dir + model_output_name,
                  verbose=True,
                  input_names=input_names,
                  output_names=output_names)

# use model simplifier after it if necessary
# https://github.com/daquexian/onnx-simplifier


