import onnx
import onnxruntime
import torch
import numpy as np

from plugin_unet.common import create_model_cpu_for_inference, create_model_for_inference
from plugin_unet.dataset import input_image_normalizer
from sd_lib.task.paths import TaskPaths

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# dummy input (general for both models)
x = torch.randn(1, 3, 256, 256)

# Create Torch Unet Model and load model.pt weights from model_dir
model_dir = TaskPaths.MODEL_DIR
model_torch = create_model_cpu_for_inference(n_cls=2, model_dir=model_dir)
# Torch Unet Model Inference
torch_out = model_torch(x)

# Create Onnx Model and load my_network.onnx weights which has already converted from Torch (see to_onnx.py)
model_output_dir = '../../test_outputs/'
model_output_name = 'model.onnx'
model_onnx_dir = model_output_dir + model_output_name
model_onnx = onnx.load(model_onnx_dir)
onnx.checker.check_model(model_onnx)
# Create ONNX Model inference module and run it
ort_session = onnxruntime.InferenceSession(model_onnx_dir)
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# Test outputs from models
test = np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
print("Exported model has been tested with ONNXRuntime, and the result looks good")

