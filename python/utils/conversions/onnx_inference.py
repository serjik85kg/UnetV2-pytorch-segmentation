import onnx
import onnxruntime
print('ONNX version: ', onnx.__version__)
print('ONNXruntime version: ', onnxruntime.__version__)
import numpy as np

from plugins.nn.unet_v2.src.dataset import input_image_normalizer
import torchvision.transforms as trans

import matplotlib.pyplot as plt

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

model_output_dir = '../../test_outputs/'
model_output_name = 'model.onnx'
model_dir = model_output_dir + model_output_name
#simple_model_dir = 'my_network_simple.onnx'  # If use simply onnx version

onnx_model = onnx.load(model_dir)
onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession(model_dir)

from PIL import Image
import torchvision.transforms as transforms

img = Image.open('../../examples/47f.jpg')
resize = transforms.Resize([256, 256])
img = resize(img)

testTensor = trans.ToTensor()(img)
testTensor = trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(testTensor)

img = input_image_normalizer(img)
norm_img = img.data.cpu().numpy()
img.unsqueeze_(0)


ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
ort_outs = ort_session.run(None, ort_inputs)

img_out = ort_outs[0]
img_out = img_out.squeeze(0)

out_show = np.argmax(img_out, axis=0)
plt.imshow(out_show)
plt.show()