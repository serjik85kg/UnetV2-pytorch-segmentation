# coding: utf-8
import cv2

import numpy as np

from sd_lib.sly_logger import logger
from sd_lib.task.paths import TaskPaths

from sd_lib.imaging import image as sly_image

from plugin_unet.common import create_model_for_inference, create_model_cpu_for_inference
from plugin_unet.dataset import input_image_normalizer
import torch
import torch.nn.functional as torch_functional

class UnetImageInferenceInterface:
    def inference(self, image):
        raise NotImplementedError()

class UnetImageInferenceBase(UnetImageInferenceInterface):
    def __init__(self, model_dir=TaskPaths.MODEL_DIR):
        logger.info('Starting base single image inference applier init.')
        self.model_dir = model_dir
        logger.info('Base single image inference applier init done.')

    def get_default_config(self):
        return {}

    def _construct_and_fill_model(self):
        raise NotImplementedError()

    def inference(self, image):
        raise NotImplementedError()

GPU_DEVICE = 'gpu_device'

class UnetV2SimpleInferencer(UnetImageInferenceBase):

    def __init__(self, model_dir=TaskPaths.MODEL_DIR):
        super().__init__()
        self._construct_and_fill_model()

    @staticmethod
    def get_default_config():
        return {
            GPU_DEVICE: 0
        }

    def _construct_and_fill_model(self):
        if torch.cuda.is_available():
            self.model = create_model_for_inference(n_cls=2, device_ids=[0], model_dir=self.model_dir)
        else:
            self.model = create_model_cpu_for_inference(n_cls=2, model_dir=self.model_dir)

    def _infer_per_pixel_scores(self, raw_input, out_shape, apply_softmax=False):
        model_input = torch.stack([raw_input], 0)  # (classes, width, height) -> torch(1, classes, width, height)
        output = self.model(model_input)
        if apply_softmax:
            output = torch_functional.softmax(output, dim=1)
        output = output.data.cpu().numpy()[0]  # from batch to 3d
        output = np.transpose(output, (1, 2, 0))
        return sly_image.resize(output, out_shape)

    def inference_single_image(self, img):
        out_shape = img.shape[:2]
        resized_img = cv2.resize(img, (256, 256))
        raw_input = input_image_normalizer(resized_img)
        output = self._infer_per_pixel_scores(raw_input, out_shape)
        return output
