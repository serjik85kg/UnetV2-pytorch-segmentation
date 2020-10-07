# coding: utf-8
from torch.nn import DataParallel
from sd_lib import logger
from sd_lib.nn.pytorch.weights import WeightsRW
from plugin_unet.unet import construct_unet


def create_model(n_cls, device_ids):
    logger.info('Will construct model.')
    model = construct_unet(n_cls=n_cls)
    logger.info('Model has been constructed (w/out weights).')
    model = DataParallel(model, device_ids=device_ids).cuda()
    logger.info('Model has been loaded into GPU(s).', extra={'remapped_device_ids': device_ids})
    return model


def create_model_for_inference(n_cls, device_ids, model_dir):
    model = create_model(n_cls, device_ids)
    model = WeightsRW(model_dir).load_safe(model)
    model.eval()
    return model

def create_model_cpu(n_cls):
    logger.info('Will construct model-CPU')
    model = construct_unet(n_cls=n_cls)
    logger.info('Model-CPU has been constructed')
    model = DataParallel(model)
    return model

def create_model_cpu_for_inference(n_cls, model_dir):
    model = create_model_cpu(n_cls=n_cls)
    model = WeightsRW(model_dir).load_safe(model)
    model.eval()
    return model

