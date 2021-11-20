# Utils

import segmentation_models_pytorch as smp
from torch.optim import SGD, Adam, RMSprop


def get_loss(cfg: dict):
    if cfg['loss_fn'] == 'cross_entropy':
        return smp.losses.SoftBCEWithLogitsLoss()
    elif cfg['loss_fn'] == "focal_loss":
        return smp.losses.FocalLoss(mode='binary')
    elif cfg['loss_fn'] == 'dice_loss':
        return smp.losses.DiceLoss(mode='binary')
    elif cfg['loss_fn'] == 'tversky_loss':
        return smp.losses.TverskyLoss(mode='binary')
    else:
        raise NotImplemented("Loss not found!")


def get_optimizer(cfg: dict, params, lr: float):
    if cfg['optim'] == 'SGD':
        return SGD(params, lr)
    elif cfg['optim'] == 'Adam':
        return Adam(params, lr)
    elif cfg['optim'] == 'RMSprop':
        return RMSprop(params, lr)
    else:
        raise NotImplemented("Optim not found!")


def dict_to_str(data: dict):
    result = []
    for key, value in data.items():
        line = f"{key}={value}"
        result.append(line)

    return ", ".join(result)
