import argparse

from datasets.base import DunhangDataset, MuralDataset


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def dataset(folder,
            image_size,
            exts=['jpg', 'jpeg', 'png', 'tiff'],
            augment_flip=False,
            convert_image_to=None,
            equalizeHist=False,
            crop_patch=True,
            sample=False,
            is_dunhuang=True):
    
    if is_dunhuang:
        return DunhangDataset(folder,
                        image_size,
                        exts=exts,
                        augment_flip=augment_flip,
                        convert_image_to=convert_image_to,
                        equalizeHist=equalizeHist,
                        crop_patch=crop_patch,
                        sample=sample)
    else:
        return MuralDataset(folder,
                        image_size,
                        exts=exts,
                        augment_flip=augment_flip,
                        convert_image_to=convert_image_to,
                        equalizeHist=equalizeHist,
                        crop_patch=crop_patch,
                        sample=sample)