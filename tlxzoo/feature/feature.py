from ..config import BaseFeatureConfig
import tensorlayerx as tlx
import numpy as np
from ..utils.registry import Registers


class BaseFeature(object):
    def __init__(
            self,
            config,
            **kwargs
    ):
        self.config = config
        self.config.feature_class = self.__class__.__name__

    @classmethod
    def from_pretrained(
            cls, pretrained_path, **kwargs
    ):
        config = BaseFeatureConfig.from_pretrained(pretrained_path, **kwargs)

        return Registers.features[config.feature_class](config)

    def save_pretrained(self, save_path):
        self.config.save_pretrained(save_path)


class BaseImageFeature(BaseFeature):
    def _image_type(self, image):
        if isinstance(image, np.ndarray):
            raise

    def resize(self, image, size):
        # self._image_type(image)
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, list):
            size = tuple(size)
        img = tlx.prepro.imresize(image, size)
        return img

    def normalize(self, image, mean, std):
        # self._image_type(image)
        if isinstance(mean, list):
            mean = np.array(mean, dtype=np.float32).reshape([1, 1, 3])
        if std:
            return (image - mean) / std
        return image - mean