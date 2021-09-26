from __future__ import division
import numpy as np
from .transform import change_transform_origin
import cv2
from PIL import Image


def read_image_bgr(path):
    """
    Read an image in BGR format.
    Args
        path: Path to the image.
    """
    # We deliberately don't use cv2.imread here, since it gives no feedback on errors while reading the image.
    image = np.asarray(Image.open(path).convert('RGB'))
    return image[:, :, ::-1].copy()


class TransformParameters:
    def __init__(self,
                 fill_mode='nearest',
                 interpolation='linear',
                 constant_value=0,
                 relative_translation=True):

        self.fill_mode = fill_mode
        self.interpolation = interpolation
        self.constant_value = constant_value
        self.relative_translation = relative_translation

    def cvBorderMode(self):
        if self.fill_mode is 'constant':
            return cv2.BORDER_CONSTANT
        if self.fill_mode is 'nearest':
            return cv2.BORDER_REPLICATE
        if self.fill_mode is 'reflect':
            return cv2.BORDER_REFLECT_101
        if self.fill_mode is 'wrap':
            return cv2.BORDER_WRAP

    def cvInterpolation(self):
        if self.interpolation is 'nearest':
            return cv2.INTER_NEAREST
        if self.interpolation is 'cubic':
            return cv2.INTER_CUBIC
        if self.interpolation is 'area':
            return cv2.INTER_AREA
        if self.interpolation is 'liner':
            return cv2.INTER_LINEAR
        if self.interpolation is 'lanczos4':
            return cv2.INTER_LANCZOS4


def apply_transform(matrix, image, params):

    outputs = cv2.warpAffine(
        image,
        matrix[:2, :],
        dsize=(image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
        borderMode=cv2.BORDER_REPLICATE
        # flags=params.cvInterpolation,
        # borderMode=params.cvBorderMode,
        # borderValue=params.constant_value
    )
    return outputs


def compute_resize_scale(image_shape, min_side=800, max_side=1333):
    # (600, 800, 3)
    (rows, cols, _) = image_shape

    # 600
    smallest_side = min(rows, cols)
    # 800
    largest_side = max(rows, cols)

    # 800 / 600 = 4 / 3 = 1.3
    scale = min_side / smallest_side

    # 800 * 1.333- = 1066 < 1333
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    return scale


def resize_image(image, min_side=600, max_side=800):
    scale = compute_resize_scale(image.shape, min_side, max_side)
    return cv2.resize(image, None, fx=scale, fy=scale), scale


def preprocess_image(x, mode='caffe'):
    x = x.astype(np.float32)
    if mode is 'tf':
        x /= 127.5
        x -= 1.

    elif mode is 'caffe':
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x


def _check_range(val_range, min_val=None, max_val=None):
    if val_range[0] > val_range[1]:
        raise ValueError('interval lower bound is smaller than upper bound !')
    if min_val is not None and val_range[0] < min_val:
        raise ValueError('invalid interval lower bound !')
    if max_val is not None and val_range[1] > max_val:
        raise ValueError('invalid interval upper bound')


def _uniform(val_range):
    return np.random.uniform(val_range[0], val_range[1])


def _clip(image):
    return np.clip(image, 0, 255).astype(np.uint8)


def adjust_transform_for_image(transform, image, relative_translation):
    h, w, c = image.shape
    result = transform

    if relative_translation:
        result[0:2, 2] *= [w, h]

    result = change_transform_origin(transform, (0.5 * w, 0.5 * h))
    return result


def adjust_contrast(image, factor):
    mean = image.mean(axis=0).mean(axis=0)
    return _clip((image - mean) * factor + mean)


def adjust_brightness(image, delta):
    return _clip(image + delta * 255)


def adjust_hue(image, delta):
    image[..., 0] = np.mod(image[..., 0] + delta * 180, 180)
    return image


def adjust_saturation(image, factor):
    image[..., 1] = np.clip(image[..., 1] * factor, 0, 255)
    return image


class VisualEffect:
    def __init__(self,
                 contrast_factor,
                 brightness_factor,
                 hue_delta,
                 saturation_factor
                 ):
        self.contrast_factor = contrast_factor
        self.brightness_factor = brightness_factor
        self.hue_delta = hue_delta
        self.saturation_factor = saturation_factor

    def __call__(self, image):
        if self.contrast_factor:
            image = adjust_contrast(image, self.contrast_factor)
        if self.brightness_factor:
            image = adjust_brightness(image, self.brightness_factor)

        if self.hue_delta or self.saturation_factor:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            if self.hue_delta:
                image = adjust_hue(image, self.hue_delta)

            if self.saturation_factor:
                image = adjust_saturation(image, self.saturation_factor)

            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        return image


def random_visual_effect_generator(contrast_range=(0.9, 1),
                                   brightness_range=(-0.1, 0.1),
                                   hue_range=(-0.05, 0.05),
                                   saturation_range=(0.95, 1.05)
                                   ):

    _check_range(contrast_range, 0)
    _check_range(brightness_range, -1, 1)
    _check_range(hue_range, -1, 1)
    _check_range(saturation_range, 0)

    def _generate():
        while True:
            yield VisualEffect(
                contrast_factor=_uniform(contrast_range),
                brightness_factor=_uniform(brightness_range),
                hue_delta=_uniform(hue_range),
                saturation_factor=_uniform(saturation_range)
            )

    return _generate()

