import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageOps, ImageFilter


def autocontrast(input_image, prob=0.5):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return input_image
    input_image = Image.fromarray(input_image[..., ::-1])
    input_image = ImageOps.autocontrast(input_image)
    input_image = np.array(input_image)[..., ::-1]
    return input_image


def equalize(input_image, prob=0.5):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return input_image
    input_image = Image.fromarray(input_image[..., ::-1])
    input_image = ImageOps.equalize(input_image)
    input_image = np.array(input_image)[..., ::-1]
    return input_image


def solarize(input_image, prob=0.5, threshold=128.):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return input_image
    input_image = Image.fromarray(input_image[..., ::-1])
    input_image = ImageOps.solarize(input_image, threshold=threshold)
    input_image = np.array(input_image)[..., ::-1]
    return input_image


def sharpness(input_image, prob=0.5, min=0, max=2, factor=None):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return input_image
    if factor is None:
        factor = np.random.uniform(min, max)
    input_image = Image.fromarray(input_image[..., ::-1])
    enhancer = ImageEnhance.Sharpness(input_image)
    input_image = enhancer.enhance(factor=factor)
    input_image = np.array(input_image)[..., ::-1]
    return input_image


def color(input_image, prob=0.5, min=0., max=1., factor=None):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return input_image
    if factor is None:
        factor = np.random.uniform(min, max)
    input_image = Image.fromarray(input_image[..., ::-1])
    enhancer = ImageEnhance.Color(input_image)
    input_image = enhancer.enhance(factor=factor)
    return np.array(input_image)[..., ::-1]


def contrast(input_image, prob=0.5, min=0.2, max=1., factor=None):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return input_image
    if factor is None:
        factor = np.random.uniform(min, max)
    input_image = Image.fromarray(input_image[..., ::-1])
    enhancer = ImageEnhance.Contrast(input_image)
    input_image = enhancer.enhance(factor=factor)
    return np.array(input_image)[..., ::-1]


def brightness(input_image, prob=0.5, min=0.8, max=1., factor=None):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return input_image
    if factor is None:
        factor = np.random.uniform(min, max)

    input_image = Image.fromarray(input_image[..., ::-1])
    enhancer = ImageEnhance.Brightness(input_image)
    input_image = enhancer.enhance(factor=factor)
    return np.array(input_image)[..., ::-1]


def blur(input_image, prob=0.5, factor=None):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return input_image
    if factor is None:
        factor = np.random.randint(1, 2)

    input_image = Image.fromarray(input_image[..., ::-1])
    input_image = input_image.filter(ImageFilter.GaussianBlur(radius=factor))
    return np.array(input_image)[..., ::-1]


def white_points(input_image, prob=0.5, min=100, max=200, factor=None):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return input_image
    if factor is None:
        factor = np.random.randint(min, max)

    noise = np.random.randint(1, factor, input_image.shape[:2], dtype=np.uint8) * 255
    noise = np.expand_dims(noise, axis=-1)
    noise = np.tile(noise, (1, 1, 3))
    noise[noise < 255] = 0
    noise[noise >= 255] = 255
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = cv2.bitwise_or(input_image, noise)

    return np.array(input_image)[..., ::-1]


class VisualEffect:
    """
    Struct holding parameters and applying image color transformation.
    Args
        solarize_threshold:
        color_factor: A factor for adjusting color.
        contrast_factor: A factor for adjusting contrast.
        brightness_factor: A factor for adjusting brightness.
        sharpness_factor: A factor for adjusting sharpness.
    """

    def __init__(
            self,
            color_factor=None,
            contrast_factor=None,
            brightness_factor=None,
            sharpness_factor=None,
            blur_factor=None,
            white_points_factor=None,
            color_prob=0.5,
            contrast_prob=0.5,
            brightness_prob=0.5,
            sharpness_prob=0.5,
            autocontrast_prob=0.5,
            equalize_prob=0.5,
            solarize_prob=0.1,
            solarize_threshold=128.,
            blur_prob=0.5,
            white_points_prob=0.05,

    ):
        self.color_factor = color_factor
        self.contrast_factor = contrast_factor
        self.brightness_factor = brightness_factor
        self.sharpness_factor = sharpness_factor
        self.blur_factor = blur_factor
        self.white_points_factor = white_points_factor

        self.color_prob = color_prob
        self.contrast_prob = contrast_prob
        self.brightness_prob = brightness_prob
        self.sharpness_prob = sharpness_prob
        self.autocontrast_prob = autocontrast_prob
        self.equalize_prob = equalize_prob
        self.solarize_prob = solarize_prob
        self.solarize_threshold = solarize_threshold

        self.blur_prob = blur_prob
        self.white_points_prob = white_points_prob

    def __call__(self, input_image):
        """
        Apply a visual effect on the image.
        Args
            image: Image to adjust
        """

        # Image Enhance
        random_enhance_id = np.random.randint(0, 4)

        if random_enhance_id == 0:
            input_image = color(input_image, prob=self.color_prob, factor=self.color_factor)
        elif random_enhance_id == 1:
            input_image = contrast(input_image, prob=self.contrast_prob, factor=self.contrast_factor)
        elif random_enhance_id == 2:
            input_image = brightness(input_image, prob=self.brightness_prob, factor=self.brightness_factor)
        else:
            input_image = sharpness(input_image, prob=self.sharpness_prob, factor=self.sharpness_factor)

        # Image Else
        random_ops_id = np.random.randint(0, 3)

        if random_ops_id == 0:
            input_image = autocontrast(input_image, prob=self.autocontrast_prob)
        elif random_ops_id == 1:
            input_image = equalize(input_image, prob=self.equalize_prob)
        else:
            input_image = solarize(input_image, prob=self.solarize_prob, threshold=self.solarize_threshold)

        # Image Noised
        random_noise_id = np.random.randint(0, 2)

        if random_noise_id == 0:
            input_image = blur(input_image, prob=self.blur_prob, factor=self.blur_factor)
        else:
            input_image = white_points(input_image,
                                       prob=self.white_points_prob,
                                       min=120, max=200,
                                       factor=self.white_points_factor)

        return input_image


if __name__ == '__main__':
    from generators.voc2 import PascalVocGenerator
    import cv2

    train_generator = PascalVocGenerator(
        r'C:\VOCdevkit\VOC2012+2007',
        'trainval',
        skip_difficult=True,
        batch_size=1
    )

    visual_effect = VisualEffect()
    for i in range(train_generator.size()):
        image = train_generator.load_image(i)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        annotations = train_generator.load_annotations(i)
        boxes = annotations['bboxes']

        for box in boxes.astype(np.int32):
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

        src_image = image.copy()

        image = visual_effect(image)

        cv2.namedWindow('image')
        cv2.imshow('image', np.concatenate([src_image, image], axis=1))
        cv2.waitKey(0)
