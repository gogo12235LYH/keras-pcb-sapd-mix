import cv2
import numpy as np
from generators.voc_generator import PascalVocGenerator
from utils.transform import random_transform_generator
from utils.image import random_visual_effect_generator
from utils.image import preprocess_image
from models import osd, loss
import tensorflow.keras as keras
import tensorflow as tf


def show_annotations():
    common_args = {
        'image_min_side': 800,
        'image_max_side': 1333,
    }
    generator = PascalVocGenerator(data_dir=r'E:\VOCdevkit\VOC2012', set_name='val', **common_args)
    for image_group, annotation_group, targets in generator:
        locations = targets[0]
        batch_regr_targets = targets[1]
        batch_cls_targets = targets[2]
        batch_centerness_targets = targets[3]
        for image, annotation, regr_targets, cls_targets, centerness_targets in zip(image_group, annotation_group,
                                                                                    batch_regr_targets,
                                                                                    batch_cls_targets,
                                                                                    batch_centerness_targets):
            print(image.shape)
            gt_boxes = annotation['bboxes']
            for gt_box in gt_boxes:
                gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_box
                cv2.rectangle(image, (int(gt_xmin), int(gt_ymin)), (int(gt_xmax), int(gt_ymax)), (0, 255, 255), 1)
            pos_indices = np.where(centerness_targets[:, 1] == 1)[0]
            for pos_index in pos_indices:
                cx, cy = locations[pos_index]
                l, t, r, b, *_ = regr_targets[pos_index]
                xmin = cx - l
                ymin = cy - t
                xmax = cx + r
                ymax = cy + b
                class_id = np.argmax(cls_targets[pos_index])
                centerness = centerness_targets[pos_index][0]

                # cv2.putText(image, '{:.1f}'.format(centerness), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                cv2.putText(image, str(class_id), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.circle(image, (round(cx), round(cy)), 3, (255, 255, 255), -1)
                cv2.rectangle(image, (round(xmin), round(ymin)), (round(xmax), round(ymax)), (0, 0, 255), 1)

            # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image', image)
            cv2.waitKey(0)


def verify_no_negative_reg():
    transform_generator = random_transform_generator(
        min_rotation=-0.1,
        max_rotation=0.1,
        min_translation=(-0.1, -0.1),
        max_translation=(0.1, 0.1),
        min_shear=-0.1,
        max_shear=0.1,
        min_scaling=(0.9, 0.9),
        max_scaling=(1.1, 1.1),
        flip_x_chance=0.5,
        flip_y_chance=0.5,
    )
    visual_effect_generator = random_visual_effect_generator(
        contrast_range=(0.9, 1.1),
        brightness_range=(-.1, .1),
        hue_range=(-0.05, 0.05),
        saturation_range=(0.95, 1.05)
    )
    common_args = {
        'batch_size': 1,
        'image_min_side': 256,
        'image_max_side': 256,
        'preprocess_image': preprocess_image,
    }
    generator = PascalVocGenerator(
        r'C:\VOCdevkit\VOC2012',
        'trainval',
        transform_generator=transform_generator,
        visual_effect_generator=visual_effect_generator,
        skip_difficult=True,
        **common_args
    )
    return generator

    # i = 0
    # o = []
    # for image_group, targets in generator:
    #     i += 1
    #     o.append((image_group, targets))
    #     print(i)
    #     if i >= 20:
    #         break
    # return o


# if __name__ == '__main__':
def main():
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = tf.compat.v1.InteractiveSession(config=config)

    def init_():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    x = verify_no_negative_reg()
    test_model = osd.test_build(num_cls=20)
    test_model.compile(optimizer=keras.optimizers.Adam(1e-5),
                       loss={
                           'regression': loss.iou_loss(),
                           'classification': loss.focal_loss(),
                           'centerness': loss.bce_center_loss()
                       })
    history = test_model.fit_generator(x, epochs=100, use_multiprocessing=False)
    # show_annotations()
    # from utils.image import TransformParameters
    # x = TransformParameters()
