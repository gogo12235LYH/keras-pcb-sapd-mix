import numpy as np
import random
import warnings
import cv2
from tensorflow import keras
from utils import get_fm_shapes
import config


class Generator(keras.utils.Sequence):
    """
    Abstract generator class.
    """

    def __init__(
            self,
            phi=0,
            image_sizes=(512, 640, 768, 896, 1024, 1280, 1408),
            misc_effect=None,
            visual_effect=None,
            batch_size=1,
            group_method='random',  # one of 'none', 'random', 'ratio'
            shuffle_groups=True,
            max_gt_boxes=100,
    ):
        """
        Initialize Generator object.
        Args:
            batch_size: The size of the batches to generate.
            group_method: Determines how images are grouped together (defaults to 'ratio', one of ('none', 'random', 'ratio')).
            shuffle_groups: If True, shuffles the groups each epoch.
            image_sizes:
        """
        self.image_size = image_sizes[phi]
        self.misc_effect = misc_effect
        self.visual_effect = visual_effect
        self.batch_size = int(batch_size)
        self.group_method = group_method
        self.shuffle_groups = shuffle_groups
        self.max_gt_boxes = max_gt_boxes

        self.pyramid_levels = (3, 4, 5, 6, 7)
        self.strides = [2 ** i for i in self.pyramid_levels]
        self.fm_shapes = get_fm_shapes((self.image_size, self.image_size), self.pyramid_levels)

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # group the images in batch to handle
        self.groups = None  # remove warning
        self.group_images()

        # Shuffle when initializing
        if self.shuffle_groups:
            random.shuffle(self.groups)

    def on_epoch_end(self):
        if self.shuffle_groups:
            random.shuffle(self.groups)

    def size(self):
        """
        Size of the dataset.
        """
        raise NotImplementedError('size method not implemented')

    def get_anchors(self):
        """
        loads the anchors from a txt file
        """
        with open(self.anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        # (N, 2), wh
        return np.array(anchors).reshape(-1, 2)

    def num_classes(self):
        """
        Number of classes in the dataset.
        """
        raise NotImplementedError('num_classes method not implemented')

    def has_label(self, label):
        """
        Returns True if label is a known label.
        """
        raise NotImplementedError('has_label method not implemented')

    def has_name(self, name):
        """
        Returns True if name is a known class.
        """
        raise NotImplementedError('has_name method not implemented')

    def name_to_label(self, name):
        """
        Map name to label.
        """
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        """
        Map label to name.
        """
        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):
        """
        Compute the aspect ratio for an image with image_index.
        """
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        raise NotImplementedError('load_image method not implemented')

    def load_annotations(self, image_index):
        """
        Load annotations for an image_index.
        """
        raise NotImplementedError('load_annotations method not implemented')

    def load_annotations_group(self, group):
        """
        Load annotations for all images in group.
        """
        annotations_group = [self.load_annotations(image_index) for image_index in group]
        for annotations in annotations_group:
            assert (isinstance(annotations,
                               dict)), '\'load_annotations\' should return a list of dictionaries, received: {}'.format(
                type(annotations))
            assert (
                    'labels' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'
            assert (
                    'bboxes' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'

        return annotations_group

    @staticmethod
    def filter_annotations(image_group, annotations_group, group):
        """
        Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
                (annotations['bboxes'][:, 0] < 0) |
                (annotations['bboxes'][:, 1] < 0) |
                (annotations['bboxes'][:, 2] <= 0) |
                (annotations['bboxes'][:, 3] <= 0) |
                (annotations['bboxes'][:, 2] > image.shape[1]) |
                (annotations['bboxes'][:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    group[index],
                    image.shape,
                    annotations['bboxes'][invalid_indices, :]
                ))
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], invalid_indices, axis=0)
            # if annotations['bboxes'].shape[0] == 0:
            #     warnings.warn('Image with id {} (shape {}) contains no valid boxes before transform'.format(
            #         group[index],
            #         image.shape,
            #     ))
        return image_group, annotations_group

    @staticmethod
    def clip_transformed_annotations(image_group, annotations_group):
        """
        Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        # test all annotations
        filtered_image_group = []
        filtered_annotations_group = []
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            image_height = image.shape[0]
            image_width = image.shape[1]
            # x1
            annotations['bboxes'][:, 0] = np.clip(annotations['bboxes'][:, 0], 0, image_width - 2)
            # y1
            annotations['bboxes'][:, 1] = np.clip(annotations['bboxes'][:, 1], 0, image_height - 2)
            # x2
            annotations['bboxes'][:, 2] = np.clip(annotations['bboxes'][:, 2], 1, image_width - 1)
            # y2
            annotations['bboxes'][:, 3] = np.clip(annotations['bboxes'][:, 3], 1, image_height - 1)
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            small_indices = np.where(
                (annotations['bboxes'][:, 2] - annotations['bboxes'][:, 0] < 3) |
                (annotations['bboxes'][:, 3] - annotations['bboxes'][:, 1] < 3)
            )[0]

            # delete invalid indices
            if len(small_indices):
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], small_indices, axis=0)

            filtered_image_group.append(image)
            filtered_annotations_group.append(annotations_group[index])

        return filtered_image_group, filtered_annotations_group

    def load_image_group(self, group):
        """
        Load images for all images in a group.
        """
        return [self.load_image(image_index) for image_index in group]

    # Augmentation method
    def random_visual_effect_group_entry(self, image, annotations):
        """
        Randomly transforms image and annotation.
        """
        # apply visual effect
        image = self.visual_effect(image)
        return image, annotations

    def random_visual_effect_group(self, image_group, annotations_group):
        """
        Randomly apply visual effect on each image.
        """
        assert (len(image_group) == len(annotations_group))

        if self.visual_effect is None:
            # do nothing
            return image_group, annotations_group

        for index in range(len(image_group)):
            # apply effect on a single group entry
            image_group[index], annotations_group[index] = self.random_visual_effect_group_entry(
                image_group[index], annotations_group[index]
            )

        return image_group, annotations_group

    def random_misc_group_entry(self, image, annotations):
        """
        Randomly transforms image and annotation.
        """
        # randomly transform both image and annotations
        image, annotations = self.misc_effect(image, annotations)
        return image, annotations

    def random_misc_group(self, image_group, annotations_group):
        """
        Randomly transforms each image and its annotations.
        """

        assert (len(image_group) == len(annotations_group))

        if self.misc_effect is None:
            return image_group, annotations_group

        for index in range(len(image_group)):
            # transform a single group entry
            image_group[index], annotations_group[index] = self.random_misc_group_entry(image_group[index],
                                                                                        annotations_group[index])

        return image_group, annotations_group

    def preprocess_group_entry(self, image, annotations):
        """
        Preprocess image and its annotations.
        """

        # preprocess the image
        image, scale, offset_h, offset_w = self.preprocess_image(image)

        # apply resizing to annotations too
        annotations['bboxes'] *= scale
        annotations['bboxes'][:, [0, 2]] += offset_w
        annotations['bboxes'][:, [1, 3]] += offset_h
        return image, annotations

    def preprocess_group(self, image_group, annotations_group):
        """
        Preprocess each image and its annotations in its group.
        """
        assert (len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # preprocess a single group entry
            image_group[index], annotations_group[index] = self.preprocess_group_entry(image_group[index],
                                                                                       annotations_group[index])

        return image_group, annotations_group

    def group_images(self):
        """
        Order the images according to self.order and makes groups of self.batch_size.
        """
        # determine the order of the images

        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                       range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group, annotations_group):
        """
        Compute inputs for the network using an image_group.
        """
        batch_images = np.array(image_group).astype(np.float32)
        batch_gt_boxes = np.zeros((len(image_group), self.max_gt_boxes, 5), dtype=np.float32)
        batch_num_gt_boxes = np.zeros((len(image_group), 1), dtype=np.int32)
        batch_fm_shapes = np.tile(self.fm_shapes[None], (len(image_group), 1, 1))

        # copy all images to the upper left part of the image batch object
        for image_index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # float64 --> float32
            boxes = annotations['bboxes'].astype(np.float32)
            # int64 --> float32
            labels = annotations['labels'].astype(np.float32)
            gt_boxes = np.concatenate([boxes, labels[:, None]], axis=-1)
            batch_gt_boxes[image_index, :gt_boxes.shape[0]] = gt_boxes
            batch_num_gt_boxes[image_index, 0] = gt_boxes.shape[0]
        return [batch_images, batch_gt_boxes, batch_num_gt_boxes, batch_fm_shapes]

    @staticmethod
    def compute_targets(image_group):
        """
        Compute target outputs for the network using images and their annotations.
        """

        targets = [
            np.zeros((len(image_group),), dtype=np.float32),  # cls_loss
            np.zeros((len(image_group),), dtype=np.float32),  # regr_loss
            np.zeros((len(image_group),), dtype=np.float32),  # feature_select_loss
        ]
        return targets

    def compute_inputs_targets(self, group, debug=False):
        """
        Compute inputs and target outputs for the network.
        """

        # load images and annotations
        # list
        image_group = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # randomly apply visual effect
        image_group, annotations_group = self.random_visual_effect_group(image_group, annotations_group)

        # randomly apply misc effect
        image_group, annotations_group = self.random_misc_group(image_group, annotations_group)

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # check validity of annotations
        image_group, annotations_group = self.clip_transformed_annotations(image_group, annotations_group)

        assert len(image_group) != 0
        assert len(image_group) == len(annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group, annotations_group)

        # compute network targets
        targets = self.compute_targets(image_group)

        if debug:
            return inputs, targets, annotations_group

        return inputs, targets

    def __len__(self):
        """
        Number of batches for generator.
        """
        return len(self.groups)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        group = self.groups[index]
        inputs, targets = self.compute_inputs_targets(group)
        return inputs, targets

    def preprocess_image(self, image):
        image_height, image_width = image.shape[:2]

        if image_height > image_width:
            scale = self.image_size / image_height
            resized_height = self.image_size
            resized_width = int(image_width * scale)
        else:
            scale = self.image_size / image_width
            resized_height = int(image_height * scale)
            resized_width = self.image_size

        image = cv2.resize(image, (resized_width, resized_height))
        new_image = np.ones((self.image_size, self.image_size, 3), dtype=np.float32) * 128.
        offset_h = (self.image_size - resized_height) // 2
        offset_w = (self.image_size - resized_width) // 2
        new_image[offset_h:offset_h + resized_height, offset_w:offset_w + resized_width] = image.astype(np.float32)

        # image's channel is 'RGB'
        if config.BACKBONE_TYPE == 'ResNetV1':
            # Caffe
            new_image = new_image[..., ::-1]  # RGB -> BGR
            new_image -= [103.939, 116.779, 123.68]

        elif config.BACKBONE_TYPE == 'ResNetV2':
            new_image /= 127.5
            new_image -= 1.

        elif config.BACKBONE_TYPE == 'EffNet':
            new_image = new_image

        elif config.BACKBONE_TYPE in ['DenseNet', 'SEResNet']:
            # Torch
            new_image /= 255.
            new_image -= self.mean
            new_image /= self.std

        return new_image, scale, offset_h, offset_w

    def get_augmented_data(self, group):
        """
        Compute inputs and target outputs for the network.
        """

        # load images and annotations
        # list
        image_group = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # randomly apply visual effect
        # image_group, annotations_group = self.random_visual_effect_group(image_group, annotations_group)

        # randomly transform data
        # image_group, annotations_group = self.random_transform_group(image_group, annotations_group)

        # randomly apply misc effect
        # image_group, annotations_group = self.random_misc_group(image_group, annotations_group)

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # check validity of annotations
        image_group, annotations_group = self.clip_transformed_annotations(image_group, annotations_group)

        assert len(image_group) != 0
        assert len(image_group) == len(annotations_group)

        # compute alphas for targets
        self.compute_alphas_and_ratios(annotations_group)

        return image_group, annotations_group
