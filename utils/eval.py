import tensorflow.keras as keras
import numpy as np
import os
import cv2
import pickle
import progressbar
from utils.compute_overlap import compute_overlap


def _compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Args:
        recall: The recall curve (list).
        precision: The precision curve (list).
    Returns:
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    m_rec = np.concatenate(([0.], recall, [1.]))
    m_pre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(m_pre.size - 1, 0, -1):
        m_pre[i - 1] = np.maximum(m_pre[i - 1], m_pre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(m_rec[1:] != m_rec[:-1])[0]

    # and sum (delta recall) * prec
    ap = np.sum((m_rec[i + 1] - m_rec[i]) * m_pre[i + 1])
    return ap


def _get_detections(generator, model, score_threshold=0.05, max_detections=100, save_path=None):
    """
    Get the detections from the model using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_class_detections, 5]
    Args:
        generator: The generator used to run images through the model.
        model: The model to run on the images.
        score_threshold: The score confidence threshold to use.
        max_detections: The maximum number of detections to use per image.
        save_path: The path to save the images with visualized detections to.
    Returns:
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in
                      range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        raw_image = generator.load_image(i)
        # FCOS
        # image = generator.preprocess_image(raw_image.copy())

        # FS
        image, scale, offset_h, offset_w = generator.preprocess_image(raw_image.copy())
        h, w = image.shape[:2]

        # FCOS
        # image, scale = generator.resize_image(image)

        # # run network
        # boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]
        #
        # # correct boxes for image scale
        # boxes /= scale

        # FS
        boxes, scores, *_, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
        boxes[..., [0, 2]] = boxes[..., [0, 2]] - offset_w
        boxes[..., [1, 3]] = boxes[..., [1, 3]] - offset_h
        boxes /= scale
        boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w - 1)
        boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h - 1)
        boxes[:, :, 2] = np.clip(boxes[:, :, 2], 0, w - 1)
        boxes[:, :, 3] = np.clip(boxes[:, :, 3], 0, h - 1)

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        # (n, 4)
        image_boxes = boxes[0, indices[scores_sort], :]
        # (n, )
        image_scores = scores[scores_sort]
        # (n, )
        image_labels = labels[0, indices[scores_sort]]
        # (n, 6)
        image_detections = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        # if save_path is not None:
        #     draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
        #     draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name,
        #                     score_threshold=score_threshold)
        #
        #     cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

    return all_detections


def _get_annotations(generator):
    """
    Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_annotations[num_images][num_classes] = annotations[num_class_annotations, 5]
    Args:
        generator: The generator used to retrieve ground truth annotations.
    Returns:
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_annotations[i][label] = annotations['bboxes'][annotations['labels'] == label, :].copy()

    return all_annotations


def evaluate(
        generator,
        model,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        save_path=None,
        epoch=0
):
    """
    Evaluate a given dataset using a given model.
    Args:
        generator: The generator that represents the dataset to evaluate.
        model: The model to evaluate.
        iou_threshold: The threshold used to consider when a detection is positive or negative.
        score_threshold: The score confidence threshold to use for detections.
        max_detections: The maximum number of detections to use per image.
        save_path: The path to save images with visualized detections to.
        epoch: epoch index
    Returns:
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections,
                                     save_path=save_path)
    all_annotations = _get_annotations(generator)
    average_precisions = {}

    # all_detections = pickle.load(open('fcos/all_detections_11.pkl', 'rb'))
    # all_annotations = pickle.load(open('fcos/all_annotations.pkl', 'rb'))
    # pickle.dump(all_detections, open('fcos/all_detections_{}.pkl'.format(epoch + 1), 'wb'))
    # pickle.dump(all_annotations, open('fcos/all_annotations_{}.pkl'.format(epoch + 1), 'wb'))

    # process detections and annotations
    for label in range(generator.num_classes()):
        if not generator.has_label(label):
            continue

        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue
                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    return average_precisions


def evaluate2(
        generator,
        model,
        score_threshold=0.01,
        max_detections=100,
):
    """
    Evaluate a given dataset using a given model.
    Args:
        generator: The generator that represents the dataset to evaluate.
        model: The model to evaluate.
        iou_threshold: The threshold used to consider when a detection is positive or negative.
        score_threshold: The score confidence threshold to use for detections.
        max_detections: The maximum number of detections to use per image.
        visualize: Show the visualized detections or not.
    Returns:
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections )

    all_annotations = _get_annotations(generator)
    coco_aps = {}
    num_tp = 0
    num_fp = 0

    iou_range = np.arange(0.5, 1, 0.05)

    for iou in range(10):
        average_precisions = {}

        # process detections and annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0

            for i in range(generator.size()):
                detections = all_detections[i][label]
                annotations = all_annotations[i][label]
                num_annotations += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
                        continue

                    overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_range[iou] and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)

                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0, 0, 0, 0
                continue

            # sort by score
            indices = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            if false_positives.shape[0] == 0:
                num_fp += 0
            else:
                num_fp += false_positives[-1]

            if true_positives.shape[0] == 0:
                num_tp += 0
            else:
                num_tp += true_positives[-1]

            # compute recall and precision
            recall = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision = _compute_ap(recall, precision)
            average_precisions[label] = average_precision, num_annotations, recall, precision

        coco_aps[iou] = average_precisions.copy()

    print('num_fp={}, num_tp={}'.format(num_fp, num_tp))

    return coco_aps
