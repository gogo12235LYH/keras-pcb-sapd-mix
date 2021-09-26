import cv2
import numpy as np


def preprocess_image(image, image_size):
    image_height, image_width = image.shape[:2]
    if image_height > image_width:
        scale = image_size / image_height
        resized_height = image_size
        resized_width = int(image_width * scale)
    else:
        scale = image_size / image_width
        resized_height = int(image_height * scale)
        resized_width = image_size
    image = cv2.resize(image, (resized_width, resized_height))
    new_image = np.ones((image_size, image_size, 3), dtype=np.float32) * 128.
    offset_h = (image_size - resized_height) // 2
    offset_w = (image_size - resized_width) // 2
    new_image[offset_h:offset_h + resized_height, offset_w:offset_w + resized_width] = image.astype(np.float32)
    new_image /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    new_image[..., 0] -= mean[0]
    new_image[..., 1] -= mean[1]
    new_image[..., 2] -= mean[2]
    new_image[..., 0] /= std[0]
    new_image[..., 1] /= std[1]
    new_image[..., 2] /= std[2]
    return new_image, scale, offset_h, offset_w


def rotate_image(image):
    rotate_degree = np.random.uniform(low=-45, high=45)
    h, w = image.shape[:2]
    # Compute the rotation matrix.
    M = cv2.getRotationMatrix2D(center=(w / 2, h / 2),
                                angle=rotate_degree,
                                scale=1)

    # Get the sine and cosine from the rotation matrix.
    abs_cos_angle = np.abs(M[0, 0])
    abs_sin_angle = np.abs(M[0, 1])

    # Compute the new bounding dimensions of the image.
    new_w = int(h * abs_sin_angle + w * abs_cos_angle)
    new_h = int(h * abs_cos_angle + w * abs_sin_angle)

    # Adjust the rotation matrix to take into account the translation.
    M[0, 2] += new_w // 2 - w // 2
    M[1, 2] += new_h // 2 - h // 2

    # Rotate the image.
    image = cv2.warpAffine(image, M=M, dsize=(new_w, new_h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(128, 128, 128))

    return image


def reorder_vertexes(vertexes):
    """
    reorder vertexes as the paper shows, (top, right, bottom, left)
    Args:
        vertexes: np.array (4, 2), should be in clockwise
    Returns:
    """
    assert vertexes.shape == (4, 2)
    xmin, ymin = np.min(vertexes, axis=0)
    xmax, ymax = np.max(vertexes, axis=0)

    # determine the first point with the smallest y,
    # if two vertexes has same y, choose that with smaller x,
    ordered_idxes = np.argsort(vertexes, axis=0)
    ymin1_idx = ordered_idxes[0, 1]
    ymin2_idx = ordered_idxes[1, 1]
    if vertexes[ymin1_idx, 1] == vertexes[ymin2_idx, 1]:
        if vertexes[ymin1_idx, 0] <= vertexes[ymin2_idx, 0]:
            first_vertex_idx = ymin1_idx
        else:
            first_vertex_idx = ymin2_idx
    else:
        first_vertex_idx = ymin1_idx
    ordered_idxes = [(first_vertex_idx + i) % 4 for i in range(4)]
    ordered_vertexes = vertexes[ordered_idxes]
    # drag the point to the corresponding edge
    ordered_vertexes[0, 1] = ymin
    ordered_vertexes[1, 0] = xmax
    ordered_vertexes[2, 1] = ymax
    ordered_vertexes[3, 0] = xmin
    return ordered_vertexes


def get_fm_shapes(image_shape, pyramid_levels):
    """
    Guess shapes based on pyramid levels.
    Args
         image_shape: The shape of the image.
         pyramid_levels: A list of what pyramid levels are used.
    Returns
        A list of image shapes at each pyramid level.
    """
    image_shape = np.array(image_shape[:2], dtype=np.int32)
    fm_shapes = np.array([(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels], dtype=np.int32)
    return fm_shapes


def post_process_boxes(boxes, scale, offset_h, offset_w, height, width):
    boxes[:, [0, 2]] = boxes[:, [0, 2]] - offset_w
    boxes[:, [1, 3]] = boxes[:, [1, 3]] - offset_h
    boxes /= scale
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height - 1)

    return boxes


def draw_boxes(image, boxes, scores, labels, colors, classes):
    for b, l, s in zip(boxes, labels, scores):
        class_id = int(l)
        class_name = classes[class_id]

        xmin, ymin, xmax, ymax = list(map(int, b))
        score = '{:.4f}'.format(s)
        color = colors[class_id]
        label = '-'.join([class_name, score])

        ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
        cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
        cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
