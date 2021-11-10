from generators.voc import PascalVocGenerator
import tensorflow as tf
from models import sapd
from utils.eval import evaluate2
import numpy as np
import config


def main(model_weight_path):
    generator = PascalVocGenerator(
        config.DATABASE_PATH,
        'test',
        skip_difficult=True,
        shuffle_groups=False,
        batch_size=1,
        phi=config.PHI
    )

    test, pred_model = sapd.SAPD(
        soft=True,
        num_cls=config.NUM_CLS,
        resnet=config.BACKBONE,
    )
    test.load_weights(model_weight_path, by_name=True, skip_mismatch=False)

    ap = evaluate2(generator, pred_model, score_threshold=0.01, max_detections=100)

    aps = []
    iou_range = np.arange(0.5, 1., 0.05)

    print(f"[INFO] Model Name:{model_weight_path}")

    for iou_index, labels_dic in ap.items():
        iou = np.round(iou_range[iou_index], 2)
        print(f"[INFO] mAP @ {iou} : ")

        total_instances = []
        precisions = []

        for label, (average_precision, num_annotations, recall, precision) in labels_dic.items():
            print(' - {:.0f} instances of class'.format(num_annotations), generator.label_to_name(label),
                  'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)

        mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)
        aps.append(mean_ap)
        print(' - mAP: {:.4f}'.format(mean_ap))

    aps = np.mean(aps)
    aps = np.round(aps, 4)
    print(f"[INFO] mAP[0.5:0.95]: {aps}")


def init_():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


if __name__ == '__main__':
    init_()
    main(
        model_weight_path='20210921-DPCB100-HA116FV3-SGDW-E100BS8B1R50D4-soft.h5'
    )
