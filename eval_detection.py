import logging
import os
import sys
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from glob import glob

import cv2
import numpy as np
import pandas as pd

from vision.utils import box_utils_numpy as box_utils
from vision.utils import measurements
from wagon_tracking.detection import WagonDetector


def get_annotation(image_id, annos_folder, class_dict):
    annotation_file = os.path.join(annos_folder, f"{image_id}.xml")
    objects = ET.parse(annotation_file).findall("object")
    boxes = []
    labels = []
    is_difficult = []
    for object in objects:
        class_name = object.find('name').text.lower().strip()
        # we're only concerned with clases in our list
        if class_name in class_dict:
            bbox = object.find('bndbox')

            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])

            labels.append(class_dict[class_name])
            is_difficult_str = object.find('difficult').text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

    return (
        np.array(boxes, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(is_difficult, dtype=np.uint8),
    )


def get_data(image_id, imgs_folder, annos_folder, class_dict):
    img_path = os.path.join(imgs_folder, f'{image_id}.jpg')
    boxes, _, _ = get_annotation(image_id, annos_folder, class_dict)
    return cv2.imread(img_path), boxes


def read_classes(label_filepath):
    class_string = ""
    with open(label_filepath, 'r') as infile:
        class_string = ','.join((line.rstrip() for line in infile))

    # classes should be a comma/newline separated list
    classes = class_string.split(',')
    # prepend BACKGROUND as first class
    classes.insert(0, 'BACKGROUND')
    classes = [elem.replace(" ", "") for elem in classes]
    class_names = tuple(classes)
    class_dict = {class_name: i for i, class_name in enumerate(class_names)}
    return class_names, class_dict


def read_datafile(datafile):
    with open(datafile) as f:
        images_ids = set((l.strip() for l in f.readlines()))

    return list(images_ids)


def compute_metrics(dataset_folder, dataset_file, detector, class_dict, iou_threshold):
    imgs_folder = os.path.join(dataset_folder, 'JPEGImages')
    annos_folder = os.path.join(dataset_folder, 'Annotations')

    gt_boxes = []
    gt_img_ids = []
    pred_boxes = []
    scores = []
    pred_img_ids = []

    data_gen = (
        get_data(image_id, imgs_folder, annos_folder, class_dict)
        for image_id in read_datafile(dataset_file)
    )

    for id, (img, img_gt_boxes) in enumerate(data_gen):
        gt_boxes.extend((box for box in img_gt_boxes))
        gt_img_ids.extend((id for _ in img_gt_boxes))

        boxes, labels, probs = detector(img)
        boxes, labels, probs = boxes.numpy(), labels.numpy(), probs.numpy()

        pred_boxes.extend((box for box in boxes))
        scores.extend((prob for prob in probs))
        pred_img_ids.extend((id for _ in probs))

    gt_boxes = np.array(gt_boxes)
    gt_img_ids = np.array(gt_img_ids)

    scores = np.array(scores)
    sorted_idxs = np.argsort(-scores)
    pred_boxes = np.array([pred_boxes[i] for i in sorted_idxs])
    pred_img_ids = np.array([pred_img_ids[i] for i in sorted_idxs])

    true_positives = np.zeros(len(pred_img_ids))
    false_positives = np.zeros(len(pred_img_ids))
    matched = set()
    for i, (p_box, p_id) in enumerate(zip(pred_boxes, pred_img_ids)):
        if p_id not in gt_img_ids:
            false_positives[i] = 1
            continue

        g_boxes = gt_boxes[gt_img_ids == p_id, :]
        ious = box_utils.iou_of(p_box, g_boxes)
        max_iou = ious.max()
        max_arg = ious.argmax()

        if max_iou > iou_threshold:
            true_positives[i] = 1
            matched.add((p_id, max_arg))
        else:
            false_positives[i] = 1

    true_positives = np.cumsum(true_positives)
    false_positives = np.cumsum(false_positives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / len(gt_boxes)
    ap = measurements.compute_voc2007_average_precision(precision, recall)
    false_negatives = len(gt_boxes) - true_positives[-1]
    f1score = 2 * (precision[-1] * recall[-1]) / (precision[-1] + recall[-1])

    return (
        true_positives[-1],
        false_positives[-1],
        false_negatives,
        precision[-1],
        recall[-1],
        f1score,
        ap,
    )


def compute_folds_metrics(
    fold_i,
    net_type,
    label_filepath,
    models_folder,
    crossvalidation_sets_folder,
    dataset_folder,
    iou_threshold,
):

    train_true_positives = 0
    train_false_positives = 0
    train_false_negatives = 0
    train_precision = 0
    train_recall = 0
    train_f1score = 0
    train_ap = 0

    test_true_positives = 0
    test_false_positives = 0
    test_false_negatives = 0
    test_precision = 0
    test_recall = 0
    test_f1score = 0
    test_ap = 0

    class_names, class_dict = read_classes(label_filepath)

    train_file = os.path.join(
        crossvalidation_sets_folder, f'fold{fold_i}', 'trainval.txt'
    )
    test_file = os.path.join(crossvalidation_sets_folder, f'fold{fold_i}', 'test.txt')
    logging.info(f'Train set from: {train_file}')
    logging.info(f'Test set from: {test_file}')

    models_paths = glob(os.path.join(models_folder, f'fold{fold_i}', '*'))
    for model_path in models_paths:
        logging.info(f'Loading model from {model_path}...')
        detector = WagonDetector(net_type, label_filepath, model_path)
        logging.info('Model loaded!')

        logging.info('Computing train metrics...')
        (
            true_positives,
            false_positives,
            false_negatives,
            precision,
            recall,
            f1score,
            ap,
        ) = compute_metrics(
            dataset_folder, train_file, detector, class_dict, iou_threshold
        )
        train_true_positives += true_positives
        train_false_positives += false_positives
        train_false_negatives += false_negatives
        train_precision += precision
        train_recall += recall
        train_f1score += f1score
        train_ap += ap
        logging.info('Train metrics computed!')

        logging.info('Computing test metrics...')
        (
            true_positives,
            false_positives,
            false_negatives,
            precision,
            recall,
            f1score,
            ap,
        ) = compute_metrics(
            dataset_folder, test_file, detector, class_dict, iou_threshold
        )
        test_true_positives += true_positives
        test_false_positives += false_positives
        test_false_negatives += false_negatives
        test_precision += precision
        test_recall += recall
        test_f1score += f1score
        test_ap += ap
        logging.info('Test metrics computed!')

    train_data = pd.Series(
        [
            train_true_positives / len(models_paths),
            train_false_positives / len(models_paths),
            train_false_negatives / len(models_paths),
            train_precision / len(models_paths),
            train_recall / len(models_paths),
            train_f1score / len(models_paths),
            train_ap / len(models_paths),
        ],
        index=['tps', 'fps', 'fns', 'precision', 'recall', 'f1score', 'ap'],
    )
    test_data = pd.Series(
        [
            test_true_positives / len(models_paths),
            test_false_positives / len(models_paths),
            test_false_negatives / len(models_paths),
            test_precision / len(models_paths),
            test_recall / len(models_paths),
            test_f1score / len(models_paths),
            test_ap / len(models_paths),
        ],
        index=['tps', 'fps', 'fns', 'precision', 'recall', 'f1score', 'ap'],
    )

    return train_data, test_data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--dataset-folder',
        type=str,
        required=True,
        help='Folder with the JPEGImages and Annotations' ' folders.',
    )
    parser.add_argument(
        '--crossvalidation-sets-folder',
        type=str,
        required=True,
        help='Folder with the folders' ' containing the folds sets.',
    )
    parser.add_argument(
        '--output-path', type=str, required=True, help='The output folder.'
    )
    parser.add_argument(
        '--net-type', type=str, default='mb1-ssd', help='The detector network type.'
    )
    parser.add_argument(
        '--label-filepath',
        type=str,
        default='resources/labels.txt',
        help='Path to the labels file.',
    )
    parser.add_argument(
        '--models-folder',
        type=str,
        default='models/folds',
        help='Folder with the folders containing' ' the folds models.',
    )
    parser.add_argument(
        '--iou-threshold', type=float, default=0.5, help='The detection iou threshold.'
    )
    args = parser.parse_args()

    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    logging.info(f'Network type: {args.net_type}')
    logging.info(f'Labels from: {args.label_filepath}')
    logging.info(f'Models folder: {args.models_folder}')
    logging.info(f'Cross-validation sets folder: {args.crossvalidation_sets_folder}')
    logging.info(f'Dataset folder: {args.dataset_folder}')

    train_df = pd.DataFrame(
        columns=['tps', 'fps', 'fns', 'precision', 'recall', 'f1score', 'ap']
    )
    test_df = pd.DataFrame(
        columns=['tps', 'fps', 'fns', 'precision', 'recall', 'f1score', 'ap']
    )

    for fold_i in range(5):
        logging.info(f'Processing fold {fold_i} started!')

        fold_train_df, fold_test_df = compute_folds_metrics(
            fold_i,
            args.net_type,
            args.label_filepath,
            args.models_folder,
            args.crossvalidation_sets_folder,
            args.dataset_folder,
            args.iou_threshold,
        )
        logging.info(f'Processing fold {fold_i} finished!')

        train_df = train_df.append(fold_train_df, ignore_index=True)
        test_df = test_df.append(fold_test_df, ignore_index=True)

    train_csv_path = os.path.join(args.output_path, 'train_eval.csv')
    train_df.to_csv(train_csv_path)
    logging.info(f'Train evaluation data saved to {train_csv_path}')

    test_csv_path = os.path.join(args.output_path, 'test_eval.csv')
    test_df.to_csv(test_csv_path)
    logging.info(f'Test evaluation data saved to {test_csv_path}')
