import logging
import os
import sys
import xml.etree.ElementTree as ET
from glob import iglob
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd

from vision.utils import box_utils_numpy as box_utils
from vision.utils.misc import Timer
from wagon_tracking.detection import WagonDetector
from wagon_tracking.restrictions import (
    DetectionDistanceRestriction,
    ROIRestriction,
    TrajectoryProfileRestriction,
)
from wagon_tracking.tracking import (
    DetectionAndTrackingTracker,
    PureDetectionTracker,
    WagonTracker,
)
from wagon_tracking.videostream import VideoFileStream

parser = ArgumentParser()
parser.add_argument(
    '-if',
    '--input-folder',
    type=str,
    required=True,
    help='The folder path where all the test videos are located.',
)
parser.add_argument(
    '-af',
    '--annotations-folder',
    type=str,
    required=True,
    help='The folder path where all the annotations of the test video are located',
)
parser.add_argument(
    '-of',
    '--output-folder',
    type=str,
    required=True,
    help='The folder where the resulting csvs will be stored.',
)
parser.add_argument(
    '-v',
    '--visual',
    action='store_true',
    help='Enables the visual feedback of the tracking.',
)
args = parser.parse_args()

input_folder = args.input_folder
assert os.path.isdir(input_folder)


annotations_folder = args.annotations_folder
assert os.path.isdir(annotations_folder)


output_folder = args.output_folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
else:
    assert os.path.isdir(output_folder)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logging.info(f'Input folder: {input_folder}')
logging.info(f'Annotations folder: {annotations_folder}')
logging.info(f'Output folder: {output_folder}')


class_names = ('BACKGROUND', 'drain')
class_dict = {class_name: i for i, class_name in enumerate(class_names)}

logging.info(f'Classes: {class_names}')


def get_annotation(image_id, video_name):
    annotation_file = os.path.join(annotations_folder, video_name, f"{image_id}.xml")
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


roi_restriction = ROIRestriction((302, 273, 1579, 796))

timer = Timer()

net_model = 'models/mobilenet_v6.pth'
logging.info(f'Net model from: {net_model}')
detector = WagonDetector(
    'mb1-ssd', 'resources/labels.txt', net_model, prob_threshold=0.4,
)


def create_proposed():
    restrictions = [
        ROIRestriction((302, 273, 1579, 796)),
        TrajectoryProfileRestriction(
            (0, 0, frame_width, frame_height),
            (0, frame_height // 2),
            distance_threshold=20,
        ),
        DetectionDistanceRestriction((2.5, 4.8), (0.5, 1.5)),
    ]
    tracker = WagonTracker(
        detector,
        frame_width // 2,
        restrictions=restrictions,
        video_fps=10.0,
        target_fps=30.0,
    )
    return tracker


def create_system1():
    return PureDetectionTracker(detector,)


def create_system2():
    return DetectionAndTrackingTracker(detector, 'csrt', 5)


trackers_func = [create_system1, create_system2, create_proposed]
techniques_names = ['system1', 'system2', 'proposed']


def compare_boxes(gt_boxes, pred_boxes):
    if len(pred_boxes) == 0:
        return [], [], list(range(len(gt_boxes))), []
    pred_boxes = pred_boxes.copy()

    detections_hits = []
    detections_ious = []
    false_negatives = []

    for gt_idx, gt_box in enumerate(gt_boxes):
        ious = box_utils.iou_of(gt_box, pred_boxes)
        pred_idx = np.argmax(ious)

        if ious[pred_idx] > 0.5:
            detections_hits.append(pred_idx)
            detections_ious.append(ious[pred_idx])
        else:
            false_negatives.append(gt_idx)

    false_positives = [
        pred_idx
        for pred_idx, _ in enumerate(pred_boxes)
        if pred_idx not in detections_hits
    ]

    return detections_hits, detections_ious, false_negatives, false_positives


stats = {}

extensions = ['avi', 'mkv']

quit = False

logging.info(f'Visual feedback enabled: {args.visual}')

if args.visual:
    cv2.namedWindow('annotated', cv2.WINDOW_NORMAL)

for path in iglob(os.path.join(input_folder, '*')):
    if (
        not os.path.isfile(path)
        or os.path.splitext(path)[1][1:].lower() not in extensions
    ):
        continue

    logging.info(f'Processing video: {path}')

    video_name = os.path.basename(path).replace('.', '_')

    cap = VideoFileStream(path, queue_sz=64, transforms=[])
    cap.start()

    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_time = int(1 / cap.get(cv2.CAP_PROP_FPS) * 1e3)

    for tracker_f, technique in zip(trackers_func, techniques_names):
        tracker = tracker_f()
        logging.info(f'Technique pass: {technique}')

        cap = VideoFileStream(path, queue_sz=64, transforms=[])
        cap.start()

        n_gt_elements_per_frame = []
        n_detections_per_frame = []
        n_false_positives_per_frame = []
        n_false_negatives_per_frame = []
        all_mean_ious_per_frame = []

        frame_count = 0
        while cap.more():
            timer.start()
            orig_image = cap.read()
            if orig_image is None:
                continue

            image_id = video_name + f'_{frame_count}'
            gt_boxes, _, _ = get_annotation(image_id, video_name)

            n_gt_elements_per_frame.append(len(gt_boxes))

            pred_boxes = np.array(
                [box for box, _ in tracker(orig_image).values()]
            ).astype(np.int)
            pred_boxes, _ = roi_restriction(pred_boxes)

            (
                detections_hits,
                detections_ious,
                false_negatives,
                false_positives,
            ) = compare_boxes(gt_boxes, pred_boxes)

            n_detections_per_frame.append(len(detections_hits))
            n_false_positives_per_frame.append(len(false_positives))
            n_false_negatives_per_frame.append(len(false_negatives))
            all_mean_ious_per_frame.append(np.mean(detections_ious))

            if args.visual:
                for idx, box in enumerate(gt_boxes):
                    if idx in false_negatives:
                        color = (0, 0, 255)
                    else:
                        color = (255, 255, 0)
                    cv2.rectangle(
                        orig_image, (box[0], box[1]), (box[2], box[3]), color, 4
                    )

                for idx, box in enumerate(pred_boxes):
                    if idx in false_positives:
                        color = (0, 255, 255)
                    else:
                        color = (0, 255, 0)
                    cv2.rectangle(
                        orig_image, (box[0], box[1]), (box[2], box[3]), color, 4
                    )

                cv2.imshow('annotated', orig_image)

                end_time = timer.end()
                wait_time = int(np.clip((frame_time - end_time) / 4, 1, frame_time))
                k = cv2.waitKey(wait_time) & 0xFF
                if k == ord('q') or k == 27:
                    quit = True
                    break

            frame_count += 1

        if quit:
            break

        n_gts = int(np.sum(n_gt_elements_per_frame))
        n_tps = int(np.sum(n_detections_per_frame))
        n_fps = int(np.sum(n_false_positives_per_frame))
        n_fns = int(np.sum(n_false_negatives_per_frame))
        precision = n_tps / (n_tps + n_fps)
        recall = n_tps / (n_tps + n_fns)
        miss_rate = 1 - recall
        f1_score = 2 * n_tps / (2 * n_tps + n_fps + n_fns)

        stats[technique] = [
            n_tps,
            n_fps,
            n_fns,
            precision,
            recall,
            miss_rate,
            f1_score,
            n_gts,
        ]

        logging.info('Pass finished!')

    if quit:
        break

    df = pd.DataFrame(
        stats.values(),
        columns=[
            'TP',
            'FP',
            'FN',
            'Precision',
            'Recall',
            'Miss Rate',
            'F1 score',
            'Ground Truths',
        ],
    )

    csv_path = os.path.join(output_folder, f'{video_name}.csv')
    df.T.astype(object).to_csv(csv_path)
    logging.info(f'Processing results saved to: {csv_path}')

    del cap

cv2.destroyAllWindows()
