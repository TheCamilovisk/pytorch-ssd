import os
import sys
import xml.etree.ElementTree as ET
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
    '-i', '--input', type=str, required=True, help='The input testing video'
)
parser.add_argument(
    '-o', '--output', type=str, required=True, help="Ther path of the csv's folder"
)
parser.add_argument(
    '-a',
    '--annotations',
    type=str,
    required=True,
    help='The folder where the reference annotations are',
)

args = parser.parse_args()

annotations = args.annotations
if not os.path.exists(annotations):
    print('Invalid annotations folder!')
    sys.exit(-1)

if not os.path.isdir(args.output):
    os.makedirs(args.output)

class_names = ('BACKGROUND', 'drain')
class_dict = {class_name: i for i, class_name in enumerate(class_names)}


def get_annotation(image_id):
    annotation_file = os.path.join(annotations, f"{image_id}.xml")
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


cap = VideoFileStream(args.input, queue_sz=64, transforms=[])
cap.start()

video_fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_time = int(1 / cap.get(cv2.CAP_PROP_FPS) * 1e3)

roi_restriction = ROIRestriction((302, 273, 1579, 796))

video_name = os.path.basename(args.input).replace('.', '_')

timer = Timer()

# Detector
detector = WagonDetector(
    'mb1-ssd', 'resources/labels.txt', 'models/mobilenet_v6.pth', prob_threshold=0.4,
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
        video_fps=cap.get(cv2.CAP_PROP_FPS),
        target_fps=30.0,
    )
    return tracker


def create_system1():
    return PureDetectionTracker(detector,)


def create_system2():
    return DetectionAndTrackingTracker(detector, 'csrt', 5)


trackers_func = [create_system1, create_system2, create_proposed]


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


techniques_names = ['system1', 'system2', 'proposed']

stats = {}

cv2.namedWindow('annotated', cv2.WINDOW_NORMAL)

for tracker_f, technique in zip(trackers_func, techniques_names):
    tracker = tracker_f()

    cap = VideoFileStream(args.input, queue_sz=64, transforms=[])
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
        gt_boxes, _, _ = get_annotation(image_id)

        n_gt_elements_per_frame.append(len(gt_boxes))

        pred_boxes = np.array([box for box, _ in tracker(orig_image).values()]).astype(
            np.int
        )
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

        for idx, box in enumerate(gt_boxes):
            if idx in false_negatives:
                color = (0, 0, 255)
            else:
                color = (255, 255, 0)
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), color, 4)

        for idx, box in enumerate(pred_boxes):
            if idx in false_positives:
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), color, 4)

        cv2.imshow('annotated', orig_image)

        end_time = timer.end()
        wait_time = int(np.clip(frame_time - end_time, 1, frame_time))
        k = cv2.waitKey(wait_time) & 0xFF
        if k == ord('q') or k == 27:
            break

        frame_count += 1

    stats[technique] = {
        'gts': n_gt_elements_per_frame,
        'dets': n_detections_per_frame,
        'fps': n_false_positives_per_frame,
        'fns': n_false_negatives_per_frame,
        'iou_m': all_mean_ious_per_frame,
    }

cv2.destroyAllWindows()

for key, data in stats.items():
    df = pd.DataFrame(data, columns=list(data.keys()))
    csv_path = os.path.join(args.output, f'{key}.csv')
    df.to_csv(csv_path)
