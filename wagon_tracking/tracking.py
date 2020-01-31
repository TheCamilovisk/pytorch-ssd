import vision.utils.box_utils_numpy as box_utils
import numpy as np

# from collections import namedtuple
from vision.utils.misc import Timer
from copy import deepcopy


# WagonInfo = namedtuple('WagonInfo', 'idx', 'front', 'back')


class WagonTracker:
    def __init__(self, detector):
        self.detector = detector
        self.drains_info = None
        self.timer = Timer()
        # self.tracked_wagons = []
        self.next_id = 0
        self.movement_vector = np.array([0.0, 0.0])

    def __call__(self, image):
        self.timer.start()
        boxes, labels, probs = self.detector(image)
        interval = self.timer.end()

        print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))

        self._update_tracking(boxes.numpy(), labels.numpy())
        return deepcopy(self.drains_info)

    def _update_tracking(self, boxes, labels):
        if self.drains_info is None and len(boxes) > 0:
            self._init_tracking_dict(boxes, labels)
            return

        updated_drains_info = self._match_boxes(boxes, labels)

        updated_drains_info = self._update_notfound_objs(updated_drains_info)

        self.drains_info = updated_drains_info

    def _init_tracking_dict(self, boxes, labels):
        self.next_id = len(boxes)
        ids = list(range(self.next_id))
        self.drains_info = list(zip(ids, boxes, labels))

    def _match_boxes(self, boxes, labels):
        updated_drains_info = []
        movement_vector = np.array([0.0, 0.0])

        for t_id, t_box, t_lbl in self.drains_info:
            if len(boxes) == 0:
                break

            ious = box_utils.iou_of(t_box, boxes)
            n_box_idx = np.argmax(ious)

            if ious[n_box_idx] > 0.0:
                u_box = boxes[n_box_idx]

                u_center = (u_box[2:] + u_box[:2]) / 2
                t_center = (t_box[2:] + t_box[:2]) / 2
                movement_vector += u_center - t_center

                updated_drains_info.append([t_id, u_box, labels[n_box_idx]])

                boxes = np.delete(boxes, (n_box_idx), axis=0)
                labels = np.delete(labels, (n_box_idx), axis=0)

        if len(updated_drains_info) > 0:
            self.movement_vector = movement_vector / len(updated_drains_info)
        else:
            # It's a hack. At the left end of the frame the object bounding boxes are
            # prone to slightly move their center up. We don't want this o happen.
            self.movement_vector[1] = 0.0

        if len(boxes) > 0:
            new_ids = list(range(self.next_id, self.next_id + 1 + len(boxes)))
            updated_drains_info.extend(list(zip(new_ids, boxes, labels)))
            self.next_id = new_ids[-1]

        return updated_drains_info

    def _update_notfound_objs(self, updated_drains_info: list):
        u_ids = [id for id, _, _ in updated_drains_info]

        not_found_info = []
        for t_id, t_box, t_lbl in self.drains_info:
            if np.linalg.norm(self.movement_vector) > 5:
                t_box[2:] += self.movement_vector
                t_box[:2] += self.movement_vector

            if t_id not in u_ids:
                not_found_info.append([t_id, t_box, t_lbl])

        updated_drains_info.extend(not_found_info)
        return updated_drains_info
