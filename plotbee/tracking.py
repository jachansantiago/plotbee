from scipy.optimize import linear_sum_assignment as hungarian
import numpy as np
from plotbee.sort import Sort, KalmanBoxTracker
from plotbee.track import Track
from collections import defaultdict
from tqdm import tqdm


class IntegerIDGen():

    def __init__(self):
        self.next_available_id = 0

    def __call__(self):
        i = self.next_available_id
        self.next_available_id += 1
        return i

def body_distance(body_a, body_b):
    x_a, y_a = body_a.center
    x_b, y_b = body_b.center

    return np.sqrt((x_a - x_b)**2 + (y_a - y_b)**2)

def cost_matrix_tracks_skeleton(frame, next_frame, threshold):
    total = len(frame)+len(next_frame)
    cost_m = np.zeros((total,total))
    for i in range(total):
        for j in range(total):
            if i < len(frame) and j <len(next_frame):
                    cost_m[i][j] = body_distance(frame[i], next_frame[j])
            else:
                cost_m[i][j] = threshold

    return cost_m

def body_cbox_overlaping_ratio(body_a, body_b):
    x1_i, y1_i, x2_i, y2_i = body_a.cbox()
    x1_j, y1_j, x2_j, y2_j = body_b.cbox()

    area_i = (x2_i - x1_i + 1) * (y2_i - y1_i + 1)

    intersec_x1 = max(x1_i, x1_j)
    intersec_y1 = max(y1_i, y1_j)

    intersec_x2 = min(x2_i, x2_j)
    intersec_y2 = min(y2_i, y2_j)

    w = max(0, intersec_x2 - intersec_x1 + 1)
    h = max(0, intersec_y2 - intersec_y1 + 1)

    overlap_ratio = float(w*h)/area_i

    return overlap_ratio

def non_max_supression_video(video, overlapThreshold):
    for frame in video:
        non_max_supression(frame, overlapThreshold)
    return




def non_max_supression(frame, overlapThreshold):
    """
    This function filter out overlaping bodies using a Overlapping threshold.

    """

    # Sort bees by y2
#     sorted_bodies = sorted(frame, key=lambda body: body.cbox()[-1])
    
    num_bodies = len(frame)
    
    for i in range(num_bodies):
        body_a = frame[i]
        
        if body_a.suppressed:
            continue
            
        for j in range(i + 1, num_bodies):
            
            body_b = frame[j]
            
            if body_b.suppressed:
                continue
                
            overlap_ratio = body_cbox_overlaping_ratio(body_a, body_b)

            if overlap_ratio > overlapThreshold:
                body_b.suppressed = True
        
    return




def hungarian_tracking(video, cost=200, nms_overlap_fraction=0.6):


    getId = IntegerIDGen()
    # Supress bodies
    non_max_supression_video(video, nms_overlap_fraction)


    for i, body in enumerate(video[0].valid_bodies):
        body.set_id(getId())
        video._tracks[body.id] = Track(body)
#             print(body)


    for i in tqdm(range(len(video) - 1)):
        current_frame = video[i].valid_bodies
        next_frame = video[i + 1].valid_bodies

        cmap = cost_matrix_tracks_skeleton(current_frame, next_frame, cost)
        _, idx = hungarian(cmap)

        for j in range(len(current_frame)):
            if cmap[j,idx[j]]<cost:

                # Create New ID
                if current_frame[j].id == -1:

                    current_frame[j].set_id(getId())
                    video._tracks[current_frame[j].id] = Track(current_frame[j])

                # Match Next Frame Detections
                next_frame[idx[j]].set_id(current_frame[j].id)
                next_frame[idx[j]].prev = current_frame[j]
                current_frame[j].next = next_frame[idx[j]]
    return


def matchIds(bboxes, predbboxes):
    a = (bboxes[:, 0:2] + bboxes[:, 2:4])/2
    b = (predbboxes[:, 0:2] + predbboxes[:, 2:4])/2

    m = np.zeros((bboxes.shape[0], predbboxes.shape[0]))

    for i, p1 in enumerate(a):
        for j, p2 in enumerate(b):
            d = np.sqrt(np.sum((p1 - p2)**2))
            m[i, j] = d
            
    bbox_ids, pred_ids = hungarian(m)
    # print(predbboxes.shape, pred_ids.shape)
    return bbox_ids, predbboxes[pred_ids, 4]


def sort_tracking(video, bbox=200, nms_overlap_fraction=0.6):
    # getId = IntegerIDGen()
    # Supress bodies
    non_max_supression_video(video, nms_overlap_fraction)
    mot_tracker = Sort()
    KalmanBoxTracker.count=0
    prev_track = defaultdict(lambda: None)

    for frame in video:
        valid_bodies = [body for body in frame if not body.suppressed]
        bboxes = [body.cbox(bbox, bbox) for body in valid_bodies]
        bboxes = np.array(bboxes)

        predbboxes = mot_tracker.update(bboxes)
        # if(predbboxes.shape[0] != bboxes.shape[0]):
        #     print(predbboxes.shape, bboxes.shape)

        bodiesIds, predIds = matchIds(bboxes, predbboxes) 

        for i, body in zip(predIds, bodiesIds):
            body = valid_bodies[body]
            body.set_id(int(i))
            # print(box_id)
            

            # Update Track LinkList DataStructure
            body.prev = prev_track[body.id]

            if prev_track[body.id] is not None:
                prev_track[body.id].next = body
            else:
                video._tracks[body.id] = Track(body)


            prev_track[body.id] = body

    return