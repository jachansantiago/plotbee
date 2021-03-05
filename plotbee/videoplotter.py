import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from plotbee.utils import id2color, rotate_bound2, trackevent2color

YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
MAGENTA = (255, 0, 255)


COLOR_BY_CONNECTION = {
    (1, 3) : BLUE, 
    (3, 2) : RED,
    (2, 4) : YELLOW,
    (2, 5) : YELLOW,
    (1, 2) : MAGENTA
}

COLOR_BY_PART = {
    '0' : BLUE,    #TAIL
    '1' : RED,     #HEAD
    '2' : MAGENTA, #ABDOMEN
    '3' : YELLOW,  #ANTENA
    '4' : YELLOW   #ANTENA
}

RADIUS = 10
THICKNESS = -1


def imshow(frame, **kwargs):
    plt.imshow(frame._image(**kwargs))

def bbox(frame, idtext=False, ax=None, suppression=False):
    if ax:
        ax.imshow(frame.bbox_image(idtext=idtext, suppression=suppression))
    else:
        plt.imshow(frame.bbox_image(idtext=idtext, suppression=suppression))

def skeleton(frame):
    plt.imshow(frame.skeleton_image)

def tracks(frame):
    plt.imshow(frame.track_image)

def parts(frame):
    plt.imshow(frame.parts_image)

def plot(frame, skeleton=False, bbox=False, tracks=False, min_parts=5):
    plt.imshow(frame._image(skeleton=skeleton,
                bbox=bbox, tracks=tracks,
                min_parts=min_parts))

def events(frame):
    plt.imshow(frame.event_image)


def bbox_drawer(frame, body, idtext=False):
    color = id2color(body.id)
    text = "id={}".format(body.id)
    font = cv2.FONT_HERSHEY_DUPLEX
    fontScale = 1.5
    p1, p2, p3, p4 = body.boundingBox()
        
    frame = cv2.line(frame, p1, p2, color=color, thickness=7)
    frame = cv2.line(frame, p2, p3, color=color, thickness=7)
    frame = cv2.line(frame, p3, p4, color=color, thickness=7)
    frame = cv2.line(frame, p4, p1, color=color, thickness=7)

    if idtext:
        cv2.putText(frame, text, p1, font, fontScale, color=color, thickness=3)

    return frame


def skeleton_drawer(frame, body, idtext=False):
    color = id2color(body.id)
    for p1, p2 in body.skeleton:
        frame = cv2.line(frame, p1, p2, color=color, thickness=7)
    return frame


def parts_drawer(frame, parts_dict):
    for part, points in parts_dict.items():
        color = COLOR_BY_PART[str(part - 1)]
        for point in points:
            p = tuple(point[:2])
            frame = cv2.circle(frame, p, RADIUS, color, THICKNESS)
    return frame


# def mapping_drawer(frame, mappings):



def extract_body(frame, body, width=200, height=400, cX=None, cY=None, ignore_angle=False):
    x, y = body.center
    
    if ignore_angle:
        angle = 0
    else:
        angle = body.angle

    return rotate_bound2(frame,x,y,angle, width, height, cX, cY)


def track_drawer(frame, body, thickness=3):
    points = list()
    color = id2color(body.id)
    x = body

    while x.next is not None:
        p = x.center
        points.append(np.int32(p))
        x = x.next

    points = np.array([points], dtype=np.int32)

    return cv2.polylines(frame, [points], False, color, thickness)

def event_track_drawer(frame, body, track, thickness=3):
    points = list()
    color = trackevent2color(track)
    if color is None:
        return frame
    x = body

    while x.next is not None:
        p = x.center
        points.append(np.int32(p))
        x = x.next

    points = np.array([points], dtype=np.int32)

    return cv2.polylines(frame, [points], False, color, thickness)


def track_images(track, figsize=(10, 20)):
    num_images = len(track)
    rows = (num_images // 10) + 1


    fig, ax = plt.subplots(nrows=rows, ncols=10, figsize=figsize)
    ax = ax.ravel()

    for i, body in enumerate(track):
        ax[i].imshow(body.image)


def tag_images(video, save_folder=None, black_listed_ids=[15, 16, 13]):
    tagged_bees = list()
    
    
    for frame in video:
        for body in frame:
            if body.tag_id is not None:
                if body.tag_id not in black_listed_ids:
                    tagged_bees.append(body)
                
    num_images = len(tagged_bees)
    rows = (num_images // 10) + 1
    figure_height = int(2.28 * rows) + 1

    fig, ax = plt.subplots(nrows=rows, ncols=10, figsize=(20, figure_height))

    ax = ax.ravel()

    for i, body in enumerate(tagged_bees):
        ax[i].imshow(body.image)
        ax[i].set_title(str(body.tag_id))
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    if save_folder is not None:
        _ , video_name = os.path.split(video.video_path)
        video_name, ext = os.path.splitext(video_name) 
        path = os.path.join(save_folder, video_name + ".pdf")
        plt.savefig(path)

def contact_sheet(bee_list, save_path=None, cols=10, tag=False):

    tag_image_folder = "/home/jchan/tags_dataset/tag25h5inv/png/"
    num_images = len(bee_list)
    rows = (num_images // cols)
    figure_height = int(2.5 * rows) + 1


    if tag:
        rows *= 2
        figure_height *= 2.0
        fig, ax = plt.subplots(nrows=rows, ncols=cols + 1, figsize=(15, figure_height))

        tag_ax = ax[::2, ...].ravel()
        body_ax = ax[1::2, ...].ravel()
        
        j = 0
        for i, (tax, bax) in enumerate(zip(tag_ax, body_ax)):
            
            if j == len(bee_list):
                break
            body = bee_list[j]

            if i % (cols + 1) == 0:
                tag_path = os.path.join(tag_image_folder, "keyed{:04}.png".format(body.tag_id))
                tag_image = cv2.imread(tag_path)
                tax.imshow(tag_image)
                tax.set_ylabel(str(body.tag_id) + "       ", rotation='horizontal')
                bax.set_visible(False)
            else:
                tax.imshow(body.tag_image())
                tax.set_xlabel(str(body.tag["hamming"]))
                tax.set_title("{0:.2f}".format(body.tag["dm"]))

                bax.imshow(body.image)
                bax.set_xlabel(str(body.frameid))
                vname = body.video_name
                vname, ext = os.path.splitext(vname)
                bax.set_ylabel(vname)
                j += 1
            tax.set_xticks([])
            tax.set_yticks([])
            bax.set_xticks([])
            bax.set_yticks([])
    else:

        fig, ax = plt.subplots(nrows=rows, ncols=cols + 1, figsize=(15, figure_height))

        axes = ax.ravel()
        j = 0
        for i, ax in enumerate(axes):
            
            if j == len(bee_list):
                break
            body = bee_list[j]

            if i % (cols + 1) == 0:
                tag_path = os.path.join(tag_image_folder, "keyed{:04}.png".format(body.tag_id))
                tag_image = cv2.imread(tag_path)
                ax.imshow(tag_image)
                ax.set_ylabel(str(body.tag_id) + "       ", rotation='horizontal')
            else:
                ax.imshow(body.tag_image())
                ax.set_xlabel(str(body.tag["hamming"]))
                ax.set_title("{0:.2f}".format(body.tag["dm"]))
                j += 1
            ax.set_xticks([])
            ax.set_yticks([])

    plt.subplots_adjust(hspace=0.4)
        

    if save_path is not None:
        path = os.path.join(save_path)
        plt.savefig(path, bbox_inches='tight')

    
