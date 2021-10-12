from plotbee.videoplotter import bodies_bbox_drawer, bodies_skeleton_drawer, bodies_track_drawer
from plotbee.videoplotter import parts_drawer, bodies_event_track_drawer, bbox_drawer
from plotbee.videoplotter import extract_body, skeleton_drawer, event_track_drawer, track_drawer
from plotbee.utils import rotate_bound2
from plotbee.utils import pointInRotatedBbox
import os
from skimage import io
from functools import lru_cache
from plotbee.body import Body
import numpy as np
import warnings
import cv2


class Frame():

    TRACK_DIRECTION = "forward"
    
    def __init__(self, bodies, frame_id, image=None, mapping=None, parts=None):
        self._id = int(frame_id)
        # self._frame = frame
        # self._tracks = tracks
        self._video = None
        # self._parts = parts
        self._bodies = bodies

        self._cached_image = image
        # self._mappings = get_mappings_by_limb(mappings)
        # self._bodies =  mapping_to_body(self, self._mappings,
        #                                 self._tracks, id_tracks, self._id)

        self._mapping = mapping
        self._parts = parts
        
    def set_video(self, video):
        self._video = video

    def get_track(self, body):
        bid = body.id
        track = self._video.tracks[bid]
        return track

    @property
    def id(self):
        return self._id


#     @property
#     def parts_image(self):
#         for body in self._bodies:
#             for part, points in body._parts.items():
#                 color = self.COLOR_BY_PART[part]
#                 for point in points:
#                     p = tuple(point[:2])
#                     frame = cv2.circle(frame, p, radius, color, thickness)
#         return frame

    # @property
    # def height(self):
    #     return self._frame.shape[0]
    
    # @property
    # def width(self):
    #     return self._frame.shape[1]

    # @property
    # def shape(self):
    #     return self._frame.shape

    # @property
    # def frame(self):
    #     return self._frame

    @property
    def bodies(self):
        return self._bodies

    @property
    def valid_bodies(self):
        valid = []
        for body in self._bodies:
            if not body.suppressed:
                valid.append(body)
        return valid
    
    def delete_virtual_bodies(self):
        self._bodies = [b for b in self._bodies if (not b.virtual)]

    def update(self, bodies):
        self._bodies += bodies


    # @property
    # def parts(self):
    #     return self._parts

    @property
    def video_name(self):
        return self._video.video_name

    def _image(self, **kwargs):
        frame_image = self.image.copy()
        frame_image = self.draw_frame_image(frame_image, **kwargs)
        return frame_image

    def draw_frame_image(self, frame_image, skeleton=False, bbox=False, tracks=False, events=False, min_parts=5, track_direction="forward", idtext=False, fontScale=1.5, fontThickness=3, thickness=7):
        filtered_bodies = [body for body in  self.bodies if len(body) >= min_parts]

        if bbox:
            frame_image = bodies_bbox_drawer(frame_image, filtered_bodies, idtext=idtext, fontScale=fontScale, fontThickness=fontThickness, linethick=thickness)

        if skeleton:
            frame_image = bodies_skeleton_drawer(frame_image, filtered_bodies, thickness=thickness)

        if tracks:
            frame_image = bodies_track_drawer(frame_image, filtered_bodies, direction=track_direction)

        if events:
            tracked_bodies = [body for body in filtered_bodies if body.id != -1]
            frame_image = bodies_event_track_drawer(frame_image, tracked_bodies)

        return frame_image

    def bbox_image(self, idtext, suppression=False):

        frame = self.image.copy()

        for body in self.bodies:
            if suppression and body.suppressed:
                continue
            frame = bbox_drawer(frame, body, idtext=idtext)

        return frame

    @property
    def skeleton_image(self):

        frame = self.image.copy()

        for body in self.bodies:
            frame = skeleton_drawer(frame, body)
        
        return frame


    def track_image(self, direction=None):

        if direction is None:
            direction = Frame.TRACK_DIRECTION

        frame = self.image.copy()

        for body in self.bodies:
            frame = track_drawer(frame, body, direction=direction)
        
        return frame

    @property
    def parts_image(self):

        frame = self.image.copy()
        for body in self._bodies:
            frame = parts_drawer(frame, body._parts)

        return frame

    @property
    def event_image(self):
        frame = self.image.copy()

        for body in self.bodies:
            if body.id == -1:
                continue
            btrack = self.get_track(body)
            frame = event_track_drawer(frame, body, btrack)
        return frame



    @property
    # @lru_cache(maxsize=1000)
    def image(self):
        # if self._cached_image is None:
        #     self._cached_image = self._video.frame_image(self.id)
        # return self._cached_image.copy()
        return self._video.frame_image(self.id)


    def extract_patch(self, x, y, angle=0, width=160, height=320, cX=None, cY=None):
        return rotate_bound2(self.image, x, y, angle, width, height, cX, cY)
    
    @staticmethod
    def _extract_bodies_images(image, frame_data, width=None, height=None, cX=None, cY=None, scale=None, suppression=False, min_parts=-1):

        if width is None:
            width = Body.width
        if height is None:
            height = Body.height
        if cX is None:
            cX = Body.cX
        if cY is None:
            cY = Body.cY

        if scale is None:
            scale = Body.scale
        
        images =list()
        bodies = list()


        for body in frame_data:
            if suppression and not body.valid:
                continue

            if len(body) < min_parts:
                continue
            cbodyimg = extract_body(image, body, width=width, 
                                    height=height, cX=cX, cY=cY, scale=scale)

            if Body.out_width is not None and Body.out_height is not None:
                cbodyimg = cv2.resize(cbodyimg, (Body.out_height, Body.out_width))
            images.append(cbodyimg)
            bodies.append(body)
        return bodies, np.array(images)
        


    def bodies_images(self, width=None, height=None, cX=None, cY=None, suppression=False, min_parts=-1):
        return self._extract_bodies_images(self.image, self)

    
    
    def __repr__(self):
        frepr = "Frame: {}".format(self.id)
        brepr = [repr(b) for b in self._bodies]
        repr_list = [frepr] + brepr
        return "\n".join(repr_list)
    
    def __len__(self):
        return len(self.bodies)

    def __getitem__(self, index):
        return self.bodies[index]

    def bodies_at_point(self, p, silent=False):
        bodies = list()

        for body in self:
            if pointInRotatedBbox(p, body.center, body.angle, body.width, body.height):
                bodies.append(body)

        if not silent:
            if len(bodies) > 1:
                warnings.warn("More than one body in {} point.".format(p))


        return bodies  

        
    def save(self, folder, skeleton=True, bbox=True, tracks=False, events=True, min_parts=-1,
             idtext=False, fontScale=2.5, fontThickness=8):
        file_format = "{:09d}.jpg"
        os.makedirs(folder, exist_ok=True)
        im = self._image(skeleton=skeleton, bbox=bbox, tracks=tracks, events=events, min_parts=min_parts, idtext=idtext, fontScale=fontScale, fontThickness=fontThickness)
        
        fname = file_format.format(self.id)
        im_path = os.path.join(folder, fname)
        io.imsave(im_path, im)
        
