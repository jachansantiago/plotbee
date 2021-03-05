from plotbee.utils import angleBetweenPoints
from plotbee.utils import rotatedBoundBoxPoints, getRotationMatrix
from plotbee.videoplotter import extract_body, skeleton_drawer
from plotbee.tag import get_tag_image
from skimage import io
import numpy as np
import cv2
# from plotbee.video import parse_parts


def parse_parts(parts):
    # json only stores strings as keys
    # openCV needs points as tuple
    return {int(k):[tuple(v[0])] for k, v in parts.items()}



def valid_fn(body):
    if body.suppressed:
        return False
    if body._id == -1:
        return False
    return True



class Body():
    y_offset = 0
    width=200
    height=400
    pollen_thereshold = 0.5
    cX=None
    cY=None
    ignore_angle = False
    valid_function = valid_fn

    @classmethod
    def load_body(cls, body_dict, frameobj):
        if "tag" not in body_dict:
            body_dict["tag"] = None

        if "features" not in body_dict:
            body_dict["features"] = np.array(None)
        else:
            body_dict["features"] = np.array(body_dict["features"])
        parsed_parts = parse_parts(body_dict["parts"]) 

        body = Body(parsed_parts, body_dict["center_part"],
                    tuple(body_dict["angle_conn"]), body_dict["connections"],
                    frameobj, body_dict["id"], body_dict["suppressed"],
                    body_dict["pollen_score"], body_dict["tag"], body_dict["features"])
        return body


    def __init__(self, parts, center, angle_conn, connections, frame, body_id=-1, suppressed=False, pollen_score=0.0, tag=None, features=None):
        self._parts = parts
        self._center_part = center
        self._connections = connections
        self._frame = frame
        self._id = int(body_id)
        self._angle_conn = angle_conn
        self._prev = None
        self._next = None
        self.suppressed = suppressed
        self.pollen_score = pollen_score
        self.tag = tag
        self.features = features
 
    @property
    def valid(self):
        return self.valid_function()

    @property
    def frameid(self):
        return self._frame.id

    @property
    def tag_id(self):
        if self.tag is None:
            return None
        else:
            return self.tag["id"]

    @property
    def video_name(self):
        return self._frame.video_name


    @property
    def connections(self):
        return self._connections


    @property
    def id(self):
        return self._id


    @property
    def parts(self):
        return self._parts
    

    @property
    def center(self):
        x, y = self._parts[self._center_part][0]
        return x, y - Body.y_offset


    @property
    def angle(self):
        p1 = self.parts[self._angle_conn[0]][0]
        p2 = self.parts[self._angle_conn[1]][0]

        return angleBetweenPoints(p1, p2)

    @property
    def pollen(self):
        if self.pollen_score > Body.pollen_thereshold:
            return True
        else:
            return False

    def _image(self, width=None, height=None, cX=None, cY=None, ignore_angle=None, erase_tag=False):

        if width is None:
            width = Body.width
        if height is None:
            height = Body.height
        if cX is None:
            cX = Body.cX
        if cY is None:
            cY = Body.cY
        if ignore_angle is None:
            ignore_angle = Body.ignore_angle

        if erase_tag and self.tag is not None:
            pts = np.array(self.tag["p"]).astype(np.int32)
            pts = pts.reshape((-1,1,2))
            frame = cv2.fillPoly(self._frame.image,[pts], (0,0,0))
            frame = cv2.polylines(frame,[pts],True,(0,0,0),35)
        else:
            frame = self._frame.image

        return extract_body(frame, self, width=width, 
                            height=height, cX=cX, cY=cY, ignore_angle=ignore_angle)

    @property
    def image(self):
        return self._image()

    @property
    def skeleton_image(self):
        frame = self._frame.image    
        frame = skeleton_drawer(frame, self)
        
        return extract_body(frame, self, width=Body.width, 
                            height=Body.height, cX=Body.cX, cY=Body.cY)


    @property
    def prev(self):
        return self._prev

    @prev.setter
    def prev(self, p):
        self._prev = p

    @property
    def next(self):
        return self._next

    @next.setter
    def next(self, n):
        self._next = n
    
    
    def set_id(self, i):
        self._id = i
    
    
    def cbox(self, w_size=100, h_size=200):
        x, y = self.center
        return (x - w_size , y - h_size, x + w_size, y + h_size)

    def boundingBox(self):
        center = self.center
        angle = self.angle
        return rotatedBoundBoxPoints(center, angle)

    @property
    def skeleton(self):
        points = list()
        for part1, part2 in self._connections:
            if part1 not in self._parts:
                continue
            if part2 not in self._parts:
                continue
            points.append((self._parts[part1][0], self._parts[part2][0]))
        return points

    
    def __len__(self):
        return len(self._parts.keys())


    def __repr__(self):
        coords = repr(self.parts)
        coords = coords[coords.find('{'):-1]
        return "Body(id={}, parts={})".format(self.id, coords)


    def save(self, path, width=None, height=None, cX=None, cY=None, erase_tag=False):
        im = self._image(width=width, height=height, cX=cX, cY=cY, erase_tag=erase_tag)
        io.imsave(path, im)


    def info(self):
        x, y = self.center
        
        info = {
            "id": self.id,
            "frame": self.frameid,
            "angle": self.angle,
            "x": x,
            "y": y,
            "parts_num": len(self)
        }
        
        return info

    def tag_erased_image(self):
        if self.tag is None:
            return self.image
        else:
            return self._image(erase_tag=True)

    # def get_abdomen_image(self):
    #     frame = self._frame.image
    #     image_size = frame.shape[:2]
    #     width=Body.width
    #     height=Body.height
    #     x, y = self.center
    #     angle = self.angle
    #     return getRotationMatrix(image_size,x,y,angle, width, height)


    def tag_image(self):
        if self.tag is None:
            return np.array([])
        else:
            return get_tag_image(self)

    def params(self):
        body_dict = dict()
        body_dict["parts"] = self.parts
        body_dict["center_part"] = self._center_part
        body_dict["connections"] = self.connections
        body_dict["frameid"] = self.frameid
        body_dict["id"] = self.id
        body_dict["angle_conn"] = self._angle_conn
        body_dict["suppressed"] = self.suppressed
        body_dict["pollen_score"] = self.pollen_score
        body_dict["tag"] = self.tag
        body_dict["features"] = np.array(self.features).tolist()

        return body_dict
