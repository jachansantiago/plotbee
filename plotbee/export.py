from plotbee.video import Video
from plotbee.body import Body
from plotbee.utils import save_json
from skimage import io
import os
import cv2
from tqdm import tqdm
from collections import defaultdict

def get_full_skeleton(video):
    max_part_body = []
    for frame in video:
        for body in frame:
            if len(body) > len(max_part_body):
                max_part_body = body
    
    parts = list(max_part_body._parts.keys())
    return parts, max_part_body._connections

def coco_file_init(keypoints, skeleton):
    
    info = {"description": "Open Pose Bees", "video": "Camera3-5min_h264",
            "date_created": "Oct_2018",
            "year": 2021,
            "contributor": "J. Chan, I. Rodriguez, R. Megret", 
            "version": 0.02}
    
    categories= [{'name':'bee',
                  'super_category':'animal',
                  'id':1,
                  'keypoints':keypoints,'skeleton':skeleton}]
    
    
    year = 2021
    date = 'MAY-2021'
    
    file={}
    file['images']=[]
    file['info']={}
    file['annotations']=[]
    file['categories']=[]
    file['categories']=categories 
    file['info']=info
    file['info']['Date_created']=date
    file['info']['year']=year
    
    return file

def image_annotation(video_name, frame_id, image_id, width=2560, height=1440):
    name_format = "{video_name}_{frame:09}.jpg"
    name = name_format.format(video_name=video_name, frame=frame_id)
    im_dict = dict()
    im_dict["id"] = image_id
    im_dict["width"] = width
    im_dict["height"] = height
    im_dict["file_name"] = name
    return im_dict

def parts_annotations(body, keypoints):
    body_parts = []
    for k in keypoints:
        if k not in body._parts:
            # missing part
            body_parts.append(0)
            body_parts.append(0)
            body_parts.append(0)
        else:
            x, y = body._parts[k][0]
            body_parts.append(x)
            body_parts.append(y)
            body_parts.append(1)
            empty_frame = False
    return body_parts
                    
    
def body_annotation(body, keypoints, annotation_id, image_id, width=300, height=450):
    body_parts = parts_annotations(body, keypoints)
    bbox = body.boundingBox()
    ann = dict()
    ann["area"] = width*height
    ann["bbox"] = [coord for coords in bbox for coord in coords ]
    ann['category_id']=1
    ann['image_id']= image_id
    ann['id']=int(annotation_id)
    ann['iscrowd'] =0
    ann['keypoints']=keypoints
    ann['num_keypoints']= len(keypoints)
    ann['segmentation']=[ann['bbox']]
    
    return ann


def extract_images(output_folder, video):
    video_name = video.video_name
    video_name, _ = os.path.splitext(video_name)
    
    stream = video.get_video_stream()
    total_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    
    success, img = stream.read()
    for fno in tqdm(range(total_frames)):
        name_format = "{video_name}_{frame:09}.jpg"
        name = name_format.format(video_name=video_name, frame=fno)
        im_path = os.path.join(output_folder, name)
        io.imsave(im_path, img)
        success, img = stream.read()
    stream.release()
    
def extract_annotations(video, width=300, height=450):
    
    keypoints, skeleton = get_full_skeleton(video)
    
    coco_file = coco_file_init(keypoints, skeleton)
    
    
    Body.width = width
    Body.height = height
    
    image_id = 1
    annotation_id = 1
    
    video_name = video.video_name
    video_name, _ = os.path.splitext(video_name)

    for frame in tqdm(video):
        
        image_data = image_annotation(video_name, frame.id, image_id)
        empty_frame = True
        
        for body in frame:
            ann = body_annotation(body, keypoints, annotation_id, image_id, width=width, height=height)
            annotation_id +=1
            coco_file['annotations'].append(ann)
            empty_frame = False
            
        if not empty_frame:
            coco_file['images'].append(image_data)
            
        image_id +=1
    return coco_file
        
    save_json(json_output, coco_file)
    
    
    

def video2coco(video_fname, output_folder, image_extraction=False, width=300, height=450):
    
    json_output= os.path.join(output_folder,'coco_annotations.json')
    os.makedirs(output_folder,exist_ok=True)
    
    
    video = Video.load(video_fname)
    
    coco_file = extract_annotations(video, width=width, height=height)
        
    save_json(json_output, coco_file)
    
    if image_extraction:
        output_folder_images = os.path.join(output_folder,'images')
        os.makedirs(output_folder_images,exist_ok=True)
        
        extract_images(output_folder_images, video)
    return
   
# sleap

def parse_videos(videos):
    parsed_videos = list()
    for video in videos:
        parsed_videos.append(video['backend']['filename'])
    return parsed_videos
    
def parse_skeletons(skeletons):
    parsed_skeletons = list()
    for skeleton in skeletons:
        name = skeleton["graph"]["name"]
        connections, keypoints = parse_skeleton(skeleton)
        parsed_skeletons.append((connections, keypoints))
    return parsed_skeletons

def parse_skeleton(skeleton):
    parsed_skeleton = list()
    for link in skeleton['links']:
        connection = (link["source"], link["target"])
        parsed_skeleton.append(connection)
    keypoints = list()
    for node in skeleton["nodes"]:
        keypoints.append(node['id'])
    return parsed_skeleton, keypoints
    
def get_body(skeleton, frame_obj, skeletons):
    parts = dict()
    
    for part, detection in skeleton["_points"].items():
        points = [(int(detection['x']), int(detection['y']))]
        parts[int(part)] = points
        
    center = 3
    angle_conn=(3, 4)
    
    skeleton_id = int(skeleton['skeleton'])
    connections = skeletons[skeleton_id]
#     connections = [[1, 4], [1, 0], [4, 3], [3, 5]]
    
    score=None
    
    if 'score' in skeleton:
        score = skeleton["score"]
    
    return Body(parts, center, angle_conn, connections, frame_obj, features=score)

def get_frame(frame, skeletons):
    frame_id = frame["frame_idx"]
    frameobj = Frame([], frame_id)
    bodies = list()
    
    for skeleton in frame['_instances']:
        body = get_body(skeleton, frameobj, skeletons)
        if 3 not in body._parts or 4 not in body._parts:
            continue
        bodies.append(body)
        
    frameobj.update(bodies)
        
    return frameobj

def sleap2pb(sleap_json):
    video_data = read_json(sleap_json)
    frames = defaultdict(list)
    track_dict = dict()
    config = defaultdict(dict)
    
    skeletons = parse_skeletons(video_data["skeletons"])
    
    videos = parse_videos(video_data["videos"])
    
    for video_id, video_filename in enumerate(videos):
        config[video_id]["VIDEO_PATH"] = video_filename
    
    for frame in video_data['labels']:
        video_id = int(frame["video"])
        frames[video_id].append(get_frame(frame, skeletons))
        
    for video_frames in frames.values():
        video_frames.sort(key=lambda x: x.id)
        
    out_videos = list()
    for video_frames, video_config in zip(frames.values(), config.values()):
        out_videos.append(Video(video_frames, track_dict, video_config))
        
    for video in out_videos:
        name = video.video_name
        name, ext = os.path.splitext(name)
        out_name = "skeleton_" + name + ".json"
        video.save(out_name)
        
    return
    
    