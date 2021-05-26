from plotbee.video import Video
from plotbee.body import Body
from plotbee.utils import save_json
from skimage import io
import os
import cv2
from tqdm import tqdm

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
    