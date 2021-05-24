# plotbee
This project is library for plotting Beepose detections

## Installation

```
git clone --recurse-submodules https://github.com/jachansantiago/plotbee.git
cd plotbee
pip install -r requirement.txt
python setup.py install
```

## Command Line

### Skeleton
```
pb skeleton --file merged_C02_170622120000_detections.json --video C02_170622120000 --method beepose
```
This produce an `output_file = skeleton_merged_C02_170622120000_detections.json`.

### Pollen Detection

```
pb pollen --file data/C02_170622120000_skeleton.json --video data/C02_170622120000.mp4 \
--model_json /home/irodriguez/JANELIA/src/BeeLab/2l_model_2020_angle_auto_compensated1.json \
--weights /home/irodriguez/JANELIA/src/BeeLab/2l_model_2020_angle_auto_compensated1.h5
```
This produce an `output_file = data/pollen_merged_C02_170622120000_detections.json`.

### Tag Detetction
#### Compute
```
pb tags --file data/C02_170622120000_skeleton.json --video data/C02_170622120000.mp4 --method compute
```

#### merge
```
pb tags --file data/C02_170622120000_skeleton.json --tags_file Tags-C02_170622120000.json --method merge
```

### Tracking
```
pb tracking --file data/C02_170622120000_skeleton.json --method hungarian
```

## Demo

Try the [demo notebook](https://github.com/jachansantiago/plotbee/blob/master/notebooks/video_example.ipynb).

### Sample data
To download sample data run the following command.
```
./download_data
```


