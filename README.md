# plotbee
Plotbee is library for plotting Beepose detections

## Installation

```
git clone --recurse-submodules https://github.com/jachansantiago/plotbee.git
cd plotbee
pip install -r requirement.txt
python setup.py install
```

## Command Line

### Skeleton
The skeleton sub-command converts beepose and SLEAP (not implemented yet) detections into plotbee format. `beepose` is the default method. Note that if the video is given at this step, no need to input it at the next steps; meanwhile, the video location does not change.

```
pb skeleton --file merged_C02_170622120000_detections.json --video C02_170622120000.mp4 --method beepose
```
This command produces an `output_file = skeleton_merged_C02_170622120000_detections.json`.

### Pollen Detection
The pollen sub-command performs pollen detection at the detection level. Model's JSON and weights are required to perform pollen detection. Pollen detection uses a parallel implementation that creates temporary files at 'pollen_temp' directory. The default amount of workers is 4. Use `CUDA_VISIBLE_DEVICES` to restrict the usage of GPU devices.

```
pb pollen --file data/C02_170622120000_skeleton.json --video data/C02_170622120000.mp4 \
--model_json /home/irodriguez/JANELIA/src/BeeLab/2l_model_2020_angle_auto_compensated1.json \
--weights /home/irodriguez/JANELIA/src/BeeLab/2l_model_2020_angle_auto_compensated1.h5 --workers 4
```
This command produces an `output_file = data/pollen_merged_C02_170622120000_detections.json`.

Benchmark for one hour video.

|   Workers     | Time          |  Memory  |
| ------------- | ------------- | -------- |
|       4       |    ~50 min    |    8GB   |
|       8       |   not yet     |    16GB  |

### Tag Detetction

Tags sub-command can compute or merge tag detections. Use `--compute` to perform the tag detection with AprilTags. Note that tag detection requires images from the video. Be sure that the skeleton file contains the right localization of the video. Alternatively, the video location can be modified with `--video`.
#### Compute
```
pb tags --file data/C02_170622120000_skeleton.json --video data/C02_170622120000.mp4 --compute
```

#### merge
Use `--merge` and `--tags_file` to combine previously computed detections into a plotbee format video. Note that `pb tags` requires one of this options `--compute` or `--merge` to run. 
```
pb tags --file data/C02_170622120000_skeleton.json --tags_file Tags-C02_170622120000.json --merge
```
Both commands produces an `output_file = data/tags_C02_170622120000_detections.json`.

### Tracking
Tracking just requires the `--method`. The method can be `hungarian` (default) or `sort`.
```
pb tracking --file data/C02_170622120000_skeleton.json --method hungarian
```
This command can produces an `output_file = data/hungarian_C02_170622120000_detections.json` or `data/sort_C02_170622120000_detections.json`.

### Full Pipeline
Use `pb pipeline` to perform the whole pipeline at once. To active each step in the pipeline use `--skeleton`, `--tags`, `--pollen`, `--tracking`. Note that the required parameters for each step need to be input as is shown above. Start with `--skeleton` is not required. 
```
pb pipeline --skeleton --file merged_C02_170622120000_detections.json --video C02_170622120000 --method beepose \
--pollen --model_json 2l_model_2020_angle_auto_compensated1.json \
--weights 2l_model_2020_angle_auto_compensated1.h5 \
--tags --tags_file Tags-C02_170622120000.json --method merge \
--tracking --method hungarian
```

### Dataset export

`pb export` create pollen and tag dataset from videos in plotbee format. The options `--pollen` and `--tags` (not implemented yet) are mutually exclusive. Image dimensions `--width` and `--height` are required fields. A fixed `--size` dataset is supported and returns a balanced dataset. `size//2` images with the highest pollen scores and `size//2` images with the lowest pollen scores. If `--size` is not provided the whole video will be exported. `--output_folder` is also required.
```
pb export --pollen --output_folder pollen_data --file test_cli/pollen_tags_skeleton_merged_C02_170628120000_detections.json --width 375 --height 450 --size 200
```


## Demo

Try the [demo notebook](https://github.com/jachansantiago/plotbee/blob/master/notebooks/video_example.ipynb).

### Sample data
To download sample data run the following command.
```
./download_data
```


