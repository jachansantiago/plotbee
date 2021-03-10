# plotbee
This project is library for plotting Beepose detections

## Installation

```
git clone --recurse-submodules https://github.com/jachansantiago/plotbee.git
cd plotbee
pip install -r requirement.txt
python setup.py install
```

## Apriltag
To use the tag detection apriltag c++ code need to be installed. To install apriltag run the following commands:
```
cd apriltag
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```
## Demo

Try the [demo notebook](https://github.com/jachansantiago/plotbee/blob/master/notebooks/video_example.ipynb).

### Sample data
To download sample data run the following command.
```
./download_data
```


