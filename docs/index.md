<!-- .. plotbee documentation master file, created by
   sphinx-quickstart on Thu Jun  2 18:10:55 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive. -->

# Welcome to plotbee's documentation!

Plotbee is a library to process, manage and visualize [Beepose](https://github.com/jachansantiago/beepose) detections.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lcppKrnbxGmJelXcuitNfclOW8_JdvEe?usp=sharing)

![](https://github.com/jachansantiago/plotbee/raw/master/imgs/video.gif)

## Installation

```
git clone --recurse-submodules https://github.com/jachansantiago/plotbee.git
cd plotbee
pip install . # or pip install .[tags] to install with apriltag
```


```{toctree}
---
maxdepth: 2
caption: Reference
---
main_objects
cli
video_example
modules
```

## Demo

Try our [demo notebook](https://colab.research.google.com/drive/1lcppKrnbxGmJelXcuitNfclOW8_JdvEe?usp=sharing).

For bees annotations try our [body annotator notebook](https://colab.research.google.com/drive/1hOMPSU5XCVi8Sja-gF9CBwomgmmeysbE?usp=sharing).

### Sample data
To download sample data run the following command.
```
./download_data
```

## Indices and tables
<!-- ================== -->
* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`

<!-- * :ref:`genindex` -->
<!-- * :ref:`modindex`
* :ref:`search` -->

