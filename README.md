# YOLOpukki - YOLO v3 & v4 adapted version

This repository contains **YOLOpukki**, [YOLOv3](https://arxiv.org/pdf/1804.02767.pdf) and [YOLOv4](https://arxiv.org/pdf/2004.10934.pdf) (and their Tiny versions) for Tensorflow 2.x for use with custom datasets.

## Requirements

- opencv-python
- tensorflow
- tensorrt (optional)
- matplotlib (for the MNIST Dataset data generator Python notebook version)

## Installation

Clone or download this GitHub repository, install requirements:
```
pip install -r ./requirements.txt
```

## Usage

Use [prepare.bat](prepare.bat) or [prepare.sh](prepare.sh) to build train and test datasets from the MNIST data.
You can use directly [prepare.py](prepare.py) or interactively run [prepare.ipynb](prepare.ipynb) Python notebook.

To customize the YOLO model, dataset, TensorRT, etc. you can edit the supplied [yolov3/configs.py](yolov3/configs.py). Or copy some preconfigured file from [yolov3/configs](yolov3/configs), [yolov3/configs.tiny](yolov3/configs.tiny), [yolov3/coco](yolov3/coco) or [yolov3/coco.tiny](yolov3/coco.tiny) subdirectory.

Run [train.bat](train.bat)/[train.sh](train.sh) to train YOLOv3/YOLOv4 on the resulting dataset or use some pretrained data in the [checkpoints](checkpoints) directory. You can use *train.xxx.py* scripts to train using the specified device - generic version or GPUx (or just overwrite [train.py](train.py) file).

- [save.bat](save.bat)/[save.sh](save.sh) to save the selected model type in Tensorflow format (set *YOLO_TYPE*, *YOLO_TINY*, *YOLO_CUSTOM_WEIGHTS*, etc. in the [yolov3/configs.py](yolov3/configs.py))
- [convert.bat](convert.bat)/[convert.sh](convert.sh) to convert saved model to all of supported TensorRT types (*FP16*/*FP32*/*INT8*)
- [mAP.bat](mAP.bat)/[mAP.sh](mAP.sh) to evaluate mAP and performance of the selected model
- [detect.bat](detect.bat)/[detect.sh](detect.sh) to demonstrate image detection (should be *YOLO_CUSTOM_WEIGHTS = True*)
- [load.bat](load.bat)/[load.sh](load.sh) for detection demo using saved Tensorflow model (custom model of the selected type must be previously saved using [save.bat](save.bat)/[save.sh](save.sh))
- [detect_video.bat](detect_video.bat)/[detect_video.sh](detect_video.sh) to view a real-time generated video detection demo (should be *YOLO_CUSTOM_WEIGHTS = True*)
- [detect_video_mp.bat](detect_video_mp.bat)/[detect_video_mp.sh](detect_video_mp.sh) to view a video detection demo using the multiprocessing feature (should be *YOLO_CUSTOM_WEIGHTS = True*)
- [detect_video_file.bat](detect_video_file.bat) [test.mp4](test.mp4)/[detect_video_file.sh](detect_video_file.sh) [test.mp4](test.mp4) (or any other video file or even live video if no file name is specified) to view a multiprocessing video file detection demo (don't forget to edit [yolov3/configs.py](yolov3/configs.py) or copy it from [yolov3/coco](yolov3/coco) or [yolov3/coco.tiny](yolov3/coco.tiny) subdirectory, should be *YOLO_CUSTOM_WEIGHTS = False* for real video files)

Use the `q` key to exit the continuous demo or any other key for a new image.

All demos will produce image [test.jpg](test.jpg), video [output.avi](output.avi) or [test.avi](test.avi) files saved into the root directory of the project.

## Performance

Approximate performance data is shown in the following table.

| Device                  | Software configuration                                 | MODEL          | Performance (FPS)     |
|-------------------------|--------------------------------------------------------|----------------|-----------------------|
| NVIDIA A100             | CUDA 11.8, Tensorflow 2.12.0, cuDNN 8.8.1              | YOLOv3         | 9.16                  |
|                         | + TensorRT 8.3.4.1 FP32                                |                | 78.53                 |
|                         | + TensorRT 8.3.4.1 FP16                                |                | 83.55                 |
|                         | + TensorRT 8.3.4.1 INT8                                |                | 84.29                 |
|                         |                                                        | YOLOv4         | 5.99                  |
|                         | + TensorRT 8.3.4.1 FP32                                |                | 62.34                 |
|                         | + TensorRT 8.3.4.1 FP16                                |                | 66.69                 |
|                         | + TensorRT 8.3.4.1 INT8                                |                | 65.02                 |
|                         |                                                        | YOLOv3 Tiny    | 30.13                 |
|                         | + TensorRT 8.3.4.1 FP32                                |                | 249.98                |
|                         | + TensorRT 8.3.4.1 FP16                                |                | 235.11                |
|                         | + TensorRT 8.3.4.1 INT8                                |                | 238.31                |
|                         |                                                        | YOLOv4 Tiny    | 24.95                 |
|                         | + TensorRT 8.3.4.1 FP32                                |                | 181.71                |
|                         | + TensorRT 8.3.4.1 FP16                                |                | 194.79                |
|                         | + TensorRT 8.3.4.1 INT8                                |                | 191.29                |
| NVIDIA RTX 2060         | CUDA 10.1, Tensorflow 2.3.4, Keras 2.4.0, cuDNN 7.6.5  | YOLOv3         | 8.88                  |
|                         |                                                        | YOLOv4         | 4.72                  |
|                         |                                                        | YOLOv3 Tiny    | 26.69                 |
|                         |                                                        | YOLOv4 Tiny    | 20.18                 |
| NVIDIA GTX TITAN        | CUDA 10.1, Tensorflow 2.3.4, Keras 2.4.0, cuDNN 7.6.5  | YOLOv3         | 9.34                  |
|                         |                                                        | YOLOv4         | 4.67                  |
|                         |                                                        | YOLOv3 Tiny    | 26.31                 |
|                         |                                                        | YOLOv4 Tiny    | 21.80                 |
| NVIDIA GTX TITAN        | CUDA 11.4, Tensorflow 2.10.1, cuDNN 8.1.0              | YOLOv3         | 4.65                  |
|                         |                                                        | YOLOv4         | 2.93                  |
|                         |                                                        | YOLOv3 Tiny    | 14.63                 |
|                         |                                                        | YOLOv4 Tiny    | 12.09                 |
| NVIDIA RTX 2080         | CUDA 11.8, Tensorflow 2.10.1, cuDNN 8.8.1              | YOLOv3         | 4.45                  |
|                         |                                                        | YOLOv4         | 2.89                  |
|                         |                                                        | YOLOv3 Tiny    | 14.82                 |
|                         |                                                        | YOLOv4 Tiny    | 12.43                 |
| 2x Intel Xeon E5-2696v2 | Tensorflow 2.10.1, oneDNN with AVX support             | YOLOv3         | 2.18                  |
|                         |                                                        | YOLOv4         | 1.51                  |
|                         |                                                        | YOLOv3 Tiny    | 10.44                 |
|                         |                                                        | YOLOv4 Tiny    | 8.60                  |

## Contributing

**[YOLOpukki](https://github.com/assa77/YOLOpukki)** was developed by *[Alexander M. Albertian](mailto:assa@4ip.ru)*.
YOLO model is based on YOLOv3 and YOLOv4 public Python code.

All contributions are welcome! Please do **NOT** use an editor that automatically reformats whitespace.

## License

All contributions are made under the [GPLv3 license](http://www.gnu.org/licenses/gpl-3.0.en.html). See [LICENSE](LICENSE).
