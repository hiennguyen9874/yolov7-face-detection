# YOLOv7 Face Detection

Custom Yolov7 to detect face and estimate landmark.

<div align="center">
    <a href="./">
        <img src="./figure/face_detection_sample.jpg" width="75%"/>
    </a>
</div>

<div align="center">
    <a href="./">
        <img src="./figure/7_Cheering_Cheering_7_543.jpg" width="75%"/>
    </a>
</div>

## Install

- `pip3 install -r requirements.txt`
- `pip3 install Cython`
- `cd evaluation && sudo python3 setup.py install`
- [daquexian/onnx-simplifier](https://github.com/daquexian/onnx-simplifier)
- [onnx-graphsurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon)
- If export tensorRT [hiennguyen9874/TensorRT](https://github.com/hiennguyen9874/TensorRT)

## Performance

| Name        | Dataset    | Epoch | Easy   | Medium | Hard   | Link                                                                                                    |
| ----------- | ---------- | ----- | ------ | ------ | ------ | ------------------------------------------------------------------------------------------------------- |
| yolov7-tiny | Winderface | 80    | 0.9402 | 0.9197 | 0.8038 | [Link](https://github.com/hiennguyen9874/yolov7-face-detection/releases/download/v0.1/yolov7-tiny33.pt) |
| yolov7-tiny | Winderface | 300   | 0.9470 | 0.9270 | 0.8183 | [Link](https://github.com/hiennguyen9874/yolov7-face-detection/releases/download/v0.1/yolov7-tiny41.pt) |
| yolov7      | Winderface | 80    | 0.9563 | 0.9423 | 0.8639 | [Link](https://github.com/hiennguyen9874/yolov7-face-detection/releases/download/v0.1/yolov73.pt)       |
| yolov7      | Winderface | 300   | 0.9645 | 0.9526 | 0.8787 | [Link](https://github.com/hiennguyen9874/yolov7-face-detection/releases/download/v0.1/yolov75.pt)       |

## Data preparation

- Download and extract [winderface](http://shuoyang1213.me/WIDERFACE/index.html) dataset.
- Download and extract annotation file [retinaface_gt_v1.1.zip](https://github.com/deepinsight/insightface/tree/master/detection/retinaface).
- Download [ground_truth](https://github.com/deepcam-cn/yolov5-face/tree/master/widerface_evaluate/ground_truth).
- Folder after download and extract all:
  ```
  - ./winderface
      - WIDER_test/
          - images/
              - 0--Parade/
              - ...
      - WIDER_train/
          - images/
              - 0--Parade/
              - ...
      - WIDER_val/
          - images/
              - 0--Parade/
              - ...
      - train/
          - labels.txt
      - val/
          - labels.txt
      - test/
          - labels.txt
      - ground_truth/
          - wider_easy_val.mat
          - wider_medium_val.mat
          - wider_hard_val.mat
          - wider_face_val.mat
  ```
- Convert label to yolo format:
  - `python3 scripts/convert_to_yolo.py --root ./widerface --image-folder WIDER_train/images --label-file train/label.txt --txt-file train.txt`
  - `python3 scripts/convert_to_yolo.py --root ./widerface --image-folder WIDER_val/images --label-file val/label.txt --txt-file val.txt`
- Modify path of `winderface` folder in [data/winderface.yaml](data/winderface.yaml)

## Detect

- `python3 detect.py --weights ./weights/yolov7-tiny.pt --source inference/images --img-size 640 --conf-thres 0.2 --iou-thres 0.5 --device 1 --no-trace`

## Testing

- `python3 eval.py --weights ./weights/yolov7-tiny.pt --data-root ./winderface --img-size 640 --conf-thres 0.02 --iou-thres 0.5 --device 0 --no-trace`

- `python3 evaluation/main.py -p ./outputs -g ./winderface/ground_truth`

## Training

- Download file [yolov7-tiny.pt](https://github.com/WongKinYiu/yolov7/releases/tag/v0.1) and save as `./weights/yolov7-tiny-origin.pt`.
- Single GPU training: `python3 ./train.py --device 0 --batch-size 16 --data data/widerface.yaml --img 640 640 --cfg cfg/yolov7-tiny-landmark.yaml --weights ./weights/yolov7-tiny-origin.pt --name yolov7-tiny --hyp data/hyp.scratch.tiny.yaml --noautoanchor --linear-lr --epochs 80`

- Multiple GPU training: `torchrun --standalone --nnodes=1 --nproc_per_node 2 ./train.py --device 0,1 --batch-size 16 --data data/widerface.yaml --img 640 640 --cfg cfg/yolov7-tiny-landmark.yaml --weights ./weights/yolov7-tiny-origin.pt --name yolov7-tiny --hyp data/hyp.scratch.tiny.yaml --noautoanchor --sync-bn --linear-lr --epochs 80`

## Export

### ONNX

- `python3 export.py --weights ./weights/yolov7-tiny.pt --img-size 640 --batch-size 1 --dynamic-batch --grid --end2end --max-wh 640 --topk-all 100 --iou-thres 0.5 --conf-thres 0.2 --device 1 --simplify --cleanup`

- [tools/Yolov7onnx_mask.ipynb](./tools/Yolov7onnx_mask.ipynb)

### TensorRT
- Install [custom TensorRT plugin](https://github.com/hiennguyen9874/TensorRT#install)

- `python3 export.py --weights ./weights/yolov7-tiny.pt --img-size 640 --batch-size 1 --dynamic-batch --grid --end2end --max-wh 640 --topk-all 100 --iou-thres 0.5 --conf-thres 0.2 --device 1 --simplify --cleanup --trt`

- `/usr/src/tensorrt/bin/trtexec --onnx=./weights/yolov7-tiny.onnx --saveEngine=./weights/yolov7-tiny-nms-trt.trt --workspace=8192 --fp16 --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:4x3x640x640 --shapes=images:1x3x640x640`

- [tools/YOLOv7trt_mask.ipynb](./tools/YOLOv7trt_mask.ipynb)

## Bugs

- val/llmks_loss = 0?

## TODO

- How to achieve mAP of yolov7-tiny in [deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face)

## Acknowledgments

- [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- [deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face)
