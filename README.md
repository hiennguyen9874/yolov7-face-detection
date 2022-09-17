# YOLOV7 Face Detection

## Training

## Testing

- `python3 eval.py --weights runs/train/yolov7-tiny14/weights/best.pt --data-root /home/coder/project/datasets/face-detections/widerface/ --img-size 640 --conf-thres 0.02 --iou-thres 0.5 --device 2 --no-trace`

- `python3 evaluation/main.py -p ./outputs -g /home/coder/project/datasets/face-detections/widerface/val`
