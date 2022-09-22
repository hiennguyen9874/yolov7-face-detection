# ONNX

- `python3 export.py --weights ./weights/yolov7-tiny33.pt --img-size 640 --batch-size 1 --dynamic-batch --grid --end2end --max-wh 640 --topk-all 100 --iou-thres 0.5 --conf-thres 0.2 --device 1 --simplify`

# TensorRT

- `python3 export.py --weights ./weights/yolov7-tiny33.pt --img-size 640 --batch-size 1 --dynamic-batch --grid --end2end --max-wh 640 --topk-all 100 --iou-thres 0.5 --conf-thres 0.2 --device 1 --simplify --cleanup --trt`

- `/usr/src/tensorrt/bin/trtexec --onnx=./weights/yolov7-tiny33.onnx --saveEngine=./weights/yolov7-tiny33-nms-trt.trt --workspace=8192 --fp16 --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:4x3x640x640 --shapes=images:1x3x640x640`
