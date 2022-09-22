import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from scipy.io import loadmat
from tqdm.auto import tqdm

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams, letterbox
from utils.general import (
    apply_classifier,
    check_img_size,
    check_imshow,
    check_requirements,
    increment_path,
    non_max_suppression,
    non_max_suppression_lmks,
    scale_coords,
    scale_coords_lmks,
    set_logging,
    strip_optimizer,
    xyxy2xywh,
)
from utils.plots import plot_one_box
from utils.torch_utils import TracedModel, load_classifier, select_device, time_synchronized


def get_gt_files(gt_folder):
    assert os.path.exists(gt_folder), gt_folder

    gt_mat = loadmat(os.path.join(gt_folder, "wider_face_val.mat"))
    hard_mat = loadmat(os.path.join(gt_folder, "wider_hard_val.mat"))
    medium_mat = loadmat(os.path.join(gt_folder, "wider_medium_val.mat"))
    easy_mat = loadmat(os.path.join(gt_folder, "wider_easy_val.mat"))

    facebox_list = gt_mat["face_bbx_list"]
    event_list = gt_mat["event_list"]
    file_list = gt_mat["file_list"]

    hard_gt_list = hard_mat["gt_list"]
    medium_gt_list = medium_mat["gt_list"]
    easy_gt_list = easy_mat["gt_list"]

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list


def detect(save_img=False):
    weights, imgsz, trace = opt.weights, opt.img_size, not opt.no_trace

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    _, event_list, file_list, _, _, _ = get_gt_files(opt.annotation_folder)

    # Run inference
    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        )  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for idx_event, (event, files) in tqdm(
        enumerate(zip(event_list, file_list)), total=len(event_list)
    ):
        event = event[0][0]

        for file in files[0]:
            file = str(file[0][0])

            file_path = os.path.join(opt.images_folder, str(event), f"{file}.jpg")
            assert os.path.exists(file_path)

            img0 = cv2.imread(file_path)

            img = letterbox(img0, imgsz, stride=stride)[0]

            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != "cpu" and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]
            ):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=opt.augment)[0]

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression_lmks(
                pred, opt.conf_thres, opt.iou_thres, agnostic=opt.agnostic_nms
            )
            t3 = time_synchronized()

            det = pred[0]

            txt_path = os.path.join(opt.outputs, str(event), f"{file}.txt")
            os.makedirs(os.path.dirname(txt_path), exist_ok=True)

            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            with open(txt_path, "w") as f:
                f.write(f"{file}\n")
                f.write(f"{len(det)}\n")

                # Write results
                for det_per_box in det:
                    xyxy, conf = (
                        det_per_box[0:4],
                        det_per_box[4],
                    )

                    xmin, ymin, xmax, ymax = xyxy

                    line = (xmin, ymin, xmax - xmin, ymax - ymin, min(conf, 1))  # label format

                    f.write(("%g " * len(line)).rstrip() % line + "\n")

    print(f"Done. ({time.time() - t0:.3f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", nargs="+", type=str, default="yolov7.pt", help="model.pt path(s)"
    )
    parser.add_argument(
        "--data-root",
        default="/home/coder/project/datasets/face-detections/widerface/",
        type=str,
    )
    parser.add_argument("--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="object confidence threshold"
    )
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IOU threshold for NMS")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--no-trace", action="store_true", help="don`t trace model")
    parser.add_argument(
        "--outputs",
        default="./outputs",
        type=str,
    )
    opt = parser.parse_args()

    opt.annotation_folder = os.path.join(opt.data_root, "ground_truth")
    opt.images_folder = os.path.join(opt.data_root, "WIDER_val", "images")

    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ["yolov7.pt"]:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
