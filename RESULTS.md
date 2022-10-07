## YOLOv7-Tiny

### One mask for all point

| Name          | Dataset           | Epoch | lmks  | loss_ota | lrf  | shear | mixup | mosaic | batchsize | Easy   | Medium | Hard   |
| ------------- | ----------------- | ----- | ----- | -------- | ---- | ----- | ----- | ------ | --------- | ------ | ------ | ------ |
| yolov7-tiny14 | Winderface        | 80    | 0.005 | 0        | 0.01 | 0.0   | 0     | 0.5    | 32        | 0.9336 | 0.9146 | 0.7996 |
| yolov7-tiny15 | Winderface        | 80    | 0.005 | 1        | 0.01 | 0.0   | 0     | 0.5    | 32        | 0.9320 | 0.9098 | 0.8016 |
| yolov7-tiny16 | Winderface        | 80    | 0.005 | 1        | 0.01 | 0.0   | 0.05  | 1.0    | 32        | 0.9364 | 0.9151 | 0.8055 |
| yolov7-tiny17 | Winderface        | 80    | 0.01  | 1        | 0.01 | 0.0   | 0.05  | 1.0    | 32        | 0.9362 | 0.9151 | 0.8065 |
| yolov7-tiny18 | Winderface        | 80    | 0.005 | 1        | 0.01 | 0.0   | 0.05  | 1.0    | 16        | 0.9384 | 0.9172 | 0.8073 |
| yolov7-tiny19 | Winderface + MTFL | 80    | 0.01  | 1        | 0.01 | 0.0   | 0.05  | 1.0    | 16        | 0.1371 | 0.0742 | 0.0309 |
| yolov7-tiny20 | Winderface        | 80    | 0.01  | 1        | 0.01 | 0.0   | 0.05  | 1.0    | 16        | 0.9371 | 0.9170 | 0.8094 |
| yolov7-tiny21 | Winderface        | 80    | 0.005 | 0        | 0.2  | 0.5   | 0     | 0.5    | 16        | 0.9391 | 0.9175 | 0.8003 |

### One mask for one point

| Name          | Dataset    | Epoch | lmks  | loss_ota | lrf | shear | mixup | mosaic | batchsize | Easy   | Medium | Hard   |
| ------------- | ---------- | ----- | ----- | -------- | --- | ----- | ----- | ------ | --------- | ------ | ------ | ------ |
| yolov7-tiny22 | Winderface | 80    | 0.005 | 0        | 0.2 | 0.5   | 0     | 0.5    | 16        | 0.9407 | 0.9191 | 0.7985 |
| yolov7-tiny23 | Winderface | 80    | 0.01  | 0        | 0.2 | 0.5   | 0     | 0.5    | 16        | 0.9401 | 0.9182 | 0.8006 |
| yolov7-tiny24 | Winderface | 80    | 0.01  | 1        | 0.2 | 0.5   | 0     | 0.5    | 16        | 0.9378 | 0.9165 | 0.8028 |
| yolov7-tiny25 | Winderface | 80    | 0.05  | 1        | 0.2 | 0.5   | 0     | 0.5    | 16        | 0.9222 | 0.8990 | 0.7768 |

### One mask for one point + dw_conv_landmark = false

| Name                            | Dataset    | Epoch | lmks  | loss_ota | lrf  | shear | mixup | mosaic | batchsize | Easy   | Medium | Hard   |
| ------------------------------- | ---------- | ----- | ----- | -------- | ---- | ----- | ----- | ------ | --------- | ------ | ------ | ------ |
| yolov7-tiny27                   | Winderface | 80    | 0.005 | 0        | 0.2  | 0.5   | 0     | 0.5    | 16        | 0.9375 | 0.9175 | 0.7992 |
| yolov7-tiny30                   | Winderface | 80    | 0.01  | 1        | 0.2  | 0.5   | 0.05  | 1.0    | 16        | 0.9378 | 0.9172 | 0.7987 |
| yolov7-tiny32 (linear_lr=False) | Winderface | 80    | 0.01  | 1        | 0.2  | 0.5   | 0.05  | 1.0    | 16        | 0.9374 | 0.9169 | 0.8011 |
| yolov7-tiny33                   | Winderface | 80    | 0.005 | 1        | 0.2  | 0.5   | 0.05  | 1.0    | 16        | 0.9402 | 0.9197 | 0.8038 |
| yolov7-tiny36 (linear_lr=False) | Winderface | 80    | 0.005 | 1        | 0.2  | 0.5   | 0.05  | 1.0    | 16        | 0.9351 | 0.9167 | 0.7995 |
| yolov7-tiny35 (linear_lr=False) | Winderface | 80    | 0.005 | 1        | 0.01 | 0.0   | 0.05  | 1.0    | 16        | 0.9339 | 0.9163 | 0.7996 |
| yolov7-tiny37                   | Winderface | 80    | 0.005 | 1        | 0.01 | 0.0   | 0.05  | 1.0    | 16        | 0.9362 | 0.9175 | 0.8010 |
| yolov7-tiny40 (No weights)      | Winderface | 300   | 0.005 | 1        | 0.2  | 0.5   | 0.05  | 1.0    | 16        | 0.9378 | 0.9199 | 0.8187 |
| yolov7-tiny41                   | Winderface | 300   | 0.005 | 1        | 0.2  | 0.5   | 0.05  | 1.0    | 16        | 0.9470 | 0.9270 | 0.8183 |

## YOLOv7

### One mask for one point + dw_conv_landmark = false

| Name    | Dataset    | Epoch | lmks  | loss_ota | lrf | cls | obj | translate | obj | scale | mixup | mosaic | batchsize | Easy   | Medium | Hard   |
| ------- | ---------- | ----- | ----- | -------- | --- | --- | --- | --------- | --- | ----- | ----- | ------ | --------- | ------ | ------ | ------ |
| yolov73 | Winderface | 80    | 0.005 | 1        | 0.1 | 0.3 | 0.7 | 0.2       | 0.7 | 0.9   | 0.15  | 1.0    | 16        | 0.9563 | 0.9423 | 0.8639 |
| yolov75 | Winderface | 300   | 0.005 | 1        | 0.1 | 0.3 | 0.7 | 0.2       | 0.7 | 0.9   | 0.15  | 1.0    | 16        | 0.9645 | 0.9526 | 0.8787 |
