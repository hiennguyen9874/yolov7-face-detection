import os
from PIL import Image
import argparse

import os
import collections


def get_shape(path: str):
    r"""Get shape of image path

    Args:
        path (str): path to image

    Returns:
        Tuple[int, int]: shape of image (height, width)
    """
    assert os.path.exists(path), path

    image = Image.open(path)

    return image.size[1], image.size[0]


def clip_and_normalize(xmin, ymin, w, h, height, width):
    xmax = xmin + w
    ymax = ymin + h
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(xmax, width - 1)
    ymax = min(ymax, height - 1)

    return (
        (xmin + xmax) / (2 * width),
        (ymin + ymax) / (2 * height),
        (xmax - xmin) / width,
        (ymax - ymin) / height,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="./winderface", help="image folders path")
    parser.add_argument(
        "--image-folder", type=str, default="WIDER_train/images", help="image folders path"
    )
    parser.add_argument(
        "--label-file", type=str, default="train/label.txt", help="image folders path"
    )
    parser.add_argument("--txt-file", type=str, default="train.txt", help="image folders path")
    arg = parser.parse_args()

    image_folder = os.path.join(arg.root, arg.image_folder)
    label_file = os.path.join(arg.root, arg.label_file)
    txt_file = os.path.join(arg.root, arg.txt_file)

    datas = collections.defaultdict(list)

    with open(label_file) as file:
        for line in file:
            if line.startswith("#"):
                image_file_path = os.path.join(
                    image_folder, line.split(" ")[1].replace("\n", "").replace(" ", "")
                )
                image_height, image_width = get_shape(image_file_path)
                assert os.path.exists(image_file_path)
            else:
                line = [float(x) for x in line.split()]

                if len(line) < 18:
                    line = line[:4]
                    datas[image_file_path].append(
                        clip_and_normalize(
                            line[0],
                            line[1],
                            line[2],
                            line[3],
                            image_height,
                            image_width,
                        )
                    )
                else:
                    line = line[:18]
                    datas[image_file_path].append(
                        (
                            (
                                clip_and_normalize(
                                    line[0],
                                    line[1],
                                    line[2],
                                    line[3],
                                    image_height,
                                    image_width,
                                )
                            ),
                            (
                                line[4] / image_width,  # landmark1_x
                                line[5] / image_height,  # landmark1_y
                                line[7] / image_width,  # landmark2_x
                                line[8] / image_height,  # landmark2_y
                                line[10] / image_width,
                                line[11] / image_height,
                                line[13] / image_width,
                                line[14] / image_height,
                                line[16] / image_width,
                                line[17] / image_height,
                            ),
                        )
                    )

    with open(txt_file, "w") as txt:
        for image_path, values in datas.items():
            txt.write(f"{image_path}\n")
            label_path = image_path.replace("images", "labels")
            label_path = os.path.splitext(label_path)[0] + ".txt"

            os.makedirs(os.path.dirname(label_path), exist_ok=True)

            with open(label_path, "w") as f:
                for value in values:
                    if len(value) == 2:
                        bboxes, landmarks = value
                        mask = -1 if any(x < 0 for x in landmarks) else 1
                        f.write(
                            "0"
                            + " "
                            + " ".join([str(x) for x in bboxes])
                            + " "
                            + " ".join([str(x) for x in landmarks])
                            + " "
                            + str(mask)
                            + "\n"
                        )
                    else:
                        bboxes = value
                        f.write("0" + " " + " ".join([str(x) for x in bboxes]) + "\n")
