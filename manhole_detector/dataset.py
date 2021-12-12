import os
import random

import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer

DATA_DIR = "/media/ashwin/DATA2/manhole-detector/data/"
CATEGORIES = {0: "open", 1: "closed", 2: "improper"}


def yolobbox2bbox(box, imgh, imgw):
    x, y, w, h = box
    x = float(x)
    y = float(y)
    w = float(w)
    h = float(h)
    # Taken from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
    # via https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection#comment102178409_44592380
    l = int((x - w / 2) * imgw)
    r = int((x + w / 2) * imgw)
    t = int((y - h / 2) * imgh)
    b = int((y + h / 2) * imgh)

    l = max(l, 0)
    r = min(r, imgw - 1)
    t = max(t, 0)
    b = min(b, imgh - 1)

    return [l, t, r, b]


def get_manhole_dicts(data_dir):
    img_dir = os.path.join(data_dir, "imgs")
    lbl_dir = os.path.join(data_dir, "lbls")

    imgs = sorted([i for i in os.listdir(img_dir) if i.endswith(".jpg")])
    lbls = sorted([i for i in os.listdir(lbl_dir) if i.endswith(".txt")])

    dataset_dicts = []
    for id, (img, lbl) in enumerate(zip(imgs, lbls)):
        record = {}
        img = os.path.join(img_dir, img)
        lbl = os.path.join(lbl_dir, lbl)

        height, width = cv2.imread(img).shape[:2]
        record["image_id"] = id
        record["file_name"] = img
        record["height"] = height
        record["width"] = width

        with open(lbl) as f:
            lines = f.readlines()

        lines = [l.rstrip() for l in lines]
        objs = []

        for ins in lines:
            ann = ins.split()
            cat = ann[0]
            bbox = ann[1:]
            category = CATEGORIES[int(cat)]
            bbox_yolo = [float(i) for i in bbox]

            bbox = yolobbox2bbox(bbox_yolo, height, width)

            obj = {
                "image_id": id,
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": int(cat),
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def register_manholes():

    for d in ["train", "val"]:
        DatasetCatalog.register(
            "manhole_" + d, lambda d=d: get_manhole_dicts(DATA_DIR + d)
        )
        MetadataCatalog.get("manhole_" + d).set(thing_classes=list(CATEGORIES.values()))


if __name__ == "__main__":
    register_manholes()
    split = "train"
    manhole_metadata = MetadataCatalog.get(f"manhole_{split}")

    dataset_dicts = get_manhole_dicts(DATA_DIR + split)
    for d in random.sample(dataset_dicts, 5):
        img = cv2.imread(d["file_name"])
        print(f'Plotting {d["file_name"]}')
        visualizer = Visualizer(img[:, :, ::-1], metadata=manhole_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)

        # plt.imshow(out.get_image())
        # plt.show()

        window_name = "dataset"
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        cv2.imshow(window_name, out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
