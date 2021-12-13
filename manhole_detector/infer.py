import argparse
import os
import random

import cv2
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from manhole_detector.cfg import setup_cfg
from manhole_detector.dataset import (DATA_DIR, get_manhole_dicts,
                                      register_manholes)

register_manholes()

def get_parser():
    parser = argparse.ArgumentParser(description="Manhole Detector")
    parser.add_argument("-i", "--input", help="Input image directory")
    parser.add_argument("-o", "--output", help="Output image directory")
    return parser

def infer_dir(directory, out_dir=None):
    cfg = setup_cfg()
    cfg.MODEL.WEIGHTS = os.path.join(
        cfg.OUTPUT_DIR, "model_0004999.pth"
    )  # path to the model we just trained
    manhole_metadata = MetadataCatalog.get("manhole_val")
    predictor = DefaultPredictor(cfg)
    out_dir = directory if not out_dir else out_dir
    for img in os.listdir(directory):
        imgpath = os.path.join(directory, img)
        imgname = os.path.basename(img)
        im = cv2.imread(imgpath)
        out = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=manhole_metadata)
        out = v.draw_instance_predictions(out["instances"].to("cpu"))
        name, ext = os.path.splitext(img)
        out_name = name + "_pred" + ext
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, out.get_image()[:, :, ::-1])


def infer_val():

    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)

    dataset_dicts = get_manhole_dicts(DATA_DIR + "val")
    manhole_metadata = MetadataCatalog.get("manhole_val")
    for d in random.sample(dataset_dicts, 50):
        im = cv2.imread(d["file_name"])
        outputs = predictor(
            im
        )  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(
            im[:, :, ::-1],
            metadata=manhole_metadata,
            scale=0.5,
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        window_name = "output"
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        cv2.imshow(window_name, out.get_image()[:, :, ::-1])
        cv2.waitKey(0)


if __name__ == "__main__":
    args = get_parser().parse_args()
    assert os.path.isdir(args.input)
    assert os.path.isdir(args.output)
    infer_dir(args.input,args.output)
