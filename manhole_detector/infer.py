import argparse
import os
import random

import cv2
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from manhole_detector.cfg import setup_cfg
from manhole_detector.dataset import (
    CATEGORIES,
    DATA_DIR,
    get_manhole_dicts,
    register_manholes,
)

register_manholes()


def get_parser():
    parser = argparse.ArgumentParser(description="Manhole Detector")
    parser.add_argument("-i", "--input", help="Input image directory")
    parser.add_argument("-o", "--output", help="Output image directory")
    parser.add_argument("-c", "--ckpt", help="Checkpoint file")
    return parser


def infer_img(img, out_dir=None, ckpt_path=None):
    assert os.path.isfile(img)
    cfg = setup_cfg()
    if not ckpt_path:
        ckpt_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    cfg.MODEL.WEIGHTS = ckpt_path
    if out_dir:
        assert os.path.isdir(out_dir)

    predictor = DefaultPredictor(cfg)
    im = cv2.imread(img)
    out = predictor(im)
    predictions = out["instances"].to("cpu")
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = (
        predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
    )
    bboxes = boxes.tensor.numpy()
    det_results = [bboxes, classes, scores, CATEGORIES]

    # Draw detections on img
    manhole_metadata = MetadataCatalog.get("manhole_val")
    v = Visualizer(im[:, :, ::-1], metadata=manhole_metadata)
    out = v.draw_instance_predictions(predictions)
    det_img = out.get_image()[:, :, ::-1]
    if out_dir:
        # Write to disk
        imgname = os.path.basename(img)
        name, ext = os.path.splitext(imgname)
        out_name = name + "_pred" + ext
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, det_img)

    return det_img, det_results


def infer_dir(in_dir, out_dir=None, ckpt_path=None):
    cfg = setup_cfg()
    if not ckpt_path:
        ckpt_path = os.path.join(cfg.OUTPUT_DIR, "model_0004999.pth")

    cfg.MODEL.WEIGHTS = ckpt_path
    manhole_metadata = MetadataCatalog.get("manhole_val")
    predictor = DefaultPredictor(cfg)
    out_dir = in_dir if not out_dir else out_dir
    for img in os.listdir(in_dir):
        imgpath = os.path.join(in_dir, img)
        im = cv2.imread(imgpath)
        out = predictor(im)

        # Draw detections on img
        v = Visualizer(im[:, :, ::-1], metadata=manhole_metadata)
        out = v.draw_instance_predictions(out["instances"].to("cpu"))
        name, ext = os.path.splitext(img)
        out_name = name + "_pred" + ext
        out_path = os.path.join(out_dir, out_name)
        # Write to disk
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
            scale=1,
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
    infer_dir(args.input, args.output, args.ckpt)
    # _, _ = infer_img(args.input, args.output, args.ckpt)
