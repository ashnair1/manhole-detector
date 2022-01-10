import argparse
import os

from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from manhole_detector.cfg import setup_cfg
from manhole_detector.dataset import register_manholes

register_manholes()

def get_parser():
    parser = argparse.ArgumentParser(description="Manhole Detector")
    parser.add_argument("-c", "--ckpt", help="Checkpoint file")
    return parser


def eval(ckpt_path):
    cfg = setup_cfg()
    if not ckpt_path:
        ckpt_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    assert os.path.isfile(ckpt_path)

    cfg.MODEL.WEIGHTS = ckpt_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("manhole_val", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "manhole_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))


if __name__ == "__main__":
    args = get_parser().parse_args()
    eval(args.ckpt)
