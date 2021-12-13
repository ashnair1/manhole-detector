import os

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

from manhole_detector.cfg import setup_cfg
from manhole_detector.dataset import register_manholes

register_manholes()


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        data_evaluator = COCOEvaluator

        return data_evaluator(
            dataset_name, output_dir=output_folder, use_fast_impl=False
        )


def main():
    cfg = setup_cfg()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()


if __name__ == "__main__":
    main()
