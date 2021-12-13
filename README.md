## Manhole Detector

Simple repo to show how [detectron2](https://github.com/facebookresearch/detectron2) can be used to build, train and test a model.

### Requirements

- torch>=1.8
- detectron2==0.6
- cudatoolkit==10.2

### Train

```python
python manhole-detector/train.py
```

### Inference

```python
python manhole-detector/infer.py -i path/to/img/directory -o path/to/output/directory
```