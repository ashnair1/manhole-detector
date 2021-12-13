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

### Docker


1. Build docker image
```
docker build --build-arg USER_ID=$UID -t manhole:v0 .
```

2. Mount volumes and run docker container

```
docker run --rm --gpus all -v /path/to/data/:/home/appuser/workspace/data -v path/to/output/:/home/appuser/workspace/output manhole:v0 python manhole_detector/infer.py -i ./data/test -o ./data/test_pred/
```

