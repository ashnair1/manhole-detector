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

### Evaluate

```python
python manhole-detector/eval.py --ckpt path/to/checkpoint
```

### Inference

```python
python manhole-detector/infer.py -i path/to/img/directory -o path/to/output/directory -c path/to/checkpoint
```

### Docker

You can either build the docker image manually or you could get the pre-built docker image

**Build manually**
```
git clone git@github.com:ashnair1/manhole-detector.git
cd manhole_detector/
docker build --build-arg USER_ID=$UID -t manhole:v1 .
```

**Pull pre-built image**
```
docker pull ash1995/manhole:v1
```


**Mount volumes and run docker container**

Assuming your test folder directory is as follows:
```
.
└── data
    ├── test
    └── test_pred
```
where `test` contains test images and `test_pred` is the directory to save predictions.

```
docker run --rm --gpus all -v /path/to/data/:/home/appuser/workspace/data manhole:v1 python manhole_detector/infer.py -i ./data/test -o ./data/test_pred/
```
**Note**: If you're building an image manually, you'll need to explicitly mount your checkpoint folder as well as the pre-built image comes with a checkpoint. Specify an additional volume mount via `-v path/to/output/:/home/appuser/workspace/output`



