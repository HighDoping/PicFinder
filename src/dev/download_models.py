import argparse
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

if __name__ == "__main__":
    # get full or minimal models
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Download full models")
    args = parser.parse_args()
    if args.full:
        model_names = [
            "yolo26n.pt",
            "yolo26s.pt",
            "yolo26m.pt",
            "yolo26l.pt",
            "yolo26x.pt",
            "yolo26n-cls.pt",
            "yolo26s-cls.pt",
            "yolo26m-cls.pt",
            "yolo26l-cls.pt",
            "yolo26x-cls.pt",
        ]
    else:
        model_names = [
            "yolo26n.pt",
            "yolo26n-cls.pt",
        ]
    os.makedirs(Path(__file__).parent.parent / "models", exist_ok=True)
    for model_name in model_names:
        model = YOLO(model_name)  # load a pretrained model (recommended for training)
        path = model.export(
            format="onnx",
            dynamic=False,
            end2end=True,
        )  # export the model to ONNX format.
        shutil.move(
            path,
            f"{Path(__file__).parent.parent}/models/{model_name.replace('.pt','.onnx')}",
        )
