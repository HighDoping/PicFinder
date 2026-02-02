# backend/image_process.py

# -*- coding: utf-8 -*-

import hashlib
import logging
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rapidocr

from backend.resources.label_list import coco, image_net
from backend.yolo import YOLO26, YOLO26Cls

rapidocr_params = {
    "Det.engine_type": rapidocr.EngineType.ONNXRUNTIME,
    "Cls.engine_type": rapidocr.EngineType.ONNXRUNTIME,
    "Rec.engine_type": rapidocr.EngineType.ONNXRUNTIME,
    "EngineConfig.onnxruntime.use_coreml": (
        True if platform.system() == "Darwin" else False
    ),
    "Det.lang_type": rapidocr.LangDet.CH,
    "Det.model_type": rapidocr.ModelType.SERVER,
    "Det.ocr_version": rapidocr.OCRVersion.PPOCRV4,
    "Rec.lang_type": rapidocr.LangRec.CH,
    "Rec.model_type": rapidocr.ModelType.SERVER,
    "Rec.ocr_version": rapidocr.OCRVersion.PPOCRV4,
}

is_nuitka = "__compiled__" in globals()

if is_nuitka or getattr(sys, "frozen", False):
    models_dir = Path(sys.argv[0]).parent / "models"
else:
    models_dir = Path(__file__).resolve().parent.parent / "models"


def get_yolo_cls_model_path(model: str) -> Path | None:
    model_paths = {
        "yolo26n": "yolo26n-cls.onnx",
        "yolo26s": "yolo26s-cls.onnx",
        "yolo26m": "yolo26m-cls.onnx",
        "yolo26l": "yolo26l-cls.onnx",
        "yolo26x": "yolo26x-cls.onnx",
    }
    if model in model_paths:
        return models_dir / model_paths[model]
    return None


def get_yolo_model_path(model: str) -> Path | None:
    model_paths = {
        "yolo26n": "yolo26n.onnx",
        "yolo26s": "yolo26s.onnx",
        "yolo26m": "yolo26m.onnx",
        "yolo26l": "yolo26l.onnx",
        "yolo26x": "yolo26x.onnx",
    }
    if model in model_paths:
        return models_dir / model_paths[model]
    return None


class ImageProcessor:
    def __init__(
        self,
        classification_model="yolo26n",
        classification_threshold=0.7,
        object_detection_model="yolo26n",
        object_detection_dataset=None,
        object_detection_conf_threshold=0.7,
        OCR_model="RapidOCR",
        **kwargs,
    ):
        if object_detection_dataset is None:
            object_detection_dataset = ["COCO"]

        self.cls_model_name = classification_model
        self.cls_threshold = classification_threshold
        self.obj_model_name = object_detection_model
        self.obj_dataset = object_detection_dataset
        self.obj_threshold = object_detection_conf_threshold
        self.ocr_model_name = OCR_model

        self.cls_net = None
        self.obj_net = None
        self.ocr_engine = None

        # 1. Load Classification Model
        if self.cls_model_name != "None":
            path = get_yolo_cls_model_path(self.cls_model_name)
            if path:
                try:
                    self.cls_net = YOLO26Cls(path, conf_thres=self.cls_threshold)
                    logging.info(f"Loaded Classification Model: {self.cls_model_name}")
                except Exception as e:
                    logging.error(f"Failed to load classification model: {e}")

        # 2. Load Object Detection Model
        if self.obj_model_name != "None":
            path = get_yolo_model_path(self.obj_model_name)
            if path:
                try:
                    self.obj_net = YOLO26(path, conf_thres=self.obj_threshold)
                    logging.info(f"Loaded Object Detection Model: {self.obj_model_name}")
                except Exception as e:
                    logging.error(f"Failed to load object detection model: {e}")
            
            # Pre-calculate dataset mappings
            self.obj_datasets_map = {}
            available_datasets = {"COCO": coco}
            for ds_name in self.obj_dataset:
                if ds_name in available_datasets:
                    self.obj_datasets_map[ds_name] = available_datasets[ds_name]

        # 3. Load OCR Engine
        if self.ocr_model_name == "RapidOCR":
            try:
                self.ocr_engine = rapidocr.RapidOCR(params=rapidocr_params)
                logging.info("Loaded RapidOCR Engine")
            except Exception as e:
                logging.error(f"Failed to load RapidOCR: {e}")

    def classify(self, image: np.ndarray):
        if not self.cls_net:
            return []
        
        class_ids, confidence = self.cls_net(image)
        if len(class_ids) == 0:
            return []
        
        class_names = [image_net[class_id][1] for class_id in class_ids]
        result = [
            (class_name, confidence[class_names.index(class_name)])
            for class_name in class_names
        ]
        return result

    def object_detection(self, image: np.ndarray):
        if not self.obj_net or not self.obj_datasets_map:
            return []

        # Run inference once
        _, scores, class_ids = self.obj_net(image)
        if len(class_ids) == 0:
            return []

        result = []
        # Map results to requested datasets (COCO, etc.)
        for _, class_name_list in self.obj_datasets_map.items():
            # Filter IDs that exist in this dataset list
            # Note: This logic assumes the model output IDs correspond to the index in the dataset list provided
            # Standard YOLO output usually matches COCO indices. 
            
            valid_entries = []
            for i, class_id in enumerate(class_ids):
                if class_id < len(class_name_list):
                    name = class_name_list[class_id]
                    score = scores[i]
                    valid_entries.append((name, score))
            
            result.extend(valid_entries)

        return result

    def ocr(self, image: np.ndarray):
        if not self.ocr_engine:
            return []
        
        result = self.ocr_engine(image, use_det=True, use_cls=True, use_rec=True)
        if result is None or len(result) == 0:
            return []
        return result

    def process_image(self, img_path: Path):
        try:
            with open(img_path, "rb") as file:
                img_file = file.read()
                img_hash = hashlib.md5(img_file).hexdigest()

            try:
                img = cv2.imread(img_path.as_posix())
                assert isinstance(img, np.ndarray)
            except Exception as e:
                return {"error": str(e)}

            res_dict = {
                "hash": img_hash,
                "path": img_path.as_posix()
            }

            # Classification
            if self.cls_net:
                cls_start = time.perf_counter()
                res_dict["classification"] = self.classify(img)
                logging.debug(f"Img:{img_path.name}, Cls Time: {time.perf_counter() - cls_start:.4f}s")

            # Object Detection
            if self.obj_net:
                obj_start = time.perf_counter()
                res_dict["object_detection"] = self.object_detection(img)
                logging.debug(f"Img:{img_path.name}, Obj Time: {time.perf_counter() - obj_start:.4f}s")

            # OCR
            if self.ocr_engine:
                ocr_start = time.perf_counter()
                res_dict["OCR"] = self.ocr(img)
                logging.debug(f"Img:{img_path.name}, OCR Time: {time.perf_counter() - ocr_start:.4f}s")

            return res_dict

        except Exception as e:
            logging.error(f"Exception:{e}, Img_path:{img_path}", exc_info=True)
            return {"error": str(e)}

def read_img(img_path, **kwargs):
    processor = ImageProcessor(**kwargs)
    return processor.process_image(img_path)