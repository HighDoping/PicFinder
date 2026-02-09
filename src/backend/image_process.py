# backend/image_process.py

# -*- coding: utf-8 -*-

import concurrent.futures
import hashlib
import logging
import platform
import sys
import time
from pathlib import Path

import cv2
import imohash
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
    "Det.ocr_version": rapidocr.OCRVersion.PPOCRV5,
    "Rec.lang_type": rapidocr.LangRec.CH,
    "Rec.model_type": rapidocr.ModelType.SERVER,
    "Rec.ocr_version": rapidocr.OCRVersion.PPOCRV5,
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
            (class_name, float(confidence[class_names.index(class_name)]))
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
                    score = float(scores[i])
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
            img_hash = imohash.hashfile(img_path, hexdigest=True)

            try:
                img_array = np.fromfile(str(img_path), dtype=np.uint8)
                img = cv2.imdecode(
                    img_array, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH
                )
                if img is None:
                    return {"error": "OpenCV could not decode the image data."}
            except Exception as e:
                return {"error": str(e)}

            res_dict = {
                "hash": img_hash,
                "path": img_path.as_posix()
            }

            def run_cls():
                if self.cls_net:
                    t0 = time.perf_counter()
                    res = self.classify(img)
                    t1 = time.perf_counter()
                    logging.debug(f"Img:{img_path.name}, Cls Time: {t1 - t0:.4f}s")
                    return "classification", res
                return "classification", None

            def run_obj():
                if self.obj_net:
                    t0 = time.perf_counter()
                    res = self.object_detection(img)
                    t1 = time.perf_counter()
                    logging.debug(f"Img:{img_path.name}, Obj Time: {t1 - t0:.4f}s")
                    return "object_detection", res
                return "object_detection", None

            def run_ocr():
                if self.ocr_engine:
                    t0 = time.perf_counter()
                    res = self.ocr(img)
                    t1 = time.perf_counter()
                    logging.debug(f"Img:{img_path.name}, OCR Time: {t1 - t0:.4f}s")
                    return "OCR", res
                return "OCR", None

            # Execute in parallel using a ThreadPool
            # We use 3 workers since there are 3 distinct tasks
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(run_cls),
                    executor.submit(run_obj),
                    executor.submit(run_ocr)
                ]

                # Wait for all to complete and collect results
                for future in concurrent.futures.as_completed(futures):
                    try:
                        key, result = future.result()
                        if result is not None:
                            res_dict[key] = result
                    except Exception as e:
                        logging.error(f"Error in parallel task: {e}")

            return res_dict

        except Exception as e:
            logging.error(f"Exception:{e}, Img_path:{img_path}", exc_info=True)
            return {"error": str(e)}

def read_img(img_path, **kwargs):
    processor = ImageProcessor(**kwargs)
    return processor.process_image(img_path)
