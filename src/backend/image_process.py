# -*- coding: utf-8 -*-

import hashlib
import importlib.util
import logging
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rapidocr
from PySide6.QtCore import QObject, QThread, Signal

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

from backend.resources.label_list import coco, image_net, open_images_v7
from backend.yolo import YOLO26, YOLO26Cls

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


def classify(image: np.ndarray, model: str, threshold: float = 0.7):
    yolo26_path = get_yolo_cls_model_path(model)
    if not yolo26_path:
        return []

    yolo_cls = YOLO26Cls(yolo26_path, conf_thres=threshold)
    class_ids, confidence = yolo_cls(image)
    if len(class_ids) == 0:
        return []
    class_names = [image_net[class_id][1] for class_id in class_ids]
    result = [
        (class_name, confidence[class_names.index(class_name)])
        for class_name in class_names
    ]
    return result


class ClassificationWorker(QObject):
    finished = Signal()
    progress = Signal(str)
    result = Signal(list)

    def __init__(
        self,
        image_list: list[Path],
        classification_model: str,
        classification_threshold: float,
        **kwargs,
    ):
        super(ClassificationWorker, self).__init__()
        self.image_list = image_list
        self.model = classification_model
        self.threshold = classification_threshold
        self.kwargs = kwargs

    def run(self):
        try:
            results = self.classify_batch(self.image_list, self.model, self.threshold)
            self.result.emit(results)
            self.finished.emit()
        except Exception as e:
            logging.error(e, exc_info=True)
            self.finished.emit()

    def classify_batch(
        self, images: list[np.ndarray], model: str, threshold: float = 0.7
    ):
        yolo26_path = get_yolo_cls_model_path(model)
        if not yolo26_path:
            return [[] for _ in images]

        yolo_cls = YOLO26Cls(yolo26_path, conf_thres=threshold)

        total_images = self.kwargs["total_files"]
        finished_files = self.kwargs["finished_files"]

        results = []
        for i, image in enumerate(images):
            progress = (
                f"Classification progress: {i + 1 + finished_files}/{total_images}"
            )
            self.progress.emit(progress)

            class_ids, confidence = yolo_cls(image)
            if len(class_ids) == 0:
                results.append([])
                continue
            class_names = [image_net[class_id][1] for class_id in class_ids]
            result = [
                (class_name, confidence[class_names.index(class_name)])
                for class_name in class_names
            ]
            results.append(result)

        return results


def object_detection(
    image: np.ndarray,
    model: str,
    dataset: list[str],
    conf_threshold: float = 0.7,
):

    def YOLO_process(yolo26_path, conf_threshold, image, class_name_list):
        yolo = YOLO26(yolo26_path, conf_threshold)
        _, scores, class_ids = yolo(image)
        if not class_ids:
            return []
        class_names = [class_name_list[class_id] for class_id in class_ids]
        return [
            (class_name, scores[class_names.index(class_name)])
            for class_name in class_names
        ]

    yolo_path = get_yolo_model_path(model)
    if not yolo_path:
        return []

    datasets = {
        "COCO": coco,
    }

    yolo_paths = []
    class_name_lists = []
    for dataset_name in dataset:
        if dataset_name in datasets:
            if dataset_name == "COCO":
                yolo_paths.append(yolo_path)
            class_name_lists.append(datasets[dataset_name])

    result = []
    for yolo26_path, class_name_list in zip(yolo_paths, class_name_lists):
        result.extend(YOLO_process(yolo26_path, conf_threshold, image, class_name_list))

    return result


class ObjectDetectionWorker(QObject):
    finished = Signal()
    progress = Signal(str)
    result = Signal(list)

    def __init__(
        self,
        image_list: list[Path],
        object_detection_model: str,
        object_detection_dataset: list[str],
        object_detection_conf_threshold: float,
        **kwargs,
    ):
        super(ObjectDetectionWorker, self).__init__()
        self.image_list = image_list
        self.model = object_detection_model
        self.dataset = object_detection_dataset
        self.conf_threshold = object_detection_conf_threshold
        self.kwargs = kwargs

    def run(self):
        try:
            results = self.object_detection_batch(
                self.image_list,
                self.model,
                self.dataset,
                self.conf_threshold,
            )
            self.result.emit(results)
            self.finished.emit()
        except Exception as e:
            logging.error(e, exc_info=True)
            self.finished.emit()

    def object_detection_batch(
        self,
        images: list[np.ndarray],
        model: str,
        dataset: list[str],
        conf_threshold: float = 0.7,
    ):
        yolo_path = get_yolo_model_path(model)
        if not yolo_path:
            return [[] for _ in images]

        datasets = {
            "COCO": coco,
        }

        yolo_paths = []
        class_name_list_list = []
        for dataset_name in dataset:
            if dataset_name == "COCO":
                yolo_paths.append(yolo_path)
                class_name_list_list.append(datasets[dataset_name])

        results = []
        yolo_list = []
        for i, yolo26_path in enumerate(yolo_paths):
            yolo_list.append(YOLO26(yolo26_path, conf_threshold))

        total_images = self.kwargs["total_files"]
        finished_files = self.kwargs["finished_files"]

        for i, image in enumerate(images):
            progress = (
                f"Object detection progress: {i + 1 + finished_files}/{total_images}"
            )
            self.progress.emit(progress)
            result = []
            for yolo, class_name_list in zip(yolo_list, class_name_list_list):
                _, scores, class_ids = yolo(image)
                if len(class_ids) == 0:
                    continue
                try:
                    class_names = [class_name_list[class_id] for class_id in class_ids]
                except IndexError as e:
                    logging.error(
                        f"Image Index:{i}, class_ids:{class_ids}, Object Detection failed. Error:{e}",
                        exc_info=True,
                    )
                    continue
                result.extend(
                    [
                        (class_name, scores[class_names.index(class_name)])
                        for class_name in class_names
                    ]
                )
            results.append(result)
        return results


def OCR(image: np.ndarray, model: str):
    if model == "RapidOCR":
        engine = rapidocr.RapidOCR(params=rapidocr_params)
        (result,) = engine(image, use_det=True, use_cls=True, use_rec=True)
        if result is None or len(result) == 0:
            return []
        return result
    else:
        return []


class OCRWorker(QObject):
    finished = Signal()
    progress = Signal(str)
    result = Signal(list)

    def __init__(self, image_list: list[np.ndarray], OCR_model: str, **kwargs):
        super(OCRWorker, self).__init__()
        self.image_list = image_list
        self.model = OCR_model
        self.kwargs = kwargs

    def run(self):
        try:
            results = self.OCR_batch(self.image_list, self.model)
            self.result.emit(results)
            self.finished.emit()
        except Exception as e:
            logging.error(e, exc_info=True)
            self.finished.emit()

    def OCR_batch(self, images: list[np.ndarray], model: str):
        if model == "RapidOCR":
            engine = rapidocr.RapidOCR(params=rapidocr_params)
            total_images = self.kwargs["total_files"]
            finished_files = self.kwargs["finished_files"]

            results = []
            for i, image in enumerate(images):
                progress = f"OCR progress: {i + 1 + finished_files}/{total_images}"
                self.progress.emit(progress)
                try:
                    result = engine(image, use_det=True, use_cls=True, use_rec=True)
                except Exception as e:
                    path_list = self.kwargs.get("path_list", [])
                    if len(path_list) > i:
                        logging.error(
                            f"Image: {path_list[i]}, OCR failed. Error:{e}",
                            exc_info=True,
                        )
                    else:
                        logging.error(
                            f"Image Index:{i}, OCR failed. Error:{e}", exc_info=True
                        )
                    results.append([])
                    continue
                if result is None or len(result) == 0:
                    results.append([])
                    continue
                results.append(result)
            return results
        else:
            return [[] for _ in images]


# %%
def read_img(
    img_path: Path,
    classification_model="yolo26n",
    classification_threshold=0.7,
    object_detection_model="yolo26n",
    object_detection_dataset=["COCO"],
    object_detection_conf_threshold=0.7,
    OCR_model="RapidOCR",
    **kwargs,
):
    try:
        with open(img_path, "rb") as file:
            img_file = file.read()
            # get md5 hash of image
            img_hash = hashlib.md5(img_file).hexdigest()

        # read image with cv2
        try:
            img = cv2.imread(img_path.as_posix())
            assert isinstance(img, np.ndarray)
        except Exception as e:
            return {"error": str(e)}

        res_dict = {}
        res_dict["hash"] = img_hash
        res_dict["path"] = img_path.as_posix()

        if classification_model != "None":
            cls_start = time.perf_counter()

            cls_res = classify(img, classification_model, classification_threshold)
            res_dict["classification"] = cls_res

            cls_end = time.perf_counter()
            logging.debug(
                f"Image:{img_path.as_posix()},Classification Time: {cls_end - cls_start}"
            )

        if object_detection_model != "None":
            obj_start = time.perf_counter()

            obj_res = object_detection(
                img,
                object_detection_model,
                object_detection_dataset,
                object_detection_conf_threshold,
            )
            res_dict["object_detection"] = obj_res

            obj_end = time.perf_counter()
            logging.debug(
                f"Image:{img_path.as_posix()},Object Detection Time: {obj_end - obj_start}"
            )

        if OCR_model != "None":
            OCR_start = time.perf_counter()

            OCR_res = OCR(img, OCR_model)
            res_dict["OCR"] = OCR_res

            OCR_end = time.perf_counter()
            logging.debug(
                f"Image:{img_path.as_posix()},OCR Time: {OCR_end - OCR_start}"
            )

        return res_dict
    except Exception as e:
        logging.error(f"Exception:{e},Img_path:{img_path}", exc_info=True)
        return {"error": str(e)}


def read_img_warper(args: tuple):
    path, kwargs = args
    return read_img(path, **kwargs)


class ReadImgWorker(QObject):
    finished = Signal()
    progress = Signal(str)
    results = Signal(list)

    def __init__(self, image_list: list[Path], **kwargs):
        super(ReadImgWorker, self).__init__()
        self.image_list = image_list
        self.kwargs = kwargs
        self.progress_dict = {}
        self.result_list = []
        self.worker_flags = {}
        self.worker_flags["hash"] = False
        self.worker_flags["classification"] = False
        self.worker_flags["object_detection"] = False
        self.worker_flags["OCR"] = False
        self.kwargs["path_list"] = image_list

    def run(self):
        # start hashing and reading images
        self.start_hash_read()

    def start_hash_read(self):
        self.hash_worker = HashReadWorker(self.image_list)
        self.hash_worker_thread = QThread(parent=self)
        self.hash_worker.moveToThread(self.hash_worker_thread)
        self.hash_worker_thread.started.connect(self.hash_worker.run)
        self.hash_worker.hash_result.connect(self.hash_result)
        self.hash_worker.img_result.connect(self.img_result)
        self.hash_worker.finished.connect(self.hash_finished)
        self.hash_worker.progress.connect(self.progress_process)
        self.worker_flags["hash"] = True
        self.hash_worker_thread.start()

    def hash_result(self, hash: list[str]):
        self.hashes = hash

    def img_result(self, img: list[np.ndarray]):
        self.imgs = img

    def hash_finished(self):
        # wait for hash read to finish
        self.hash_worker_thread.quit()
        self.hash_worker_thread.wait()
        self.hash_worker_thread.deleteLater()
        self.worker_flags["hash"] = False

        # start reading images
        if self.kwargs["classification_model"] != "None":
            self.start_classify_read()
        if self.kwargs["object_detection_model"] != "None":
            self.start_obj_read()
        if self.kwargs["OCR_model"] != "None":
            self.start_OCR_read()

    def start_classify_read(self):
        self.classify_worker = ClassificationWorker(self.imgs, **self.kwargs)
        self.classify_worker_thread = QThread(parent=self)
        self.classify_worker.moveToThread(self.classify_worker_thread)
        self.classify_worker_thread.started.connect(self.classify_worker.run)
        self.classify_worker.result.connect(self.classify_result)
        self.classify_worker.finished.connect(self.classify_finished)
        self.classify_worker.progress.connect(self.progress_process)
        self.worker_flags["classification"] = True
        self.classify_worker_thread.start()

    def start_obj_read(self):
        self.obj_worker = ObjectDetectionWorker(self.imgs, **self.kwargs)
        self.obj_worker_thread = QThread(parent=self)
        self.obj_worker.moveToThread(self.obj_worker_thread)
        self.obj_worker_thread.started.connect(self.obj_worker.run)
        self.obj_worker.result.connect(self.obj_result)
        self.obj_worker.finished.connect(self.obj_finished)
        self.obj_worker.progress.connect(self.progress_process)
        self.worker_flags["object_detection"] = True
        self.obj_worker_thread.start()

    def start_OCR_read(self):
        self.OCR_worker = OCRWorker(self.imgs, **self.kwargs)
        self.OCR_worker_thread = QThread(parent=self)
        self.OCR_worker.moveToThread(self.OCR_worker_thread)
        self.OCR_worker_thread.started.connect(self.OCR_worker.run)
        self.OCR_worker.result.connect(self.OCR_result)
        self.OCR_worker.finished.connect(self.OCR_finished)
        self.OCR_worker.progress.connect(self.progress_process)
        self.worker_flags["OCR"] = True
        self.OCR_worker_thread.start()

    def classify_finished(self):
        self.classify_worker_thread.quit()
        self.classify_worker_thread.wait()
        logging.debug("Classification finished")
        self.worker_flags["classification"] = False
        self.check_worker_finished()

    def obj_finished(self):
        self.obj_worker_thread.quit()
        self.obj_worker_thread.wait()
        logging.debug("Object detection finished")
        self.worker_flags["object_detection"] = False
        self.check_worker_finished()

    def OCR_finished(self):
        self.OCR_worker_thread.quit()
        self.OCR_worker_thread.wait()
        logging.debug("OCR finished")
        self.worker_flags["OCR"] = False
        self.check_worker_finished()

    def check_worker_finished(self):
        if (
            self.worker_flags["classification"] == False
            and self.worker_flags["object_detection"] == False
            and self.worker_flags["OCR"] == False
        ):
            self.result_emit()

    def result_emit(self):
        for i, img_path in enumerate(self.image_list):
            result_dict = {}
            result_dict["hash"] = self.hashes[i]
            result_dict["path"] = img_path

            try:
                result_dict["classification"] = self.classify_res[i]
            except AttributeError:
                result_dict["classification"] = []
            except IndexError:
                result_dict["classification"] = []
                logging.error(
                    f"Classification failed for image:{img_path.as_posix()}",
                    exc_info=True,
                )

            try:
                result_dict["object_detection"] = self.obj_res[i]
            except AttributeError:
                result_dict["object_detection"] = []
            except IndexError:
                result_dict["object_detection"] = []
                logging.error(
                    f"Object Detection failed for image:{img_path.as_posix()}",
                    exc_info=True,
                )

            try:
                result_dict["OCR"] = self.OCR_res[i]
            except AttributeError:
                result_dict["OCR"] = []
            except IndexError:
                result_dict["OCR"] = []
                logging.error(
                    f"OCR failed for image:{img_path.as_posix()}",
                    exc_info=True,
                )

            self.result_list.append(result_dict)

        self.results.emit(self.result_list)
        self.finished.emit()

    def classify_result(self, result: list):
        self.classify_res = result

    def obj_result(self, result: list):
        self.obj_res = result

    def OCR_result(self, result: list):
        self.OCR_res = result

    def progress_process(self, progress: str):
        if progress.startswith("Classification"):
            self.progress_dict["Classification"] = progress
        elif progress.startswith("Object detection"):
            self.progress_dict["Object detection"] = progress
        elif progress.startswith("OCR"):
            self.progress_dict["OCR"] = progress

        progress_str = ", ".join(self.progress_dict.values())
        self.progress.emit(progress_str)


class HashReadWorker(QObject):
    finished = Signal()
    progress = Signal(str)
    hash_result = Signal(list)
    img_result = Signal(list)

    def __init__(self, file_paths: list[Path]):
        super(HashReadWorker, self).__init__()
        self.file_paths = file_paths

    def run(self):
        try:
            hash_list = []
            img_list = []
            for i, file_path in enumerate(self.file_paths):
                self.progress.emit(i + 1)
                try:
                    with open(file_path, "rb") as file:
                        file_bytes = file.read()
                        hash = hashlib.md5(file_bytes).hexdigest()
                        try:
                            img = cv2.imdecode(
                                np.frombuffer(file_bytes, np.uint8),
                                cv2.IMREAD_COLOR,
                            )
                            if not isinstance(img, np.ndarray):
                                img = np.zeros((100, 100, 3), dtype=np.uint8)
                                logging.error(
                                    f"Image:{file_path.as_posix()}, cv2 read failed",
                                    exc_info=True,
                                )
                        except Exception as e:
                            img = np.zeros((100, 100, 3), dtype=np.uint8)
                            logging.error(
                                f"Image:{file_path.as_posix()}, cv2 read failed",
                                exc_info=True,
                            )
                    hash_list.append(hash)
                    img_list.append(img)
                except Exception as e:
                    logging.error(e, exc_info=True)
                    continue
            self.hash_result.emit(hash_list)
            self.img_result.emit(img_list)
            self.finished.emit()
        except Exception as e:
            logging.error(e, exc_info=True)
            self.finished.emit()
            self.finished.emit()
