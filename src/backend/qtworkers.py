# -*- coding: utf-8 -*-

import hashlib
import logging
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from backend.db_ops import DB
from backend.image_process import ReadImgWorker


class SearchWorker(QObject):
    finished = Signal()
    progress = Signal(int)
    result = Signal(list)

    def __init__(self, db_path: Path, query: str):
        super(SearchWorker, self).__init__()
        self.db = DB(db_path)
        self.query = query

    def run(self):
        try:
            result = self.db.search(self.query)
            self.db.close()
            self.result.emit(result)
            self.finished.emit()
        except Exception as e:
            logging.error(e, exc_info=True)
            self.finished.emit()


class IndexWorker(QObject):
    finished = Signal()
    progress = Signal(str)

    def __init__(self, folder_path: Path, **kwargs):
        super(IndexWorker, self).__init__()
        self.folder = folder_path
        self.kwargs = kwargs

        self.batch_size = kwargs["batch_size"]
        self.index = 0

    def run(self):
        try:
            db_path = self.folder / "PicFinder.db"
            self.db = DB(db_path)

            self.db.add_history(
                classification_model=self.kwargs["classification_model"],
                classification_threshold=self.kwargs["classification_threshold"],
                object_detection_model=self.kwargs["object_detection_model"],
                object_detection_dataset=self.kwargs["object_detection_dataset"],
                object_detection_confidence=self.kwargs[
                    "object_detection_conf_threshold"
                ],
                object_detection_iou=self.kwargs["object_detection_iou_threshold"],
                OCR_model=self.kwargs["OCR_model"],
                full_update=self.kwargs["FullUpdate"],
            )

            self.read_folder(self.folder)
        except Exception as e:
            logging.error(e, exc_info=True)
            self.finished.emit()

    def save_to_db(self, result: dict):

        if "error" in result.keys():
            return

        rel_path = Path(result["path"]).relative_to(self.folder).as_posix()

        try:
            classification, classification_confidence_avg = self.combine_classification(
                result["classification"]
            )
        except KeyError:
            classification = ""
            classification_confidence_avg = 0
        try:
            object, object_confidence_avg = self.combine_object_detection(
                result["object_detection"]
            )
        except KeyError:
            object = ""
            object_confidence_avg = 0
        try:
            OCR, ocr_confidence_avg = self.combine_ocr(result["OCR"])
        except KeyError:
            OCR = ""
            ocr_confidence_avg = 0

        self.db.insert(
            result["hash"],
            rel_path,
            classification,
            classification_confidence_avg,
            object,
            object_confidence_avg,
            OCR,
            ocr_confidence_avg,
        )

    def read_folder(self, folder_path: Path):

        self.remove_deleted_files(folder_path)
        file_list = self.sync_file_list(folder_path)
        # from generator to list
        self.file_list = list(file_list)
        self.total_files = len(self.file_list)
        self.kwargs["total_files"] = self.total_files

        logging.info(f"Indexing {self.total_files} files")

        self.kwargs["finished_files"] = self.index

        self.run_img_worker(
            self.file_list[self.index : self.index + self.batch_size], **self.kwargs
        )

    def read_folder_results(self, results: list):
        for result in results:
            self.save_to_db(result)

    def img_worker_finished(self):
        self.read_img_worker_thread.quit()
        self.read_img_worker_thread.wait()

        if self.index < self.total_files:
            self.index += self.batch_size
            self.kwargs["finished_files"] = self.index
            batch = self.file_list[self.index : self.index + self.batch_size]
            self.run_img_worker(batch, **self.kwargs)
        else:
            self.full_finished()

    def run_img_worker(self, file_list: list, **kwargs):

        self.read_img_worker = ReadImgWorker(file_list, **kwargs)
        self.read_img_worker_thread = QThread()
        self.read_img_worker.moveToThread(self.read_img_worker_thread)
        self.read_img_worker_thread.started.connect(self.read_img_worker.run)
        self.read_img_worker.progress.connect(self.progress_process)
        self.read_img_worker.results.connect(self.read_folder_results)
        self.read_img_worker.finished.connect(self.img_worker_finished)
        self.read_img_worker.finished.connect(self.read_img_worker_thread.quit)
        self.read_img_worker.finished.connect(self.read_img_worker.deleteLater)
        self.read_img_worker_thread.finished.connect(
            self.read_img_worker_thread.deleteLater
        )
        self.read_img_worker_thread.start()

    def full_finished(self):
        self.db.close()
        self.finished.emit()

    def progress_process(self, progress):
        logging.debug(f"Progress: {progress}")
        self.progress.emit(progress)

    def sync_file_list(self, folder_path: Path):
        supported_suffix = [
            ".bmp",
            ".dib",
            ".jpeg",
            ".jpg",
            ".jpe",
            ".jp2",
            ".png",
            ".webp",
            ".avif",
            ".pbm",
            ".pgm",
            ".ppm",
            ".pxm",
            ".pnm",
            ".pfm",
            ".sr",
            ".ras",
            ".tiff",
            ".tif",
            ".exr",
            ".hdr",
            ".pic",
        ]

        existing_entries = self.db.fetch_all()

        for file in folder_path.rglob("*"):
            if file.is_file() and file.suffix.lower() in supported_suffix:
                if self.kwargs["FullUpdate"]:
                    yield file
                else:
                    rel_path = file.relative_to(folder_path).as_posix()
                    if rel_path in existing_entries.keys():
                        existing_hash = hashlib.md5(file.read_bytes()).hexdigest()
                        if existing_hash == existing_entries[rel_path]:
                            continue
                        else:
                            yield file
                    else:
                        yield file

    def remove_deleted_files(self, folder_path: Path):
        existing_entries = self.db.fetch_all()
        for path in existing_entries.keys():
            if not (folder_path / path).exists():
                logging.info(f"Removing {path} from database")
                self.db.remove(path)

    def combine_classification(self, classification_list):
        if classification_list is None or classification_list == []:
            classification = ""
            classification_confidence_avg = 0
        else:
            classification = " ".join([res[0] for res in classification_list])
            classification_confidence_list = [res[1] for res in classification_list]
            classification_confidence_avg = sum(
                classification_confidence_list  # type: ignore
            ) / len(classification_confidence_list)
        return classification, classification_confidence_avg

    def combine_object_detection(self, object_detection_list):
        if object_detection_list is None or object_detection_list == []:
            object = ""
            object_confidence_avg = 0
        else:
            obj_list = []
            for res in object_detection_list:
                if isinstance(res[0], list):
                    obj_list.append(res[0][1])
                else:
                    obj_list.append(res[0])
            object = " ".join(obj_list)
            object_confidence_list = [res[1] for res in object_detection_list]
            object_confidence_avg = sum(object_confidence_list) / len(  # type: ignore
                object_confidence_list
            )
        return object, object_confidence_avg

    def combine_ocr(self, ocr_list):
        if ocr_list is None or ocr_list == []:
            OCR = ""
            ocr_confidence_avg = 0
        else:
            OCR = " ".join(ocr_list.txts)
            ocr_confidence_list = ocr_list.scores
            ocr_confidence_avg = sum(ocr_confidence_list) / len(ocr_confidence_list)
        return OCR, ocr_confidence_avg
