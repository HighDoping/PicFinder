# -*- coding: utf-8 -*-

import hashlib
import logging
from multiprocessing import Pool
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from backend.db_ops import DB
from backend.image_process import read_img_warper


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

    def run(self):
        try:
            db_path = self.folder / "PicFinder.db"
            self.db = DB(db_path)

            results = self.read_folder(self.folder, **self.kwargs)
            self.db.add_history(
                classification_model=self.kwargs["classification_model"],
                classification_threshold=self.kwargs["classification_threshold"],
                object_detection_model=self.kwargs["object_detection_model"],
                object_detection_confidence=self.kwargs[
                    "object_detection_conf_threshold"
                ],
                object_detection_iou=self.kwargs["object_detection_iou_threshold"],
                OCR_model=self.kwargs["OCR_model"],
                full_update=self.kwargs["FullUpdate"],
            )
            for result in results:
                self.save_to_db(result)
            self.db.close()
            self.finished.emit()
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

    def read_folder(self, folder_path: Path, **kwargs):

        self.remove_deleted_files(folder_path)
        file_list = self.sync_file_list(folder_path)
        # from generator to list
        file_list = list(file_list)

        input_list = [(file, kwargs) for file in file_list]

        logging.info(f"Indexing {len(input_list)} files")

        with Pool(processes=4) as p:
            total_files = len(input_list)
            for i, result in enumerate(
                p.imap(read_img_warper, input_list, chunksize=1)
            ):
                self.progress.emit(f"{i + 1}/{total_files}")
                yield result

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
            object = " ".join([res[0] for res in object_detection_list])
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
            OCR = " ".join([res[0] for res in ocr_list])
            ocr_confidence_list = [res[1] for res in ocr_list]
            ocr_confidence_avg = sum(ocr_confidence_list) / len(ocr_confidence_list)
        return OCR, ocr_confidence_avg
