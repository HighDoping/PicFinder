# -*- coding: utf-8 -*-

import hashlib
import logging
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import imohash
from PySide6.QtCore import QObject, Signal

from backend.db_ops import DB
from backend.image_process import ImageProcessor

try:
    from backend.clip_model import CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


class SearchWorker(QObject):
    finished = Signal()
    progress = Signal(int)
    result = Signal(list)

    def __init__(
        self, 
        db_path: Path, 
        query: str,
        enable_CLIP: bool = False,
        clip_model_name: str = "None",
        clip_threshold: float = 0.5
    ):
        super(SearchWorker, self).__init__()
        self.db = DB(db_path)
        self.query = query
        self.enable_CLIP = enable_CLIP
        self.clip_model_name = clip_model_name
        self.clip_threshold = clip_threshold

    def run(self):
        try:
            # Always perform FTS5 search
            fts_results = self.db.search(self.query)
            logging.debug(f"FTS5 search returned {len(fts_results)} results")
            
            # If CLIP is enabled, perform hybrid search
            if self.enable_CLIP and self.clip_model_name != "None" and CLIP_AVAILABLE:
                try:
                    # Load CLIP model and encode query
                    clip_model = CLIPModel()
                    query_embedding = clip_model.encode_text(self.query)
                    
                    # Search by embedding similarity
                    clip_results = self.db.search_by_embedding(
                        query_embedding,
                        threshold=self.clip_threshold
                    )
                    logging.debug(f"CLIP search returned {len(clip_results)} results")
                    
                    # Merge results (union, deduplicated by id)
                    result_dict = {result[0]: result for result in fts_results}
                    for clip_result in clip_results:
                        if clip_result[0] not in result_dict:
                            result_dict[clip_result[0]] = clip_result
                    
                    merged_results = list(result_dict.values())
                    logging.info(
                        f"Hybrid search: {len(fts_results)} FTS5 + "
                        f"{len(clip_results)} CLIP = {len(merged_results)} total"
                    )
                    self.result.emit(merged_results)
                    
                except Exception as e:
                    logging.warning(f"CLIP search failed, using FTS5 results only: {e}")
                    self.result.emit(fts_results)
            else:
                # CLIP not enabled or not available, return FTS5 results only
                self.result.emit(fts_results)
            
            self.db.close()
            self.finished.emit()
        except Exception as e:
            logging.error(e, exc_info=True)
            self.finished.emit()


class IndexWorker(QObject):
    finished = Signal()
    progress = Signal(int, int)

    def __init__(self, folder_path: Path, **kwargs):
        super(IndexWorker, self).__init__()
        self.folder = folder_path
        self.kwargs = kwargs
        self.stopped = False
        self.processor = ImageProcessor(**self.kwargs)
        self.parallel_workers = self.kwargs.get("parallel", 1)

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
                OCR_model=self.kwargs["OCR_model"],
                CLIP_model=self.kwargs.get("CLIP_model", "None"),
                full_update=self.kwargs["FullUpdate"],
            )

            self.read_folder(self.folder)

            self.full_finished()

        except Exception as e:
            logging.error(e, exc_info=True)
            self.finished.emit()

    def read_folder(self, folder_path: Path):
        self.remove_deleted_files(folder_path)
        file_list = self.sync_file_list(folder_path)
        self.file_list = list(file_list)
        self.total_files = len(self.file_list)
        logging.info(f"Indexing {self.total_files} files")

        if not self.processor:
            logging.error("ImageProcessor not initialized")
            return

        pending_futures = set()
        processed_count = 0

        # Create the thread pool
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            for file in self.file_list:
                if self.stopped:
                    break

                # 1. Submit a new task
                future = executor.submit(self.processor.process_image, file)
                pending_futures.add(future)

                # 2. If pool is full, wait for at least one to finish
                if len(pending_futures) >= self.parallel_workers:
                    # Blocks until at least one future is done
                    done, pending_futures = wait(
                        pending_futures, return_when=FIRST_COMPLETED
                    )

                    # Process the results of the finished tasks
                    for f in done:
                        try:
                            result = f.result()
                            self.save_to_db(result)
                        except Exception as e:
                            logging.error(f"Worker exception: {e}")

                        processed_count += 1
                        self.progress.emit(processed_count, self.total_files)

            # 3. Process any remaining tasks after the loop finishes (or if stopped)
            for f in pending_futures:
                if self.stopped:
                    f.cancel()
                    continue
                try:
                    result = f.result()
                    self.save_to_db(result)
                except Exception as e:
                    logging.error(f"Worker exception: {e}")

                processed_count += 1
                self.progress.emit(processed_count, self.total_files)

        logging.info("Indexing completed")
        self.progress.emit(self.total_files, self.total_files)

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

        # Insert main record and get picture ID
        picture_id = self.db.insert(
            result["hash"],
            rel_path,
            classification,
            classification_confidence_avg,
            object,
            object_confidence_avg,
            OCR,
            ocr_confidence_avg,
        )
        
        # Insert CLIP embedding if available
        if "clip_embedding" in result and result["clip_embedding"] is not None:
            try:
                self.db.insert_embedding(picture_id, result["clip_embedding"])
                logging.debug(f"Saved CLIP embedding for {rel_path}")
            except Exception as e:
                logging.error(f"Failed to save CLIP embedding for {rel_path}: {e}")

    def full_finished(self):
        self.db.close()
        self.finished.emit()

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
                        existing_hash = imohash.hashfile(file,hexdigest=True)
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
