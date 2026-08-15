# %%
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import sqlite_vec

TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pictures (
    id INTEGER PRIMARY KEY,
    hash TEXT,
    path TEXT UNIQUE,
    classification TEXT,
    classification_confidence REAL,
    object TEXT,
    object_confidence REAL,
    OCR TEXT,
    ocr_confidence REAL,
    created_at INTEGER DEFAULT (strftime('%s', 'now'))
);
"""

HISTORY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY,
    classification_model TEXT,
    classification_threshold REAL,
    object_detection_model TEXT,
    object_detection_dataset TEXT,
    object_detection_confidence REAL,
    OCR_model TEXT,
    CLIP_model TEXT,
    Full_update BOOLEAN,
    indexed_at INTEGER DEFAULT (strftime('%s', 'now'))
);
"""

SEARCH_TABLE_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS pictures_fts USING fts5(
    id,
    classification,
    object,
    OCR,
    content = pictures,
    content_rowid = id,
    tokenize = "simple"
);
"""
TRIGGER_SQL = """
CREATE TRIGGER IF NOT EXISTS pictures_ai AFTER INSERT ON pictures BEGIN
    INSERT INTO pictures_fts(rowid, classification, object, OCR) VALUES (new.id, new.classification, new.object, new.OCR);
END;
CREATE TRIGGER IF NOT EXISTS pictures_ad AFTER DELETE ON pictures BEGIN
    INSERT INTO pictures_fts(pictures_fts, rowid, classification, object, OCR) VALUES('delete', old.id, old.classification, old.object, old.OCR);
END;
CREATE TRIGGER IF NOT EXISTS pictures_au AFTER UPDATE ON pictures BEGIN
    INSERT INTO pictures_fts(pictures_fts, rowid, classification, object, OCR) VALUES('delete', old.id, old.classification, old.object, old.OCR);
    INSERT INTO pictures_fts(rowid, classification, object, OCR) VALUES (new.id, new.classification, new.object, new.OCR);
END;
"""
SEARCH_SIMPLE_SQL = """
SELECT * FROM pictures WHERE id IN (SELECT id FROM pictures_fts WHERE pictures_fts MATCH simple_query(?) ORDER BY rank);
"""
INIT_JIEBA_SQL = """
SELECT jieba_dict(?);
"""
SEARCH_JIEBA_SQL = """
SELECT * FROM pictures WHERE id IN (SELECT id FROM pictures_fts WHERE pictures_fts MATCH jieba_query(?) ORDER BY rank);
"""
# insert, update if path exists
INSERT_SQL = """
INSERT INTO pictures (hash, path, classification, classification_confidence, object, object_confidence, OCR, ocr_confidence)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(path) DO UPDATE SET
    classification = excluded.classification,
    classification_confidence = excluded.classification_confidence,
    object = excluded.object,
    object_confidence = excluded.object_confidence,
    OCR = excluded.OCR,
    ocr_confidence = excluded.ocr_confidence;
"""

FETCH_SQL = """
SELECT * FROM pictures WHERE path = ?;
"""

REMOVE_SQL = """
DELETE FROM pictures WHERE path = ?;
"""

HISTORY_INSERT_SQL = """
INSERT INTO history (classification_model, classification_threshold, object_detection_model,object_detection_dataset, object_detection_confidence, OCR_model, CLIP_model, Full_update)
VALUES (?, ?, ?, ?, ?, ?, ?, ?);
"""

RETURN_ALL_SQL = """
SELECT * FROM pictures;
"""

# Vector table for CLIP embeddings
VEC_TABLE_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS vec_pictures USING vec0(
    embedding float[512],
    +path TEXT
);
"""

# `path` is the application-level unique key. sqlite-vec keeps an internal rowid,
# but it is intentionally not used to relate vectors to pictures.
DELETE_EMBEDDING_SQL = """
DELETE FROM vec_pictures WHERE path = ?;
"""

INSERT_EMBEDDING_SQL = """
INSERT INTO vec_pictures(embedding, path) VALUES (?, ?);
"""

# Search by minimum cosine similarity.
SEARCH_BY_EMBEDDING_SQL = """
SELECT p.*, 1.0 - vec_distance_cosine(v.embedding, ?) AS clip_similarity
FROM pictures p
JOIN vec_pictures v ON v.path = p.path
WHERE 1.0 - vec_distance_cosine(v.embedding, ?) >= ?
ORDER BY clip_similarity DESC;
"""

# prepare for multi-platform
if sys.platform == "win32":
    lib_dir_name = "libsimple-windows-x64"
    extention_name = "simple"
elif sys.platform == "linux":
    lib_dir_name = "libsimple-linux-ubuntu-latest"
    extention_name = "libsimple"
elif sys.platform == "darwin":
    lib_dir_name = "libsimple-osx-x64"
    extention_name = "libsimple"
else:
    lib_dir_name = "libsimple-windows-x64"
    extention_name = "simple"

is_nuitka = "__compiled__" in globals()

if is_nuitka or getattr(sys, "frozen", False):
    lib_dir = Path(sys.argv[0]).parent / "lib" / "backend" / "libsimple" / lib_dir_name
else:
    lib_dir = Path(__file__).resolve().parent / "libsimple" / lib_dir_name


class DB:
    def __init__(self, path, jieba=False):
        extention_path = lib_dir / extention_name
        dict_path = lib_dir / "dict"

        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA temp_store = 2;")
        self.conn.enable_load_extension(True)
        self.conn.load_extension(extention_path.as_posix())

        # Load sqlite-vec extension for vector similarity search
        try:
            sqlite_vec.load(self.conn)
            self.vec_available = True
            logging.info("sqlite-vec extension loaded successfully")
        except Exception as e:
            self.vec_available = False
            logging.warning(f"sqlite-vec not available: {e}")

        self.conn.execute(TABLE_SQL)
        self.conn.execute(HISTORY_TABLE_SQL)
        self.conn.execute(SEARCH_TABLE_SQL)
        self.conn.executescript(TRIGGER_SQL)

        # Create vector table if extension is available
        if self.vec_available:
            try:
                existing_columns = {
                    row[1]
                    for row in self.conn.execute("PRAGMA table_info(vec_pictures)")
                }
                if existing_columns and "path" not in existing_columns:
                    logging.warning(
                        "Migrating vector table to path-based keys; existing CLIP "
                        "embeddings will be removed and recreated on the next full index"
                    )
                    self.conn.execute("DROP TABLE vec_pictures")
                self.conn.execute(VEC_TABLE_SQL)
                logging.debug("Path-keyed vector table initialized")
            except Exception as e:
                logging.error(f"Failed to create vector table: {e}")
                self.vec_available = False

        self.jieba = jieba
        if jieba:
            self.init_jieba(dict_path.as_posix())

    def search(self, query):
        # if query is empty, return all
        if not query or query == "":
            return self.conn.execute(RETURN_ALL_SQL).fetchall()
        if self.jieba:
            return self.conn.execute(SEARCH_JIEBA_SQL, (query,)).fetchall()
        else:
            return self.conn.execute(SEARCH_SIMPLE_SQL, (query,)).fetchall()

    def check_hash(self, hash):
        return self.conn.execute(
            "SELECT * FROM pictures WHERE hash = ?", (hash,)
        ).fetchall()

    def init_jieba(self, dict_path):
        self.conn.execute(INIT_JIEBA_SQL, (dict_path,))

    def add_history(
        self,
        classification_model,
        classification_threshold,
        object_detection_model,
        object_detection_dataset,
        object_detection_confidence,
        OCR_model,
        CLIP_model,
        full_update,
    ):
        object_detection_dataset = ",".join(object_detection_dataset)
        self.conn.execute(
            HISTORY_INSERT_SQL,
            (
                classification_model,
                classification_threshold,
                object_detection_model,
                object_detection_dataset,
                object_detection_confidence,
                OCR_model,
                CLIP_model,
                full_update,
            ),
        )
        self.conn.commit()

    def fetch(self, path):
        return self.conn.execute(FETCH_SQL, (path,)).fetchone()

    def fetch_all(self):
        results = self.conn.execute("SELECT * FROM pictures").fetchall()
        # path:hash
        res_dict = {result[2]: result[1] for result in results}
        return res_dict

    def insert(
        self,
        hash,
        path,
        classification,
        classification_confidence,
        object,
        object_confidence,
        OCR,
        ocr_confidence,
    ):
        cursor = self.conn.execute(
            INSERT_SQL,
            (
                hash,
                path,
                classification,
                classification_confidence,
                object,
                object_confidence,
                OCR,
                ocr_confidence,
            ),
        )
        self.conn.commit()
        # Return the ID of the inserted/updated row
        if cursor.lastrowid and cursor.lastrowid > 0:
            return cursor.lastrowid
        else:
            # For ON CONFLICT updates, get the ID from the path
            result = self.conn.execute(
                "SELECT id FROM pictures WHERE path = ?", (path,)
            ).fetchone()
            return result[0] if result else None

    def remove(self, path):
        if self.vec_available:
            self.conn.execute(DELETE_EMBEDDING_SQL, (path,))
        self.conn.execute(REMOVE_SQL, (path,))
        self.conn.commit()

    def insert_embedding(self, path: str, embedding: np.ndarray):
        """Insert or replace a CLIP embedding, identified by picture path."""
        if not self.vec_available:
            logging.warning("Cannot insert embedding: sqlite-vec not available")
            return False

        if not path:
            logging.error("Cannot insert embedding: picture path is empty")
            return False
        if not isinstance(embedding, np.ndarray):
            logging.error(
                "Cannot insert embedding for path %s: expected ndarray, got %s",
                path,
                type(embedding).__name__,
            )
            return False
        if embedding.shape != (512,):
            logging.error(
                "Cannot insert embedding for path %s: expected shape (512,), got %s",
                path,
                embedding.shape,
            )
            return False
        if not np.isfinite(embedding).all():
            logging.error(
                "Cannot insert embedding for path %s: embedding contains non-finite values",
                path,
            )
            return False

        try:
            # sqlite-vec accepts numpy arrays directly
            # sqlite-vec auxiliary columns cannot declare a UNIQUE constraint, so
            # replace by path explicitly before inserting the new vector.
            self.conn.execute(DELETE_EMBEDDING_SQL, (path,))
            logging.debug("Inserting embedding for path %s", path)
            logging.debug(
                f"Embedding shape: {embedding.shape}, dtype: {embedding.dtype}"
            )
            logging.debug(f"Embedding sample: {embedding[:5]}")  # Log first
            self.conn.execute(
                INSERT_EMBEDDING_SQL, (embedding.astype(np.float32), path)
            )
            self.conn.commit()
            logging.debug("Inserted CLIP embedding for path %s", path)
            return True
        except Exception as e:
            logging.error(
                "Failed to insert embedding for path %s: %s",
                path,
                e,
                exc_info=True,
            )
            return False

    def search_by_embedding(
        self, query_embedding: np.ndarray, min_similarity: float = 0.5
    ):
        """Search pictures by embedding similarity.

        Args:
            query_embedding: Query embedding vector (512-dim)
            min_similarity: Minimum inclusive cosine similarity (default 0.5).
                            Higher values require a closer match; 1.0 is an
                            identical normalized embedding.

        Returns:
            List of picture records sorted by similarity
        """
        if not self.vec_available:
            logging.warning("Cannot search by embedding: sqlite-vec not available")
            return []

        try:
            # Check if vector table has any data
            count = self.conn.execute("SELECT COUNT(*) FROM vec_pictures").fetchone()[0]

            if count == 0:
                logging.warning("No embeddings found in database")
                return []

            # sqlite-vec accepts numpy arrays directly
            query_vec = query_embedding.astype(np.float32)

            # Filter using the same cosine similarity shown in result file_info.
            results = self.conn.execute(
                SEARCH_BY_EMBEDDING_SQL, (query_vec, query_vec, min_similarity)
            ).fetchall()

            logging.debug(
                "Found %s images with CLIP similarity >= %.4f",
                len(results),
                min_similarity,
            )
            return results

        except Exception as e:
            logging.error(f"Failed to search by embedding: {e}")
            return []

    def close(self):
        self.conn.close()
