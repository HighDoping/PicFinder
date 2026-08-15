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
    embedding float[512]
);
"""

# Delete existing embedding for a picture
DELETE_EMBEDDING_SQL = """
DELETE FROM vec_pictures WHERE rowid = ?;
"""

# Insert embedding for a picture
INSERT_EMBEDDING_SQL = """
INSERT INTO vec_pictures(rowid, embedding) VALUES (?, ?);
"""

# Search by embedding similarity with threshold
SEARCH_BY_EMBEDDING_SQL = """
SELECT p.* FROM pictures p
WHERE p.id IN (
    SELECT rowid FROM vec_pictures
    WHERE vec_distance_cosine(embedding, ?) < ?
)
ORDER BY vec_distance_cosine(
    (SELECT embedding FROM vec_pictures WHERE rowid = p.id),
    ?
);
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
                self.conn.execute(VEC_TABLE_SQL)
                logging.debug("Vector table initialized")
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
        self.conn.execute(REMOVE_SQL, (path,))
        self.conn.commit()

    def insert_embedding(self, picture_id: int, embedding: np.ndarray):
        """Insert or update CLIP embedding for a picture."""
        if not self.vec_available:
            logging.warning("Cannot insert embedding: sqlite-vec not available")
            return

        try:
            # sqlite-vec accepts numpy arrays directly
            # First delete any existing embedding for this picture
            self.conn.execute(DELETE_EMBEDDING_SQL, (picture_id,))
            # Then insert the new embedding
            self.conn.execute(
                INSERT_EMBEDDING_SQL, (picture_id, embedding.astype(np.float32))
            )
            self.conn.commit()
        except Exception as e:
            logging.error(f"Failed to insert embedding for picture {picture_id}: {e}")

    def search_by_embedding(self, query_embedding: np.ndarray, threshold: float = 0.5):
        """Search pictures by embedding similarity.

        Args:
            query_embedding: Query embedding vector (512-dim)
            threshold: Cosine distance threshold (default 0.5)
                      Lower distance = more similar
                      Distance range: [0, 2] where 0 = identical

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

            # Search with threshold (pass query embedding for distance calculation)
            results = self.conn.execute(
                SEARCH_BY_EMBEDDING_SQL, (query_vec, threshold, query_vec)
            ).fetchall()

            logging.debug(
                f"Found {len(results)} similar images (threshold={threshold})"
            )
            return results

        except Exception as e:
            logging.error(f"Failed to search by embedding: {e}")
            return []

    def close(self):
        self.conn.close()
