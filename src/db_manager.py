import sqlite3
import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple, Optional

class DBManager:
    """
    统一管理 SQLite 关系型数据和 FAISS 向量索引。
    """
    def __init__(self, db_path: str = "data/knowledge.db", index_path: str = "data/vector.index"):
        self.db_path = db_path
        self.index_path = index_path
        self._init_db()
        self.index = None
        self.vector_dim = 1024 # BGE-M3 的维度

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 文件注册表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT UNIQUE,
                file_hash TEXT,
                title TEXT,
                status TEXT DEFAULT 'active'
            )
        """)
        
        # 切片表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY, -- 这里的 ID 将与 FAISS ID 严格对应
                file_id INTEGER,
                content TEXT,
                meta_info TEXT, -- JSON 格式
                FOREIGN KEY(file_id) REFERENCES files(id)
            )
        """)
        
        conn.commit()
        conn.close()

    def get_file_by_path(self, filepath: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM files WHERE filepath = ?", (filepath,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def add_file(self, filepath: str, file_hash: str, title: str) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO files (filepath, file_hash, title) VALUES (?, ?, ?)",
            (filepath, file_hash, title)
        )
        file_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return file_id

    def delete_file_chunks(self, file_id: int):
        """
        删除一个文件的所有切片。注意：这会导致 FAISS ID 不连续，
        因此通常在增量更新后我们需要全量重建 FAISS 索引。
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        conn.commit()
        conn.close()

    def insert_chunks(self, file_id: int, chunks: List[Any]):
        """
        插入切片到 SQLite。返回起始 ID。
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        inserted_ids = []
        for chunk in chunks:
            meta_json = json.dumps(chunk.metadata, ensure_ascii=False)
            cursor.execute(
                "INSERT INTO chunks (file_id, content, meta_info) VALUES (?, ?, ?)",
                (file_id, chunk.content, meta_json)
            )
            inserted_ids.append(cursor.lastrowid)
            
        conn.commit()
        conn.close()
        return inserted_ids

    def load_index(self):
        import faiss
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            # 使用 Inner Product (IP) 索引，配合归一化向量即为余弦相似度
            self.index = faiss.IndexFlatIP(self.vector_dim)

    def save_index(self):
        import faiss
        if self.index:
            faiss.write_index(self.index, self.index_path)

    def rebuild_index(self, embeddings: np.ndarray, ids: List[int]):
        """
        全量重建 FAISS 索引。
        """
        import faiss
        self.index = faiss.IndexFlatIP(self.vector_dim)
        if len(ids) > 0:
            # FAISS IndexIDMap 允许我们手动指定 ID
            self.index = faiss.IndexIDMap(self.index)
            self.index.add_with_ids(np.array(embeddings).astype('float32'), np.array(ids).astype('int64'))
        self.save_index()

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        在 FAISS 中搜索并关联 SQLite 中的文本。
        """
        if self.index is None:
            self.load_index()
            
        # 搜索
        query_vector = np.array([query_vector]).astype('float32')
        scores, ids = self.index.search(query_vector, top_k)
        
        results = []
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1: continue
            cursor.execute("SELECT * FROM chunks WHERE id = ?", (int(idx),))
            row = cursor.fetchone()
            if row:
                res = dict(row)
                res['score'] = float(score)
                res['meta_info'] = json.loads(res['meta_info'])
                results.append(res)
        
        conn.close()
        return results

    def get_all_embeddings_data(self) -> Tuple[List[int], List[str]]:
        """
        从数据库获取所有切片及其内容（用于重建索引时重新计算向量，或者如果 DB 存了 BLOB 也可以直接取）。
        目前我们简单起见，返回 ID 和内容。
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, content FROM chunks")
        rows = cursor.fetchall()
        conn.close()
        
        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        return ids, texts
