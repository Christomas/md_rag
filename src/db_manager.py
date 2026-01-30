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
        
        # 检查并添加 embedding 列（BLOB 类型）
        cursor.execute("PRAGMA table_info(chunks)")
        columns = [info[1] for info in cursor.fetchall()]
        if 'embedding' not in columns:
            print("正在升级数据库 Schema: 添加 embedding 列...")
            cursor.execute("ALTER TABLE chunks ADD COLUMN embedding BLOB")
            
        # --- FTS5 全文索引支持 ---
        # 1. 创建 FTS 虚拟表
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content, 
                content='chunks', 
                content_rowid='id'
            )
        """)
        
        # 2. 创建触发器：当 chunks 插入时自动同步
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
              INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
            END;
        """)
        
        # 3. 创建触发器：当 chunks 删除时自动同步
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
              INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, old.content);
            END;
        """)
        
        # 4. 创建触发器：当 chunks 更新时自动同步
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
              INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, old.content);
              INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
            END;
        """)
        # -----------------------
        
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

    def get_all_files(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM files")
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def delete_file_and_chunks(self, file_id: int):
        """
        彻底删除文件及其所有关联切片。
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("BEGIN TRANSACTION")
            cursor.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
            cursor.execute("DELETE FROM files WHERE id = ?", (file_id,))
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

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
    
    def update_chunk_embeddings(self, updates: List[Tuple[int, bytes]]):
        """
        批量更新切片的向量数据。
        updates: list of (id, embedding_bytes)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executemany("UPDATE chunks SET embedding = ? WHERE id = ?", 
                           [(emb, cid) for cid, emb in updates])
        conn.commit()
        conn.close()

    def get_missing_embedding_ids(self) -> List[int]:
        """
        一次性获取所有未计算向量的 ID。
        避免在循环中使用 OFFSET/LIMIT 导致数据库扫描变慢。
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM chunks WHERE embedding IS NULL")
        rows = cursor.fetchall()
        conn.close()
        return [r[0] for r in rows]

    def get_chunks_by_ids(self, ids: List[int]) -> List[Tuple[int, str]]:
        """
        根据 ID 列表批量获取内容。
        """
        if not ids:
            return []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # 动态构建 WHERE id IN (...)
        placeholders = ','.join(['?'] * len(ids))
        cursor.execute(f"SELECT id, content FROM chunks WHERE id IN ({placeholders})", ids)
        rows = cursor.fetchall()
        conn.close()
        
        # 保证返回顺序与输入 id 顺序一致
        result_map = {r[0]: r[1] for r in rows}
        ordered_results = []
        for i in ids:
            if i in result_map:
                ordered_results.append((i, result_map[i]))
        return ordered_results
    
    def count_chunks_without_embedding(self) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def rebuild_fts(self):
        """
        全量重建 FTS 索引（适用于初次启用 FTS 或数据不一致时）。
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        print("正在重建全文搜索索引 (FTS)...")
        # 清空并重新填充
        cursor.execute("DELETE FROM chunks_fts")
        cursor.execute("INSERT INTO chunks_fts(rowid, content) SELECT id, content FROM chunks")
        conn.commit()
        conn.close()
        print("FTS 索引重建完成。")

    def search_keyword(self, query: str, top_k: int = 50) -> List[Dict]:
        """
        使用 FTS5 进行关键词检索 (BM25)。
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 法律检索更适合短语匹配
        # 如果 query 包含空格，我们可以拆分关键词
        try:
            # 这里的 rank 是 SQLite 内部的 BM25 分数，越小（负数）越相关
            cursor.execute("""
                SELECT c.*, f.filepath, fts.rank
                FROM chunks c
                JOIN files f ON c.file_id = f.id
                JOIN chunks_fts fts ON c.id = fts.rowid
                WHERE chunks_fts MATCH ? 
                ORDER BY rank 
                LIMIT ?
            """, (query, top_k))
            
            rows = cursor.fetchall()
            results = []
            for r in rows:
                res = dict(r)
                # 转换 rank 到 0-1 范围的分数以便合并 (这里用一个简单的负对数映射)
                # 或者直接用 1.0 - (rank / min_rank)
                res['score'] = 1.0 # FTS 结果作为强相关基准，先给满分 1.0
                res['meta_info'] = json.loads(res['meta_info'])
                results.append(res)
            return results
            
        except sqlite3.OperationalError as e:
            print(f"FTS Search Error: {e}")
            return []
        finally:
            conn.close()

    def get_all_vectors(self) -> Tuple[List[int], np.ndarray]:
        """
        获取所有已计算的向量用于构建索引。
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # 只取有向量的数据
        cursor.execute("SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL")
        
        ids = []
        vectors = []
        
        # 逐行读取以节省内存
        while True:
            rows = cursor.fetchmany(10000)
            if not rows:
                break
            for r in rows:
                ids.append(r[0])
                # 假设存储的是 float32 的 bytes
                vectors.append(np.frombuffer(r[1], dtype='float32'))
                
        conn.close()
        if not vectors:
            return [], np.empty((0, self.vector_dim), dtype='float32')
            
        return ids, np.vstack(vectors)

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
            # 联查 files 表获取 filepath，用于后续的权重重排
            cursor.execute("""
                SELECT c.*, f.filepath 
                FROM chunks c
                JOIN files f ON c.file_id = f.id
                WHERE c.id = ?
            """, (int(idx),))
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
