import os
# 关键优化：彻底禁用所有底层并行库，防止 macOS 上的 Segmentation Fault
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import hashlib
import sys
import gc
import shutil
import numpy as np
import sqlite3
import faiss
import time
import torch # 导入以清理显存
from src.parser import LegalDocParser
from src.embedding import EmbeddingEngine
from src.db_manager import DBManager

import argparse

# 默认配置
DEFAULT_FETCH = 64
DEFAULT_BATCH = 8
FETCH_SIZE = DEFAULT_FETCH
COMPUTE_BATCH = DEFAULT_BATCH
TEMP_INDEX_DIR = "data/temp_indices"

def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def build_vectors_incremental(db_manager, engine):
    """
    增量构建向量：
    1. 启动时一次性查出所有缺失向量的 ID 列表。
    2. 遍历 ID 列表，分批获取文本并计算。
    """
    missing_ids = db_manager.get_missing_embedding_ids()
    total_missing = len(missing_ids)
    
    if total_missing == 0:
        print("所有切片均已有向量，跳过计算步骤。")
        return

    print(f"发现 {total_missing} 条数据缺少向量，开始增量计算...")
    print(f"配置: 拉取批次={FETCH_SIZE} | 计算批次={COMPUTE_BATCH} | 设备={engine.model.device}")
    
    start_time = time.time()
    processed = 0
    
    # 将 missing_ids 分成固定大小的批次
    for i in range(0, total_missing, FETCH_SIZE):
        # 检查是否达到本次运行的上限
        if MAX_LIMIT > 0 and processed >= MAX_LIMIT:
            print(f"\n[Info] 已达到本次运行上限 ({MAX_LIMIT} 条)，准备退出并触发重启...")
            break

        batch_ids = missing_ids[i:i + FETCH_SIZE]
        
        # 1. 批量获取文本内容
        rows = db_manager.get_chunks_by_ids(batch_ids)
        if not rows: continue
        
        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        
        # 2. 计算向量
        try:
            embeddings = engine.embed_documents(texts, batch_size=COMPUTE_BATCH, show_progress=False)
        except Exception as e:
            if "MPS backend out of memory" in str(e) or "Invalid buffer size" in str(e):
                print(f"\n[Warning] 显存压力过大，强制清理并降级批次...")
                torch.mps.empty_cache()
                gc.collect()
                embeddings = engine.embed_documents(texts, batch_size=1, show_progress=False)
            else:
                raise e
        
        # 3. 回写数据库
        updates = []
        for cid, emb in zip(ids, embeddings):
            updates.append((cid, emb.astype('float32').tobytes()))
        db_manager.update_chunk_embeddings(updates)
        
        # 4. 进度与资源管理
        processed += len(rows)
        elapsed = time.time() - start_time
        speed = processed / elapsed if elapsed > 0 else 0
        eta = (total_missing - processed) / speed / 60 if speed > 0 else 0
        
        print(f"[计算中] 进度: {processed}/{total_missing} ({(processed/total_missing):.1%}) | 速度: {speed:.1f} docs/s | ETA: {eta:.1f} min", end="\r")
        sys.stdout.flush()
        
        # 彻底释放当前批次内存
        del texts, embeddings, updates, rows, ids, batch_ids
        if processed % 128 == 0:
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    print(f"\n向量计算阶段完成！耗时: {(time.time()-start_time)/60:.1f} 分钟")

def rebuild_faiss_index(db_manager):
    """
    从数据库加载所有向量并重建 Faiss 索引。
    """
    print("正在从数据库加载向量以构建索引...")
    ids, vectors = db_manager.get_all_vectors()
    
    if len(ids) == 0:
        print("没有向量数据，无法构建索引。")
        return

    print(f"加载完成，共 {len(ids)} 条。开始构建 Faiss 索引...")
    index = faiss.IndexFlatIP(1024)
    index = faiss.IndexIDMap(index)
    
    index.add_with_ids(vectors, np.array(ids).astype('int64'))
    
    db_manager.index = index
    db_manager.save_index()
    print("索引构建并保存完成。")

def build_pipeline(data_dir: str, db_manager: DBManager, engine: EmbeddingEngine):
    parser = LegalDocParser()
    print(f"扫描目录: {data_dir}...")
    files_processed = 0
    ignored_dirs = ["temp_indices", ".git"]
    
    pending_files = []
    for root, dirs, files in os.walk(data_dir):
        if any(ignored in root for ignored in ignored_dirs): continue
        for file in files:
            if file.endswith('.md'):
                pending_files.append(os.path.join(root, file))
                
    print(f"找到 {len(pending_files)} 个 Markdown 文件。正在同步数据库文本...")

    # --- 补齐：清理已从磁盘删除的文件 ---
    all_db_files = db_manager.get_all_files() # 需要在 db_manager 中添加此方法
    disk_rel_paths = {os.path.relpath(p, data_dir) for p in pending_files}
    
    deleted_count = 0
    for db_file in all_db_files:
        if db_file['filepath'] not in disk_rel_paths:
            print(f"清理已删除文件: {db_file['filepath']}")
            db_manager.delete_file_and_chunks(db_file['id']) # 需要添加此方法
            deleted_count += 1
    if deleted_count > 0:
        print(f"已清理 {deleted_count} 个失效文件记录。")
    # -----------------------------------

    for filepath in pending_files:
        rel_path = os.path.relpath(filepath, data_dir)
        file_hash = get_file_hash(filepath)
        
        existing = db_manager.get_file_by_path(rel_path)
        if existing and existing['file_hash'] == file_hash:
            continue
        
        # 如果文件变更，先删除旧数据的 chunks (级联删除会导致向量也丢失，符合逻辑)
        if existing:
            db_manager.delete_file_chunks(existing['id'])
            
        title = os.path.basename(filepath).replace('.md', '')
        file_id = db_manager.add_file(rel_path, file_hash, title)
        chunks = parser.parse(filepath)
        
        if chunks:
            # 插入 chunks 时，embedding 列默认为 NULL
            db_manager.insert_chunks(file_id, chunks)
            files_processed += 1
            
        if files_processed % 100 == 0:
            print(f"已处理 {files_processed} 个变更文件...")

    print("数据库文本同步完成。")
    
    # 阶段一：补全缺失的向量
    build_vectors_incremental(db_manager, engine)
    
    # 阶段二：总是重建索引，确保数据一致性
    rebuild_faiss_index(db_manager)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build knowledge graph index")
    parser.add_argument("data_dir", nargs='?', default="md_vault", help="Path to markdown data")
    parser.add_argument("--fetch_size", type=int, default=DEFAULT_FETCH, help="SQL fetch batch size")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH, help="Embedding compute batch size")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cpu"], help="Device to use (mps or cpu)")
    parser.add_argument("--limit", type=int, default=0, help="Max items to process before exiting (0 for no limit)")
    
    args = parser.parse_args()
    
    # Update globals
    FETCH_SIZE = args.fetch_size
    COMPUTE_BATCH = args.batch_size
    MAX_LIMIT = args.limit
    data_dir = args.data_dir

    db = DBManager()

    # 根据参数选择设备
    print(f"初始化模型，使用设备: {args.device}")
    engine = EmbeddingEngine(device=args.device) 

    build_pipeline(data_dir, db, engine)
