import os
# 关键修复：在导入任何科学计算库之前，强制禁用所有并行化
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 解决 macOS 特有的 OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import hashlib
import sys
import gc
import glob
import shutil
import numpy as np
import sqlite3
import faiss
from src.parser import LegalDocParser
from src.embedding import EmbeddingEngine
from src.db_manager import DBManager

# 配置
BATCH_SIZE = 64         # 降低 Batch Size 以进一步减少内存压力
TEMP_INDEX_DIR = "data/temp_indices"

def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def build_vectors_stream(db_manager, engine):
    """
    流式构建向量：
    1. 从 DB 分页读文本
    2. 计算向量
    3. 写入临时 numpy 文件
    4. 最后构建 FAISS
    """
    os.makedirs(TEMP_INDEX_DIR, exist_ok=True)
    vec_file = os.path.join(TEMP_INDEX_DIR, "vectors.npy")
    id_file = os.path.join(TEMP_INDEX_DIR, "ids.npy")
    
    conn = sqlite3.connect(db_manager.db_path)
    total_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"总数据量: {total_count}")
    
    if total_count == 0:
        return

    # 准备 memmap 文件 (磁盘映射内存)
    fp_vec = np.memmap(vec_file, dtype='float32', mode='w+', shape=(total_count, 1024))
    fp_ids = np.memmap(id_file, dtype='int64', mode='w+', shape=(total_count,))
    
    cursor = conn.execute("SELECT id, content FROM chunks")
    
    processed = 0
    while True:
        rows = cursor.fetchmany(BATCH_SIZE)
        if not rows:
            break
            
        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        
        # 计算向量
        embeddings = engine.embed_documents(texts, batch_size=len(texts), show_progress=False)
        
        # 写入 memmap
        end = processed + len(rows)
        fp_vec[processed:end] = embeddings
        fp_ids[processed:end] = ids
        
        processed = end
        
        # 强制刷盘 & GC
        if processed % 1000 < BATCH_SIZE:
            fp_vec.flush()
            fp_ids.flush()
            gc.collect()
            print(f"进度: {processed}/{total_count} ({(processed/total_count)*100:.1f}%)")
            sys.stdout.flush()
            
    conn.close()
    fp_vec.flush()
    fp_ids.flush()
    print("向量计算完成，开始构建最终索引...")
    
    # 构建索引
    # 此时我们只需要读 memmap
    index = faiss.IndexFlatIP(1024)
    index = faiss.IndexIDMap(index)
    
    # 批量添加 (FAISS 添加也是分块的，不会一次性读入)
    CHUNK_SIZE = 10000
    for i in range(0, total_count, CHUNK_SIZE):
        end = min(i + CHUNK_SIZE, total_count)
        # 读入一小块 RAM
        v_chunk = np.array(fp_vec[i:end]) 
        i_chunk = np.array(fp_ids[i:end])
        index.add_with_ids(v_chunk, i_chunk)
        print(f"索引构建: {end}/{total_count}")
        sys.stdout.flush()
        
    db_manager.index = index
    db_manager.save_index()
    
    # 清理临时文件
    try:
        shutil.rmtree(TEMP_INDEX_DIR)
    except:
        pass
        
    print("全部完成！")

def build_pipeline(data_dir: str, db_manager: DBManager, engine: EmbeddingEngine):
    parser = LegalDocParser()
    
    print(f"扫描目录: {data_dir}...")
    files_processed = 0
    
    # 1. 扫描文件并入库 (不计算向量，只存文本)
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if not file.endswith('.md'): continue
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, data_dir)
            file_hash = get_file_hash(filepath)
            
            existing = db_manager.get_file_by_path(rel_path)
            if existing and existing['file_hash'] == file_hash:
                continue
            
            # 解析并入库
            if files_processed % 100 == 0:
                print(f"解析中... 已处理 {files_processed} 新文件")
                sys.stdout.flush()
                
            db_manager.delete_file_chunks(existing['id'] if existing else -1)
            title = os.path.basename(filepath).replace('.md', '')
            file_id = db_manager.add_file(rel_path, file_hash, title)
            chunks = parser.parse(filepath)
            if chunks:
                db_manager.insert_chunks(file_id, chunks)
            files_processed += 1

    print(f"新文件解析完成。开始检查索引状态...")
    
    # 2. 检查索引
    # 如果是增量更新，我们这里采用了简单的策略：只要有更新，就全量重构索引。
    # 对于 10万 级别数据，全量重构比维护增量索引更稳健。
    if files_processed > 0 or db_manager.index is None or db_manager.index.ntotal == 0:
        build_vectors_stream(db_manager, engine)
    else:
        print("索引已是最新。")

if __name__ == "__main__":
    data_dir = "md_vault"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    db = DBManager()
    # 强制 CPU 并开启低内存模式
    engine = EmbeddingEngine(device="cpu") 
    
    build_pipeline(data_dir, db, engine)