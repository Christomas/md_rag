import os
import sys
import json
import subprocess
import numpy as np
from src.db_manager import DBManager

def get_query_vector(query: str) -> np.ndarray:
    """
    通过子进程获取向量，避免 torch 和 faiss 冲突。
    """
    cmd = [sys.executable, "-m", "src.embed_tool", query]
    # 设置 PYTHONPATH 确保能找到 src
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"Embedding error: {result.stderr}")
        sys.exit(1)
    
    vec_list = json.loads(result.stdout.strip())
    return np.array(vec_list).astype('float32')

def retrieve(query: str, top_k: int = 5):
    # 1. 获取向量 (在独立进程中)
    query_vec = get_query_vector(query)
    
    # 2. 搜索 (在当前进程中，当前进程没有加载 torch)
    db = DBManager()
    results = db.search(query_vec, top_k=top_k)
    
    # 3. 打印结果
    if not results:
        print("未找到相关法律条款。")
        return

    for i, res in enumerate(results):
        meta = res['meta_info']
        source = meta.get('source', '未知来源')
        path = meta.get('path', '')
        content = res['content']
        score = res['score']
        
        print(f"--- [Result {i+1}] Score: {score:.4f} ---")
        print(f"<<Source: {source} | Path: {path}>>")
        print(content)
        print("-" * 40)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        retrieve(query)
    else:
        print("用法: python3 src/retrieve.py <查询词>")