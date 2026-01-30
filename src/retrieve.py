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

def rrf_merge(vector_results, keyword_results, k=60):
    """
    Reciprocal Rank Fusion (RRF) 合并算法。
    """
    scores = {}
    details = {}
    
    # RRF 常数，通常取 60
    c = 60
    
    # 处理向量结果
    for rank, res in enumerate(vector_results):
        doc_id = res['id']
        scores[doc_id] = scores.get(doc_id, 0) + (1.0 / (c + rank + 1))
        details[doc_id] = res

    # 处理关键词结果
    for rank, res in enumerate(keyword_results):
        doc_id = res['id']
        scores[doc_id] = scores.get(doc_id, 0) + (1.0 / (c + rank + 1))
        # 如果都在，保留向量结果的 score 信息，因为 FTS score 只是虚拟的 1.0
        if doc_id not in details:
            details[doc_id] = res

    # 转换为列表并排序
    merged = []
    for doc_id, score in scores.items():
        item = details[doc_id]
        item['rrf_score'] = score
        merged.append(item)
    
    merged.sort(key=lambda x: x['rrf_score'], reverse=True)
    return merged

def retrieve(query: str, top_k: int = 20):
    db = DBManager()
    
    # 1. 并行召回 (Recall Phase)
    print(f"检索中: '{query}' ...")
    
    # Path A: 向量检索
    query_vec = get_query_vector(query)
    vector_candidates = db.search(query_vec, top_k=30)
    print(f"  - 向量召回: {len(vector_candidates)} 条")
    
    # Path B: 关键词检索
    keyword_candidates = db.search_keyword(query, top_k=30)
    print(f"  - 关键词召回: {len(keyword_candidates)} 条")
    
    # 2. 合并 (Merge Phase)
    merged_candidates = rrf_merge(vector_candidates, keyword_candidates)
    print(f"  - 合并后候选: {len(merged_candidates)} 条")
    
    # 3. 加权重排 (Rerank Phase)
    weighted_results = []
    for res in merged_candidates:
        # 使用 RRF 分数作为基础分
        base_score = res['rrf_score']
        filepath = res.get('filepath', '')
        
        # 定义分层权重策略 (保持原有的业务偏好)
        weight = 1.0
        if "A核心条文" in filepath:
            weight = 1.5  # 核心法典加权
        elif "B三大诉讼" in filepath:
            weight = 1.3
        elif "C分类" in filepath:
            weight = 1.1
        elif "F3工作文件" in filepath:
            weight = 0.5  # 降权噪音文件
            
        final_score = base_score * weight
        res['final_score'] = final_score
        weighted_results.append(res)
    
    # 按最终得分排序
    weighted_results.sort(key=lambda x: x['final_score'], reverse=True)
    final_results = weighted_results[:top_k]

    # 4. 打印结果
    if not final_results:
        print("未找到相关法律条款。")
        return

    print(f"\n=== 最终推荐结果 (Top {len(final_results)}) ===")
    for i, res in enumerate(final_results):
        meta = res['meta_info']
        source = meta.get('source', '未知来源')
        path = meta.get('path', '')
        content = res['content']
        score = res['final_score']
        
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