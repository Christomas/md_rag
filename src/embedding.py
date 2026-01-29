from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingEngine:
    """
    封装 BGE-M3 模型，用于计算文本向量。
    """
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        # 强制使用 CPU 以保证稳定性，尤其是在内存受限或 MPS 兼容性有问题时
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str], batch_size: int = 16, show_progress: bool = False) -> np.ndarray:
        """
        计算文档列表的向量。
        """
        return self.model.encode(
            texts, 
            batch_size=batch_size, 
            normalize_embeddings=True, 
            show_progress_bar=show_progress
        )

    def embed_query(self, text: str) -> np.ndarray:
        """
        计算查询词的向量。
        """
        return self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]

if __name__ == "__main__":
    # 快速测试
    engine = EmbeddingEngine()
    vec = engine.embed_query("中华人民共和国宪法")
    print(f"Embedding shape: {vec.shape}")
