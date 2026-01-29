import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
from sentence_transformers import SentenceTransformer
import json

def main():
    if len(sys.argv) < 2:
        return
    
    query = " ".join(sys.argv[1:])
    # 强制 CPU
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    
    # 输出 JSON 格式，方便解析
    print(json.dumps(vec.tolist()))

if __name__ == "__main__":
    main()
