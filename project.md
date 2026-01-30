# 项目：本地法律 AI 知识库 (复刻 NotebookLM)

> **版本**: 0.1
> **日期**: 2026-01-27
> **目标**: 基于本地 Markdown 法律数据库 (`md_vault`)，构建一个**离线优先、零幻觉**的 AI 法律问答系统。
> **核心哲学**: 运行时零依赖（仅依赖 Gemini CLI），构建时重逻辑（处理脏数据）。

---

## 1. 整体架构 (System Architecture)

系统采用 **构建态 (Build Time)** 与 **运行态 (Runtime)** 严格解耦的设计。

### 1.1 构建态 (Build Time) —— “炼丹炉”
*   **职责**: 负责脏活累活。清洗数据、切片、计算向量、维护索引。
*   **工具**: Python 3.10+, `sentence-transformers` (本地 BGE-M3 模型), `faiss-cpu`, SQLite。
*   **输入**: `/md_vault` 中的 Markdown 文件。
*   **产出**: 
    1.  `knowledge.db` (SQLite): 包含完整切片文本、元数据、文件哈希指纹。
    2.  `vector.index` (FAISS): 包含所有切片的 Dense Vector，用于相似度检索。

### 1.2 运行态 (Runtime) —— “大脑”
*   **职责**: 理解用户意图、调用检索工具、生成带引用的回答。
*   **工具**: **Gemini CLI (Agent)** + `retrieve.py`。
*   **流程**:
    1.  用户提问。
    2.  Agent 自动调用 `retrieve.py` 获取 Top-K 切片（包含原文和 `<<Source>>` 标记）。
    3.  Agent 阅读切片，生成回答，并强制使用 `[文件 > 章节 > 条款]` 格式进行引用。

---

## 2. 核心算法细节 (Core Algorithms)

### 2.1 智能切片器 (LegalDocParser) —— RAG 的灵魂
法律文档结构复杂，标准 Markdown 切分器完全不可用。我们需要实现一个**基于状态机的专用解析器**。

#### 2.1.1 触发特征 (Regex Triggers)
*   **Article (实体锚点)**: `^第[零一二三四五六七八九十百千]+条\s`
*   **Chapter (层级节点)**: `^第[零一二三四五六七八九十百千]+[编章节]\s`
*   **Special (特殊锚点)**: `^附`, `^【.*】` (案例标记, 如【裁判要旨】)
*   **Reset (复位信号)**: 在正文开始后，再次检测到 `^第一条\s`。

#### 2.1.2 状态机逻辑 (State Machine Logic)

1.  **前置内容捕获 (Pre-Article Context)**
    *   **逻辑**: 在遇到**第一个**“实体锚点”或“层级节点”之前的所有文本。
    *   **处理**: 独立存为 `Chunk 0`。
    *   *目的*: 保留“制定说明”、“修改说明”等全局上下文，不丢失，也不混入第一条。

2.  **路径解析与防干扰 (Path Resolution & Anti-Hallucination)**
    *   **痛点**: 许多文件开头有“目录”区域（连续的“第一章...第二章...”），如果直接解析，会导致路径栈堆叠（第一章 > 第二章 > ...），且干扰正文检索。
    *   **策略**: **延迟生效 (Pending Context)**。
        *   当扫描到 `Chapter` (编/章/节) 时，**不**直接更新 `current_path`。
        *   而是将其放入 `pending_chapters` 缓冲区。
        *   **只有**当扫描到 `Article` (条款) 或 `Special` (特殊锚点) 时：
            *   清空旧的 `current_path`。
            *   将 `pending_chapters` 里的内容固化为新的 `current_path`。
            *   清空 `pending_chapters`。
    *   *效果*: 完美跳过目录区，只有真正包含内容的章节才会被记入路径。

3.  **多文档拆分 (Multi-Document Split)**
    *   **痛点**: 一个 `.md` 文件可能包含多个法律文件，导致后半部分标题错误。
    *   **策略**: **强复位 (Hard Reset)**。
        *   如果状态机 `has_started_articles` 为 True，却再次遇到了 `^第一条\s`。
        *   **Action**: 
            1.  强制结束当前文档流。
            2.  **逆向回溯 (Look-behind)**: 从当前行向上扫描约 20 行。
            3.  寻找最近的 `H1/H2` 或“短文本行”（跳过可能存在的层级节点）作为**新子文档的标题**。
            4.  重置 `current_path`，开始处理子文档。

4. **动态合并缓冲区 (Dynamic Merge Buffer)**
    *   **痛点**: 法律条款长短不一，短条款（如“本法自发布之日起施行”）单独切片无检索价值，且打断了“定义-范围-罚则”的逻辑链。
    *   **策略**: **贪婪合并 (Greedy Merger)**。
        *   **基础粒度**: 依然先识别 `Article` (条款) 和 `List` (列表) 等逻辑单元。
        *   **合并逻辑**: 将识别出的单元放入缓冲区，**不立即输出**。
        *   **触发条件**: 
            *   **Soft Limit**: 当 `Buffer长度` + `新单元长度` > `目标阈值` (默认 1000~1500 字符，可根据模型上下文调整) 时，提交 Buffer 为一个 Chunk。
            *   **Hard Boundary**: 遇到 **路径变更 (Path Change)** (如从第一章进入第二章) 或 **文件切换** 时，**必须**强制提交 Buffer。这保证了 Metadata 的纯净性。
    *   *效果*: 显著减少碎片化切片，提升 Embedding 的语义密度，让模型能看到完整的法律逻辑。

5. **三级级联切分 (Cascade Splitting)**
    *   **职责**: 当单条正文内容依然过长或缺乏明确锚点时，作为底层的物理切分引擎。
    *   **Level 1 (Strict)**: 命中 `Article` 锚点。按条款切分。 (优先级最高)
    *   **Level 2 (List)**: 未命中 L1，但命中 `^一、` 或 `^1.`。按列表项切分。
    *   **Level 3 (Fallback)**: 均未命中。使用递归字符切分 (RecursiveCharacterTextSplitter)，每 500-1000 字一块，同时保证每块内包含完整的行。

#### 2.1.3 上下文注入 (Context Injection)
每个切片的纯文本前，**必须**拼接元数据头：
`<<Source: {Title} | Path: {current_path}>>\n{Content}`
*目的*: 让 Embedding 模型理解这段话属于哪部法律的哪个章节，解决“孤立语义”问题。

---

### 2.2 增量更新策略 (Incremental Updates)

*   **状态追踪**: SQLite `file_registry` 表记录 `{filepath, file_hash, last_updated}`。
*   **构建模式**: 采用 `loop_build.sh` 守护进程模式，每处理 N 条数据自动重启，彻底解决 Python/MPS 内存泄漏问题。
*   **流程**:
    1.  **Scan**: 遍历磁盘，对比 Hash，标记 New/Modified/Deleted。
    2.  **Clean**: 物理删除失效文件的 chunks 和向量。
    3.  **Process**: 解析新文件 -> 存入 SQLite。
    4.  **Vectorize**: 增量计算缺失的 `embedding` (支持断点续传)。
    5.  **Rebuild**: 全量重建 FAISS 索引和 FTS 全文索引。

### 2.3 混合检索策略 (Hybrid Retrieval) —— [已实装]

为解决法律场景下“语义模糊”与“精确条文”的矛盾，采用双路召回机制。

1.  **Path A: 语义召回 (Semantic)**
    *   工具: `BGE-M3` + `FAISS`
    *   目标: 召回 Top-30。解决“意思对但词不对”的问题（如搜“杀人”召回“故意杀人罪”）。
2.  **Path B: 关键词召回 (Lexical)**
    *   工具: `SQLite FTS5` (BM25)
    *   目标: 召回 Top-30。解决“精确法条号”或“特定术语”的问题（如搜“刑法第二十条”）。
3.  **Merge: 互惠秩融合 (RRF)**
    *   算法: $Score = \sum \frac{1}{60 + Rank_i}$
    *   效果: 即使某条结果仅在单路中出现（如仅关键词命中），也能获得足够高的排名。
4.  **Rerank: 业务加权**
    *   对 `A核心条文` (1.5x) 等目录进行加权，输出最终 Top-20。

---

## 3. 数据存储结构 (Schema)

### 3.1 SQLite (`knowledge.db`)

**Table: `files`** (文件注册表)
| Field | Type | Note |
| :--- | :--- | :--- |
| `id` | INTEGER PK | |
| `filepath` | TEXT UNIQUE | 相对路径 |
| `file_hash` | TEXT | MD5/SHA256 |
| `title` | TEXT | 解析出的真实标题 |
| `status` | TEXT | 'active', 'archived' |

**Table: `chunks`** (切片数据)
| Field | Type | Note |
| :--- | :--- | :--- |
| `id` | INTEGER PK | 对应 FAISS ID |
| `file_id` | INTEGER FK | |
| `content` | TEXT | 切片纯文本 |
| `meta_info` | TEXT | JSON: {path: "...", source: "..."} |
| `embedding` | BLOB | **核心**: 1024维 float32 向量 |

**Virtual Table: `chunks_fts`** (全文索引)
*   **Type**: FTS5 Virtual Table
*   **Content**: 镜像 `chunks.content`
*   **Triggers**: 自动随 `chunks` 表增删改同步。

### 3.2 FAISS (`vector.index`)
*   **类型**: `IndexFlatIP` (内积/余弦相似度) 或 `IndexHNSW` (大规模时用)。
*   **ID 映射**: FAISS 中的 ID 直接对应 SQLite `chunks.id`。

---

## 4. 目录结构 (Directory Layout)

```text
/md_rag/
├── md_vault/               # (Source) 原始 Markdown 数据
├── test/                   # (Source) 原型测试数据 (随机抽取 500 files，保证分布均衡、大小均衡)
├── data/                   # (Output) 生成的数据产物
│   ├── knowledge.db        # SQLite Metadata & Text
│   └── vector.index        # FAISS Index
├── src/                    # (Code) 源代码
│   ├── parser.py           # 核心：LegalDocParser 类实现
│   ├── embedding.py        # 核心：BGE-M3 模型封装
│   ├── db_manager.py       # 工具：SQLite 和 FAISS 的增删改查
│   ├── builder.py          # 主程序：构建脚本 (含增量逻辑)
│   ├── curator.py          # 工具：数据清洗脚本
│   └── retrieve.py         # 运行时：搜索脚本
└── project.md              # 本文档
```

---

## 5. 实施路线图 (Roadmap)

### Phase 1: 原型验证 (Prototype)
1.  **Environment**: `pip install sentence-transformers faiss-cpu numpy`。
2.  **Core**: 实现 `src/parser.py` (含所有状态机逻辑)。
3.  **Build**: 编写 `src/builder.py` (先跑通全量构建)。
4.  **Test**: 对 `test` 文件夹生成索引，人工抽检切片质量。
5.  **Run**: 编写 `src/retrieve.py`，Gemini CLI 接入测试。

### Phase 2: 生产化 (Production)
1.  **Scale**: 针对 `md_vault` 运行构建（预计 M1 芯片耗时 2-4 小时）。
2.  **Incremental**: 完善 `builder.py` 的 Hash 对比逻辑。
3.  **Curation**: 实施基于规则的初筛，保留价值数据。

### Phase 3: 优化 (Refinement)
1.  **Prompt Engineering**: 优化 Gemini 的 System Prompt，确保严格引用格式。
2.  **UI**: (可选) 封装简单的 CLI 交互界面。

---

## 6. Pending Optimizations (Next Steps)

### 6.1 智能重排与生成 (LLM Reranking & Synthesis)
*   **Strategy**:
    *   输入 Top-20 条文 (约 30k-40k Token)。
    *   Gemini 识别法条冲突、适用层级及核心要点。
    *   输出：结构化法律建议 + 精确引用。

### 6.2 动态裁剪 (Sub-chunk Cropping) - Optional
针对大粒度切片 (1000+ 字符) 造成的 Token 浪费问题，考虑在检索后进行动态视窗裁剪。

*   **Logic**:
    *   在命中 Chunk 中定位关键词/语义中心。
    *   截取中心前后 N 行作为 "Snippet"。
*   **Risk**: 可能会丢失“例外条款”或上下文约束 (断章取义)。需谨慎评估法律风险后再实施。