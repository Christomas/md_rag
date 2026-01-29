# Legal RAG System (本地法律 AI 知识库)

基于 **Markdown** 法律数据库构建的离线、高性能法律问答系统。
本系统旨在解决法律大模型常见的“幻觉”问题，通过 RAG (Retrieval-Augmented Generation) 技术，强制 AI 基于检索到的真实法律条款回答问题，并提供精确到条款的引用。

## ✨ 特性

- **🏠 离线优先**: 数据（SQLite）、索引（FAISS）和嵌入模型（BGE-M3）完全运行在本地，无需联网，保障数据隐私。
- **🎯 零幻觉**: 强制“检索优先”，回答必须基于检索到的 `<<Source>>` 上下文，拒绝编造。
- **🧩 智能切片**: 专为法律文档设计的状态机解析器，精准识别“编-章-节-条”层级，保留完整法律逻辑。
- **🚀 高性能**: 使用 FAISS 向量索引，支持毫秒级检索；基于 BGE-M3 模型提供高质量语义理解。
- **🔄 增量更新**: 智能追踪文件哈希，仅处理变更文件，极大降低维护成本。

## 🛠️ 环境要求

- Python 3.10+
- 操作系统: macOS / Linux (Windows 未经完整测试)

## 📦 安装与配置

1.  **克隆仓库**
    ```bash
    git clone https://github.com/Christomas/md_rag.git
    cd md_rag
    ```

2.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

3.  **准备数据**
    将您的 Markdown 格式法律文档放入 `md_vault/` 目录中。
    > 示例结构：
    > `md_vault/民商法/中华人民共和国民法典.md`

## 🚀 使用指南

### 1. 构建知识库 (Build Index)

首次运行或更新文档后，需要构建向量索引：

```bash
# 默认扫描 md_vault 目录
PYTHONPATH=. python3 src/builder.py

# 或者指定其他目录
PYTHONPATH=. python3 src/builder.py path/to/your/docs
```

该过程会：
1. 解析 Markdown 文件。
2. 生成文本切片并存入 SQLite (`data/knowledge.db`)。
3. 计算向量并构建 FAISS 索引 (`data/vector.index`)。

### 2. 测试检索 (Retrieve)

验证检索效果，查找最相关的法律条款：

```bash
PYTHONPATH=. python3 src/retrieve.py "非法集资的定义"
```

系统将返回 Top-5 相关切片及其原始出处。

### 3. 集成 AI 助手 (Agent)

如果您使用 Gemini CLI 或其他 Agent 框架，可配置 Agent 在回答法律问题前自动调用上述检索脚本，并将检索结果作为 Context 注入 Prompt。

**System Prompt 示例**:
> 详见 [GEMINI.md](GEMINI.md)

## 📂 项目结构

```text
.
├── data/               # [自动生成] 存放数据库和索引 (已忽略)
│   ├── knowledge.db    # SQLite: 文本与元数据
│   └── vector.index    # FAISS: 向量索引
├── md_vault/           # [输入] 您的 Markdown 法律文档库 (已忽略)
├── src/                # 源代码
│   ├── parser.py       # 法律文档专用解析器
│   ├── embedding.py    # BGE-M3 模型封装
│   ├── builder.py      # 索引构建脚本
│   └── retrieve.py     # 检索脚本
├── GEMINI.md           # AI 助手配置/提示词
└── project.md          # 详细技术文档
```

## 📜 许可证

MIT License
