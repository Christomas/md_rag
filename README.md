# Legal RAG System (本地法律 AI 知识库)

基于 **Markdown** 法律数据库构建的离线、高性能法律问答系统。
本系统旨在解决法律大模型常见的“幻觉”问题，通过 RAG (Retrieval-Augmented Generation) 技术，强制 AI 基于检索到的真实法律条款回答问题，并提供精确到条款的引用。

## ✨ 特性

- **🏠 离线优先**: 数据（SQLite）、索引（FAISS）和嵌入模型（BGE-M3）完全运行在本地，无需联网。
- **🔍 混合检索**: 结合 **BGE-M3 语义向量** 与 **SQLite FTS5 关键词** 检索，既懂“意思”又懂“精确条文号”。
- **🎯 零幻觉**: 强制“检索优先”，回答必须基于检索到的 `<<Source>>` 上下文，拒绝编造。
- **🧩 智能切片**: 专为法律文档设计的状态机解析器，精准识别“编-章-节-条”层级。
- **🔄 自动维护**: 提供 `loop_build.sh` 守护脚本，支持断点续传、内存防泄漏和增量同步。

## 🛠️ 环境要求

- Python 3.10+
- 操作系统: macOS (Apple Silicon 优化) / Linux

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

## 🚀 使用指南

### 1. 构建知识库 (Build Index)

推荐使用守护脚本启动构建，它会自动处理内存释放和增量更新：

```bash
chmod +x loop_build.sh
./loop_build.sh
```

该脚本会：
1. 自动扫描 `md_vault` 变更（增/删/改）。
2. 在后台分批计算向量（支持断点续传）。
3. 自动构建 FAISS 向量索引与 FTS5 全文索引。

### 2. 测试检索 (Retrieve)

验证混合检索效果（向量+关键词并行召回）：

```bash
PYTHONPATH=. python3 src/retrieve.py "醉酒驾车撞人怎么判"
```

系统将返回 Top-20 加权排序后的相关切片。

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
