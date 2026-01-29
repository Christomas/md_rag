import re
import os
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class Chunk:
    content: str
    metadata: Dict[str, str]

class ChunkMerger:
    """
    负责将细粒度的文本片段（如单个条款）合并为更有意义的逻辑块。
    """
    def __init__(self, target_size: int = 1200, max_size: int = 2500):
        self.target_size = target_size
        self.max_size = max_size
        self.buffer: List[str] = []
        self.buffer_len = 0
        self.current_meta: Dict[str, str] = {}

    def add(self, text: str, metadata: Dict[str, str]) -> List[Chunk]:
        """
        添加文本，并返回可能生成的 Chunks 列表。
        """
        chunks = []
        
        # 1. 检查硬边界 (元数据变更)
        path_changed = False
        if self.buffer and self.current_meta:
            if metadata.get('path') != self.current_meta.get('path') or \
               metadata.get('source') != self.current_meta.get('source'):
                path_changed = True

        # 如果路径变更，必须强制提交缓冲区
        if path_changed:
            flushed = self._flush()
            if flushed: chunks.append(flushed)
            self.current_meta = metadata.copy()

        # 2. 处理超长文本 (Cascade Level 3 兜底)
        # 如果单段文本已经超过了 max_size，先清空 buffer，然后对该文本进行拆分
        if len(text) > self.max_size:
            flushed = self._flush()
            if flushed: chunks.append(flushed)
            
            # 简单的物理切分
            text_chunks = self._split_long_text(text, self.target_size)
            for tc in text_chunks:
                chunks.append(Chunk(content=tc, metadata=metadata.copy()))
            
            # 重置 meta
            self.current_meta = metadata.copy()
            return chunks

        # 3. 正常合并逻辑
        # 如果加上新文本会超过 max_size，先提交旧的
        if self.buffer_len + len(text) > self.max_size:
            flushed = self._flush()
            if flushed: chunks.append(flushed)
        
        if not self.buffer:
            self.current_meta = metadata.copy()
            
        self.buffer.append(text)
        self.buffer_len += len(text)
        
        # 如果达到软限制，提交
        if self.buffer_len >= self.target_size:
            flushed = self._flush()
            if flushed: chunks.append(flushed)
            
        return chunks

    def _split_long_text(self, text: str, size: int) -> List[str]:
        """简单的物理切分，保留行完整性"""
        lines = text.split('\n')
        results = []
        curr = []
        curr_len = 0
        for l in lines:
            if curr_len + len(l) > size and curr:
                results.append("\n".join(curr))
                curr = []
                curr_len = 0
            curr.append(l)
            curr_len += len(l)
        if curr:
            results.append("\n".join(curr))
        return results

    def _flush(self) -> Optional[Chunk]:
        if not self.buffer:
            return None
        content = "\n".join(self.buffer)
        chunk = Chunk(content=content, metadata=self.current_meta.copy())
        self.buffer = []
        self.buffer_len = 0
        return chunk

    def flush(self) -> Optional[Chunk]:
        return self._flush()


class LegalDocParser:
    """
    状态机解析器：实现智能路径提取、多文档拆分和级联切分。
    """
    # Level 1: 条款
    PATTERN_ARTICLE = re.compile(r"^\s*第[零一二三四五六七八九十百千0-9]+条(?:之[零一二三四五六七八九十0-9]+)?[\s：]")
    # Level 2: 章节编
    PATTERN_CHAPTER = re.compile(r"^\s*第[零一二三四五六七八九十百千0-9]+[编章节][\s：$]")
    # Level 2: 列表锚点
    PATTERN_LIST = re.compile(r"^\s*(?:[一二三四五六七八九十]+、|\d+\.|[（(][一二三四五六七八九十]+\d*[）)]|[（(]\d+[）)])")
    # 特殊：裁判要旨/附录
    PATTERN_SPECIAL = re.compile(r"^\s*(?:【[^】]+】|附[录件])")

    def __init__(self, target_size: int = 1200):
        self.target_size = target_size

    def parse(self, filepath: str) -> List[Chunk]:
        if not os.path.exists(filepath):
            return []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        filename = os.path.basename(filepath).replace('.md', '')
        lines = content.split('\n')
        
        merger = ChunkMerger(target_size=self.target_size)
        all_chunks = []

        # State Variables
        current_path_stack = [] 
        pending_chapters = []   
        has_started_articles = False
        pre_article_buffer = []
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean:
                continue

            # A. Hard Reset Detection (多文档拆分)
            # 如果已经开始正文，又遇到了 "第一条"，可能是新法规
            if has_started_articles and self.PATTERN_ARTICLE.match(line_clean) and ("第一条" in line_clean or "第1条" in line_clean):
                flushed = merger.flush()
                if flushed: all_chunks.append(flushed)
                current_path_stack = []
                pending_chapters = []
                has_started_articles = False

            # B. 结构锚点识别
            is_article = self.PATTERN_ARTICLE.match(line_clean)
            is_chapter = self.PATTERN_CHAPTER.match(line_clean)
            is_special = self.PATTERN_SPECIAL.match(line_clean)
            is_list = self.PATTERN_LIST.match(line_clean)

            if is_chapter:
                pending_chapters.append(line_clean)
                continue

            if is_article or is_special or is_list:
                has_started_articles = True
                if pending_chapters:
                    self._update_path_stack(current_path_stack, pending_chapters)
                    pending_chapters = []
                
                meta = {"source": filename, "path": " > ".join(current_path_stack)}
                all_chunks.extend(merger.add(line, meta))
                continue

            # C. 正文内容
            if not has_started_articles:
                pre_article_buffer.append(line)
            else:
                meta = {"source": filename, "path": " > ".join(current_path_stack)}
                # 列表项也可以作为触发点，但在我们的贪婪合并逻辑中，它只是普通正文行
                all_chunks.extend(merger.add(line, meta))

        # D. 文件结束处理
        # 1. 前言处理
        if pre_article_buffer:
            header_text = "\n".join(pre_article_buffer)
            if len(header_text) > 100:
                all_chunks.insert(0, Chunk(content=header_text, metadata={"source": filename, "path": "前言/说明"}))

        # 2. Final Flush
        final_chunk = merger.flush()
        if final_chunk: all_chunks.append(final_chunk)

        return all_chunks

    def _update_path_stack(self, stack: List[str], pending: List[str]):
        level_map = {'编': 0, '章': 1, '节': 2}
        for p in pending:
            level = -1
            for key, val in level_map.items():
                if key in p:
                    level = val
                    break
            if level != -1:
                while len(stack) <= level: stack.append("")
                stack[level] = p
                del stack[level+1:]
            else:
                stack.append(p)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        parser = LegalDocParser()
        chunks = parser.parse(sys.argv[1])
        for i, c in enumerate(chunks):
            print(f"--- Chunk {i} [{c.metadata['path']}] ---")
            print(c.content[:100] + "...")
