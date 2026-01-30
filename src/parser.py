import re
import os
import yaml
import hashlib
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
        if len(text) > self.max_size:
            flushed = self._flush()
            if flushed: chunks.append(flushed)
            
            text_chunks = self._split_long_text(text, self.target_size)
            for tc in text_chunks:
                chunks.append(self._create_chunk(tc, metadata.copy()))
            
            self.current_meta = metadata.copy()
            return chunks

        # 3. 正常合并逻辑
        if self.buffer_len + len(text) > self.max_size:
            flushed = self._flush()
            if flushed: chunks.append(flushed)
        
        if not self.buffer:
            self.current_meta = metadata.copy()
            
        self.buffer.append(text)
        self.buffer_len += len(text)
        
        if self.buffer_len >= self.target_size:
            flushed = self._flush()
            if flushed: chunks.append(flushed)
            
        return chunks

    def _split_long_text(self, text: str, size: int) -> List[str]:
        # 1. 特殊处理 HTML 表格 (已实现表头注入)
        if "<table>" in text and "</table>" in text:
            return self._split_html_table(text, size)
            
        group_id = hashlib.md5(text[:100].encode()).hexdigest()[:8]
        
        # 2. 递归式语义切分
        # 先按段落切
        paragraphs = text.split('\n')
        chunks = []
        curr_text = ""
        part_idx = 1
        
        for para in paragraphs:
            para = para.strip()
            if not para: continue
            
            # 如果单段就超过了 size，则按句子切
            if len(para) > size:
                # 先把之前累积的提交
                if curr_text:
                    chunks.append(curr_text + f"\n[关联块: {group_id} - Part {part_idx}]")
                    curr_text = ""
                    part_idx += 1
                
                # 按句号、感叹号、问号切分
                sentences = re.split(r'([。！？])', para)
                # re.split 会把分隔符也作为元素，需要合并回去
                combined_sentences = []
                for i in range(0, len(sentences)-1, 2):
                    combined_sentences.append(sentences[i] + sentences[i+1])
                if len(sentences) % 2 == 1:
                    combined_sentences.append(sentences[-1])
                
                s_curr = ""
                for s in combined_sentences:
                    if len(s_curr) + len(s) > size and s_curr:
                        chunks.append(s_curr + f"\n[关联块: {group_id} - Part {part_idx}]")
                        s_curr = s
                        part_idx += 1
                    else:
                        s_curr += s
                if s_curr:
                    curr_text = s_curr # 留给下一个段落合并或最终提交
            else:
                # 正常合并段落
                if len(curr_text) + len(para) > size and curr_text:
                    chunks.append(curr_text + f"\n[关联块: {group_id} - Part {part_idx}]")
                    curr_text = para
                    part_idx += 1
                else:
                    curr_text = (curr_text + "\n" + para) if curr_text else para
                    
        if curr_text:
            chunks.append(curr_text + f"\n[关联块: {group_id} - Part {part_idx}]")
            
        return chunks

    def _split_html_table(self, table_text: str, size: int) -> List[str]:
        """
        拆分 HTML 表格并注入表头。
        """
        group_id = hashlib.md5(table_text[:100].encode()).hexdigest()[:8]
        # 提取表头 (<tr>...</tr> 结构，通常是第一行)
        header_match = re.search(r'<tr.*?>.*?</tr>', table_text, re.DOTALL)
        header = header_match.group(0) if header_match else ""
        
        # 提取所有数据行
        rows = re.findall(r'<tr.*?>.*?</tr>', table_text, re.DOTALL)
        if len(rows) <= 1: return [table_text] # 只有表头或单行
        
        data_rows = rows[1:] if header else rows
        
        results = []
        curr_batch = [header] if header else []
        curr_len = len(header)
        part_idx = 1
        
        for row in data_rows:
            if curr_len + len(row) > size and len(curr_batch) > (1 if header else 0):
                # 封闭当前表格块
                content = "<table>" + "".join(curr_batch) + "</table>"
                content += f"\n[关联表格: {group_id} - Part {part_idx}]"
                results.append(content)
                
                curr_batch = [header] if header else []
                curr_len = len(header)
                part_idx += 1
                
            curr_batch.append(row)
            curr_len += len(row)
            
        if curr_batch:
            content = "<table>" + "".join(curr_batch) + "</table>"
            content += f"\n[关联表格: {group_id} - Part {part_idx}]"
            results.append(content)
            
        return results

    def _create_chunk(self, content: str, meta: Dict[str, str]) -> Chunk:
        # --- Context Injection ---
        source = meta.get('source', '未知法规')
        path = meta.get('path', '')
        
        # 注入文档级元数据 (YAML Info)
        doc_info = []
        if '效力位阶' in meta: doc_info.append(f"[{meta['效力位阶']}]")
        if '时效性' in meta: doc_info.append(f"[{meta['时效性']}]")
        doc_info_str = " ".join(doc_info)
        
        injected_content = f"【{source}】{doc_info_str} {path}\n{content}"
        return Chunk(content=injected_content, metadata=meta)

    def _flush(self) -> Optional[Chunk]:
        if not self.buffer:
            return None
        
        raw_content = "\n".join(self.buffer)
        chunk = self._create_chunk(raw_content, self.current_meta.copy())
        
        self.buffer = []
        self.buffer_len = 0
        return chunk

    def flush(self) -> Optional[Chunk]:
        return self._flush()


class LegalDocParser:
    """
    状态机解析器：实现智能路径提取、多文档拆分和级联切分。
    """
    PATTERN_ARTICLE = re.compile(r"^\s*第[零一二三四五六七八九十百千0-9]+条(?:之[零一二三四五六七八九十0-9]+)?[\s：]")
    PATTERN_CHAPTER = re.compile(r"^\s*第[零一二三四五六七八九十百千0-9]+[编章节][\s：$]")
    PATTERN_LIST = re.compile(r"^\s*(?:[一二三四五六七八九十]+、|\d+\.|[（(][一二三四五六七八九十]+\d*[）)]|[（(]\d+[）)])")
    PATTERN_SPECIAL = re.compile(r"^\s*(?:【[^】]+】|附[录件])")
    
    # 垃圾块检测：NotebookLM 校验块
    PATTERN_JUNK_START = re.compile(r"^##\s*NotebookLM\s*校验块")

    def __init__(self, target_size: int = 1200):
        self.target_size = target_size

    def parse(self, filepath: str) -> List[Chunk]:
        if not os.path.exists(filepath):
            return []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        filename = os.path.basename(filepath).replace('.md', '')
        
        # 0. 解析 YAML Front Matter
        doc_meta = {"source": filename}
        yaml_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1)
            try:
                # 简单解析，如果 yaml 库失败则忽略
                parsed_yaml = yaml.safe_load(yaml_content)
                if isinstance(parsed_yaml, dict):
                    # 提取关键字段
                    for key in ['效力位阶', '时效性', '制定机关', '公布日期']:
                        if key in parsed_yaml:
                            doc_meta[key] = str(parsed_yaml[key])
            except:
                pass
            # 移除 YAML 部分，避免重复处理
            content = content[yaml_match.end():]

        lines = content.split('\n')
        merger = ChunkMerger(target_size=self.target_size)
        all_chunks = []

        current_path_stack = [] 
        pending_chapters = []   
        has_started_articles = False
        pre_article_buffer = []
        
        # 垃圾块跳过标志
        skip_mode = False
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            
            # 1. 垃圾块检测 (NotebookLM 校验块)
            if self.PATTERN_JUNK_START.match(line_clean):
                skip_mode = True
                continue
            # 校验块通常在文件末尾，或者由 '---' 结束，这里简单处理：
            # 如果进入 skip_mode，且遇到下一个一级标题或结束符，可能需要恢复（但在本语境下，校验块通常在末尾）
            # 观察数据发现校验块后通常没有正文了，或者由 --- 分隔。
            # 为安全起见，我们假设校验块包含 "法宝引证码" 等特征，持续跳过直到文件结束或遇到明确的新段落
            if skip_mode:
                # 如果遇到新的分割线，可能是下一个文档的开始（多文档情况），或者校验块结束
                if line_clean.startswith('---'):
                    skip_mode = False
                continue

            if not line_clean:
                continue

            # A. Hard Reset Detection (多文档拆分)
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
                
                # 合并 Doc Meta 和 Path Meta
                meta = doc_meta.copy()
                meta["path"] = " > ".join(current_path_stack)
                all_chunks.extend(merger.add(line, meta))
                continue

            # C. 正文内容
            if not has_started_articles:
                pre_article_buffer.append(line)
            else:
                meta = doc_meta.copy()
                meta["path"] = " > ".join(current_path_stack)
                all_chunks.extend(merger.add(line, meta))

        # D. 文件结束处理
        if pre_article_buffer:
            header_text = "\n".join(pre_article_buffer)
            if len(header_text) > 50:
                meta = doc_meta.copy()
                meta["path"] = "前言/说明"
                # 使用智能切分逻辑
                text_chunks = merger._split_long_text(header_text, merger.target_size)
                for i, tc in enumerate(text_chunks):
                    all_chunks.insert(i, merger._create_chunk(tc, meta.copy()))

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
            print(f"--- Chunk {i} [{c.metadata.get('path')}] ---")
            print(c.content[:150] + "...")
