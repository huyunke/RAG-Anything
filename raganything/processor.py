"""
RAGAnything 的文档处理功能

包含用于解析文档及处理多模态内容的方法
"""

import os
import time
import hashlib
import json
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

from raganything.base import DocStatus
from raganything.parser import MineruParser, MineruExecutionError, get_parser
from raganything.utils import (
    separate_content,
    insert_text_content,
    insert_text_content_with_multimodal_content,
    get_processor_for_type,
)
import asyncio
from lightrag.utils import compute_mdhash_id


class ProcessorMixin:
    """ProcessorMixin 类：包含 RAGAnything 的文档处理功能"""

    def _get_file_reference(self, file_path: str) -> str:
        """
        根据 use_full_path 配置获取文件引用

        参数说明：
            file_path: 文件的路径（可以是绝对路径或相对路径）

        返回：
            str: 如果 use_full_path 为 True，则返回完整路径；否则返回纯文件名（不带路径）。
        """
        if self.config.use_full_path:
            return str(file_path)
        else:
            return os.path.basename(file_path)

    def _generate_cache_key(
            self, file_path: Path, parse_method: str = None, **kwargs
    ) -> str:
        """
        Generate cache key based on file path and parsing configuration

        Args:
            file_path: Path to the file
            parse_method: Parse method used
            **kwargs: Additional parser parameters

        Returns:
            str: Cache key for the file and configuration
        """

        # Get file modification time
        mtime = file_path.stat().st_mtime

        # Create configuration dict for cache key
        config_dict = {
            "file_path": str(file_path.absolute()),
            "mtime": mtime,
            "parser": self.config.parser,
            "parse_method": parse_method or self.config.parse_method,
        }

        # Add relevant kwargs to config
        relevant_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
               in [
                   "lang",
                   "device",
                   "start_page",
                   "end_page",
                   "formula",
                   "table",
                   "backend",
                   "source",
               ]
        }
        config_dict.update(relevant_kwargs)

        # Generate hash from config
        config_str = json.dumps(config_dict, sort_keys=True)
        cache_key = hashlib.md5(config_str.encode()).hexdigest()

        return cache_key

    def _generate_content_based_doc_id(self, content_list: List[Dict[str, Any]]) -> str:
        """
        基于文档内容生成唯一的 doc_id（文档 ID）

        参数说明：
            content_list: 已解析的内容列表（包含文字、图片、表格等块）

        返回：
            str: 带有 "doc-" 前缀的、基于内容生成的唯一标识字符串
        """
        # 导入 LightRAG 提供的 MD5 哈希计算工具
        from lightrag.utils import compute_mdhash_id

        # 用于存放所有提取出的关键内容片段，最后合并计算哈希
        content_hash_data = []

        for item in content_list:
            if isinstance(item, dict):
                # 处理纯文本：直接使用去除首尾空格后的文字内容
                if item.get("type") == "text" and item.get("text"):
                    content_hash_data.append(item["text"].strip())
                # 处理图片：使用图片的路径作为标识（路径变了，指纹就会变）
                elif item.get("type") == "image" and item.get("img_path"):
                    content_hash_data.append(f"image:{item['img_path']}")
                # 处理表格：使用表格的主体数据作为标识
                elif item.get("type") == "table" and item.get("table_body"):
                    content_hash_data.append(f"table:{item['table_body']}")
                # 处理公式：使用公式的文本作为标识
                elif item.get("type") == "equation" and item.get("text"):
                    content_hash_data.append(f"equation:{item['text']}")
                else:
                    # 处理其他未知类型：将其强制转为字符串作为标识
                    content_hash_data.append(str(item))

        # 将所有提取到的关键片段用换行符连接，形成该文档的“内容签名”
        content_signature = "\n".join(content_hash_data)

        # 通过 MD5 算法将长长的签名转换成一个短小且唯一的哈希 ID
        doc_id = compute_mdhash_id(content_signature, prefix="doc-")

        return doc_id

    async def _get_cached_result(
            self, cache_key: str, file_path: Path, parse_method: str = None, **kwargs
    ) -> tuple[List[Dict[str, Any]], str] | None:
        """
        Get cached parsing result if available and valid

        Args:
            cache_key: Cache key to look up
            file_path: Path to the file for mtime check
            parse_method: Parse method used
            **kwargs: Additional parser parameters

        Returns:
            tuple[List[Dict[str, Any]], str] | None: (content_list, doc_id) or None if not found/invalid
        """
        if not hasattr(self, "parse_cache") or self.parse_cache is None:
            return None

        try:
            cached_data = await self.parse_cache.get_by_id(cache_key)
            if not cached_data:
                return None

            # Check file modification time
            current_mtime = file_path.stat().st_mtime
            cached_mtime = cached_data.get("mtime", 0)

            if current_mtime != cached_mtime:
                self.logger.debug(f"Cache invalid - file modified: {cache_key}")
                return None

            # Check parsing configuration
            cached_config = cached_data.get("parse_config", {})
            current_config = {
                "parser": self.config.parser,
                "parse_method": parse_method or self.config.parse_method,
            }

            # Add relevant kwargs to current config
            relevant_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                   in [
                       "lang",
                       "device",
                       "start_page",
                       "end_page",
                       "formula",
                       "table",
                       "backend",
                       "source",
                   ]
            }
            current_config.update(relevant_kwargs)

            if cached_config != current_config:
                self.logger.debug(f"Cache invalid - config changed: {cache_key}")
                return None

            content_list = cached_data.get("content_list", [])
            doc_id = cached_data.get("doc_id")

            if content_list and doc_id:
                self.logger.debug(
                    f"Found valid cached parsing result for key: {cache_key}"
                )
                return content_list, doc_id
            else:
                self.logger.debug(
                    f"Cache incomplete - missing content or doc_id: {cache_key}"
                )
                return None

        except Exception as e:
            self.logger.warning(f"Error accessing parse cache: {e}")

        return None

    async def _store_cached_result(
            self,
            cache_key: str,
            content_list: List[Dict[str, Any]],
            doc_id: str,
            file_path: Path,
            parse_method: str = None,
            **kwargs,
    ) -> None:
        """
        Store parsing result in cache

        Args:
            cache_key: Cache key to store under
            content_list: Content list to cache
            doc_id: Content-based document ID
            file_path: Path to the file for mtime storage
            parse_method: Parse method used
            **kwargs: Additional parser parameters
        """
        if not hasattr(self, "parse_cache") or self.parse_cache is None:
            return

        try:
            # Get file modification time
            file_mtime = file_path.stat().st_mtime

            # Create parsing configuration
            parse_config = {
                "parser": self.config.parser,
                "parse_method": parse_method or self.config.parse_method,
            }

            # Add relevant kwargs to config
            relevant_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                   in [
                       "lang",
                       "device",
                       "start_page",
                       "end_page",
                       "formula",
                       "table",
                       "backend",
                       "source",
                   ]
            }
            parse_config.update(relevant_kwargs)

            cache_data = {
                cache_key: {
                    "content_list": content_list,
                    "doc_id": doc_id,
                    "mtime": file_mtime,
                    "parse_config": parse_config,
                    "cached_at": time.time(),
                    "cache_version": "1.0",
                }
            }
            await self.parse_cache.upsert(cache_data)
            # Ensure data is persisted to disk
            await self.parse_cache.index_done_callback()
            self.logger.info(f"Stored parsing result in cache: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Error storing to parse cache: {e}")

    async def parse_document(
            self,
            file_path: str,
            output_dir: str = None,
            parse_method: str = None,
            display_stats: bool = None,
            **kwargs,
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        带缓存支持的文档解析函数

        参数说明：
            file_path: 待解析文件的路径
            output_dir: 输出目录（默认为配置中的 config.parser_output_dir）
            parse_method: 解析方法（默认为配置中的 config.parse_method）
            display_stats: 是否显示内容统计信息（默认为配置中的 config.display_content_stats）
            **kwargs: 传递给解析器的额外参数（如语言、设备、起始页码、是否开启公式/表格识别等）

        返回：
            tuple[List[Dict[str, Any]], str]: (内容块列表, 文档唯一ID)
        """

        # 初始化：如果调用时没传参数，则使用 config.py 里的默认值
        if output_dir is None:
            # 默认值./output
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            # 默认值auto
            parse_method = self.config.parse_method
        if display_stats is None:
            # 默认值True
            display_stats = self.config.display_content_stats

        self.logger.info(f"正在启动文档解析: {file_path}")

        # 检查文件是否存在
        file_path = Path(file_path)  # 将文件路径转换为 Path 对象，以便后续操作
        if not file_path.exists():
            raise FileNotFoundError(f"找不到文件: {file_path}")

        # 回调管理器，用于监控进度
        callback_file = str(file_path)
        callback_manager = getattr(self, "callback_manager", None)
        parse_start_time = time.time()

        # 触发“解析开始”的回调函数（用于界面进度条或日志记录）
        if callback_manager is not None:
            callback_manager.dispatch(
                "on_parse_start",
                file_path=callback_file,
                parser=self.config.parser,
            )

        # 缓存检查：这是为了省钱省时间。如果文件没变，直接用上次解析的结果
        # 生成缓存键（基于文件内容、解析方法和参数）
        cache_key = self._generate_cache_key(file_path, parse_method, **kwargs)

        # 尝试从磁盘或内存获取已有的解析结果
        cached_result = await self._get_cached_result(
            cache_key, file_path, parse_method, **kwargs
        )

        # 如果缓存有效，则直接返回缓存结果
        if cached_result is not None:
            content_list, doc_id = cached_result
            self.logger.info(f"发现有效缓存，直接使用缓存结果: {file_path}")
            if display_stats:
                self.logger.info(
                    f"* 缓存内容块总数: {len(content_list)}"
                )
            # 触发“解析完成”回调
            if callback_manager is not None:
                duration = time.time() - parse_start_time
                callback_manager.dispatch(
                    "on_parse_complete",
                    file_path=callback_file,
                    content_blocks=len(content_list),
                    doc_id=doc_id,
                    duration_seconds=duration,
                )
            return content_list, doc_id

        # 解析器选择：根据文件后缀名（.pdf, .docx 等）决定用哪个方法
        # ext: 文件后缀名，如 .pdf, .docx 等
        ext = file_path.suffix.lower()

        try:
            doc_parser = getattr(self, "doc_parser", None)
            if doc_parser is None:
                # 动态加载解析器（如 MinerU 或 PaddleOCR）
                doc_parser = get_parser(self.config.parser)
                self.doc_parser = doc_parser

            # 记录日志
            self.logger.info(
                f"正在使用 {self.config.parser} 解析器，方法模式为: {parse_method}"
            )

            # 情况1：PDF 文件处理
            if ext in [".pdf"]:
                self.logger.info("检测到 PDF 文件，正在调用 PDF 专用解析流程...")
                # 确保 parse_pdf 是异步执行，在独立线程运行，防止阻塞主异步循环
                # content_list: 解析出的内容列表
                content_list = await asyncio.to_thread(
                    doc_parser.parse_pdf,
                    pdf_path=file_path,
                    output_dir=output_dir,
                    method=parse_method,
                    **kwargs,
                )
            # 情况2：图片文件处理
            elif ext in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tiff",
                ".tif",
                ".gif",
                ".webp",
            ]:
                self.logger.info("检测到图像文件，正在调用图像解析流程...")
                try:
                    # 确保 parse_image 是异步执行，在独立线程运行，防止阻塞主异步循环
                    # content_list: 解析出的内容列表
                    content_list = await asyncio.to_thread(
                        doc_parser.parse_image,
                        image_path=file_path,
                        output_dir=output_dir,
                        **kwargs,
                    )
                except NotImplementedError:
                    # 如果当前解析器不支持图片，则强制降级使用 MinerU 处理
                    self.logger.warning(
                        f"{self.config.parser} 解析器不支持图片解析，正在回退至 MinerU"
                    )
                    # 确保 parse_image 是异步执行，在独立线程运行，防止阻塞主异步循环
                    # content_list: 解析出的内容列表
                    content_list = await asyncio.to_thread(
                        MineruParser().parse_image,
                        image_path=file_path,
                        output_dir=output_dir,
                        **kwargs,
                    )
            # 情况3：Office 文件,html文件处理
            elif ext in [
                ".doc",
                ".docx",  # 赛题要求解析的文档类型之一
                ".ppt",
                ".pptx",
                ".xls",
                ".xlsx",  # 赛题要求解析的文档类型之一
                ".html",
                ".htm",
                ".xhtml",
            ]:
                self.logger.info(
                    "检测到 Office 或 HTML 文档，正在调用对应的解析流程..."
                )
                # 确保 parse_office_doc 是异步执行，在独立线程运行，防止阻塞主异步循环
                # content_list: 解析出的内容列表
                content_list = await asyncio.to_thread(
                    doc_parser.parse_office_doc,
                    doc_path=file_path,
                    output_dir=output_dir,
                    **kwargs,
                )
            # 情况4：其他文件处理
            else:
                # 对于其他或未知的格式，使用通用解析器
                self.logger.info(
                    f"针对 {ext} 文件使用通用解析逻辑 (模式={parse_method})..."
                )
                # 确保 parse_document 是异步执行，在独立线程运行，防止阻塞主异步循环
                # content_list: 解析出的内容列表
                content_list = await asyncio.to_thread(
                    doc_parser.parse_document,
                    file_path=file_path,
                    method=parse_method,
                    output_dir=output_dir,
                    **kwargs,
                )

        except MineruExecutionError as e:
            self.logger.error(f"MinerU 执行指令失败: {e}")
            # 如果报错了，通过回调通知外部
            if callback_manager is not None:
                callback_manager.dispatch(
                    "on_parse_error",
                    file_path=callback_file,
                    error=e,
                    parser=self.config.parser,
                )
            raise
        except Exception as e:
            self.logger.error(
                f"使用 {self.config.parser} 解析时发生未知错误: {str(e)}"
            )
            if callback_manager is not None:
                callback_manager.dispatch(
                    "on_parse_error",
                    file_path=callback_file,
                    error=e,
                    parser=self.config.parser,
                )
            raise

        # 解析完成，进行结果处理，生成doc_id并保存缓存
        msg = f"文件 {file_path} 解析完成！共提取到 {len(content_list)} 个内容块"
        self.logger.info(msg)

        if len(content_list) == 0:
            raise ValueError("解析失败：未提取到任何内容")

        # 根据解析出的内容生成唯一的 doc_id
        doc_id = self._generate_content_based_doc_id(content_list)

        # 将结果存入缓存，下次再跑同样的文件就秒开了
        await self._store_cached_result(
            cache_key, content_list, doc_id, file_path, parse_method, **kwargs
        )

        # 统计展示：如果display_stats = True，控制台会打印出具体提取到了多少文字、表格和图片
        if display_stats:
            self.logger.info("\n内容提取概况:")
            self.logger.info(f"* 内容块总数: {len(content_list)}")

            # 按内容类型（text, image, table 等）进行分类统计
            block_types: Dict[str, int] = {}
            for block in content_list:
                # 确保每一个块都是字典格式
                if isinstance(block, dict):
                    # 获取块的类型，拿不到就标记为 unknown
                    block_type = block.get("type", "unknown")
                    if isinstance(block_type, str):
                        # 核心计数逻辑：如果类型已存在，数量+1；如果第一次出现，设为 1
                        block_types[block_type] = block_types.get(block_type, 0) + 1

            self.logger.info("* 各类内容块分布:")
            for block_type, count in block_types.items():
                self.logger.info(f"  - {block_type}: {count}")

        # 最终完成回调
        # 检查是否配置了回调管理器（如果没人监听，就不发通知）
        if callback_manager is not None:
            # 计算整个解析阶段耗费了多少秒
            duration = time.time() - parse_start_time
            callback_manager.dispatch(
                "on_parse_complete",  # 事件名称
                file_path=callback_file,  # 告诉外部哪个文件解析完成了
                content_blocks=len(content_list),  # 告诉外部解析出了多少个内容块
                doc_id=doc_id,  # 告诉外部生成的 doc_id 是什么
                duration_seconds=duration,  # 告诉外部解析花了多少秒
            )

        return content_list, doc_id

    async def _process_multimodal_content(
            self,
            multimodal_items: List[Dict[str, Any]],
            file_path: str,
            doc_id: str,
            pipeline_status: Optional[Any] = None,
            pipeline_status_lock: Optional[Any] = None,
    ):
        """
        处理多模态内容（调用专门的处理器，如图片/表格处理器）

        参数说明：
            multimodal_items: 多模态项目列表（提取出的图片、表格等）
            file_path: 文件路径（用于引用）
            doc_id: 文档 ID，用于确保分片能正确关联到原文档
            pipeline_status: 流水线状态对象（用于前端显示进度）
            pipeline_status_lock: 流水线状态锁（防止多线程冲突）
        """

        # 检查是否有需要处理的内容
        if not multimodal_items:
            self.logger.debug("没有需要处理的多模态内容")
            return

        callback_manager = getattr(self, "callback_manager", None)
        mm_start_time = time.time()

        # 触发“多模态处理开始”的回调通知
        if callback_manager is not None:
            callback_manager.dispatch(
                "on_multimodal_start",
                file_path=file_path,
                item_count=len(multimodal_items),
                doc_id=doc_id,
            )

        # 状态检查逻辑：处理 LightRAG 的“早熟”问题
        try:
            existing_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            if existing_doc_status:
                # 检查多模态内容是否已经处理过（通过自定义标签 multimodal_processed）
                multimodal_processed = existing_doc_status.get(
                    "multimodal_processed", False
                )

                if multimodal_processed:
                    self.logger.info(
                        f"文档 {doc_id} 的多模态内容之前已处理完成，跳过"
                    )
                    return

                # 如果状态显示文本已处理完成 (DocStatus.PROCESSED)，但多模态还没做
                doc_status = existing_doc_status.get("status", "")
                if doc_status == DocStatus.PROCESSED and not multimodal_processed:
                    self.logger.info(
                        f"文档 {doc_id} 的文本处理已完成，但多模态内容仍需继续处理"
                    )

                # 如果状态显示文本和多模态都已处理完成
                elif doc_status == DocStatus.PROCESSED and multimodal_processed:
                    self.logger.info(
                        f"文档 {doc_id} 已完全处理（文本 + 多模态均完成）"
                    )
                    return

        except Exception as e:
            self.logger.debug(f"检查 {doc_id}状态时出错（可能是新文档）: {e}")
            # Continue with processing if cache check fails

        # Use ProcessorMixin's own batch processing that can handle multiple content types
        log_message = "正在开始多模态内容处理..."
        self.logger.info(log_message)
        if pipeline_status_lock and pipeline_status:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        try:
            # 确保 LightRAG 已准备就绪
            await self._ensure_lightrag_initialized()

            # 调用“类型感知”的批量处理器，它能自动把图片发给图片专家，表格发给表格专家
            await self._process_multimodal_content_batch_type_aware(
                multimodal_items=multimodal_items, file_path=file_path, doc_id=doc_id
            )

            # 标记该文档的多模态处理已正式完成
            await self._mark_multimodal_processing_complete(doc_id)

            log_message = "多模态内容处理完成"
            self.logger.info(log_message)
            # 更新状态锁中的消息
            if pipeline_status_lock and pipeline_status:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)

            # 触发“多模态处理完成”回调
            if callback_manager is not None:
                duration = time.time() - mm_start_time
                callback_manager.dispatch(
                    "on_multimodal_complete",
                    file_path=file_path,
                    processed_count=len(multimodal_items),
                    duration_seconds=duration,
                    doc_id=doc_id,
                )

        except Exception as e:
            # 异常处理：如果批量处理失败，尝试逐个处理（Fallback 机制）
            self.logger.error(f"多模态处理过程中出错: {e}")
            # Fallback to individual processing if batch processing fails
            self.logger.warning("正在切换到单项逐个处理模式")
            await self._process_multimodal_content_individual(
                multimodal_items, file_path, doc_id
            )

            # 即便是保底方案跑完，也要标记为完成
            await self._mark_multimodal_processing_complete(doc_id)

    async def _process_multimodal_content_individual(
            self, multimodal_items: List[Dict[str, Any]], file_path: str, doc_id: str
    ):
        """
        Process multimodal content individually (fallback method)

        Args:
            multimodal_items: List of multimodal items
            file_path: File path (for reference)
            doc_id: Document ID for proper chunk association
        """
        # Use full path or basename based on config
        file_name = self._get_file_reference(file_path)

        # Collect all chunk results for batch processing (similar to text content processing)
        all_chunk_results = []
        multimodal_chunk_ids = []

        # Get current text chunks count to set proper order indexes for multimodal chunks
        existing_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
        existing_chunks_count = (
            existing_doc_status.get("chunks_count", 0) if existing_doc_status else 0
        )

        for i, item in enumerate(multimodal_items):
            try:
                content_type = item.get("type", "unknown")
                self.logger.info(
                    f"Processing item {i + 1}/{len(multimodal_items)}: {content_type} content"
                )

                # Select appropriate processor
                processor = get_processor_for_type(self.modal_processors, content_type)

                if processor:
                    # Prepare item info for context extraction
                    item_info = {
                        "page_idx": item.get("page_idx", 0),
                        "index": i,
                        "type": content_type,
                    }

                    # Process content and get chunk results instead of immediately merging
                    (
                        enhanced_caption,
                        entity_info,
                        chunk_results,
                    ) = await processor.process_multimodal_content(
                        modal_content=item,
                        content_type=content_type,
                        file_path=file_name,
                        item_info=item_info,  # Pass item info for context extraction
                        batch_mode=True,
                        doc_id=doc_id,  # Pass doc_id for proper association
                        chunk_order_index=existing_chunks_count
                                          + i,  # Proper order index
                    )

                    # Collect chunk results for batch processing
                    all_chunk_results.extend(chunk_results)

                    # Extract chunk ID from the entity_info (actual chunk_id created by processor)
                    if entity_info and "chunk_id" in entity_info:
                        chunk_id = entity_info["chunk_id"]
                        multimodal_chunk_ids.append(chunk_id)

                    self.logger.info(
                        f"{content_type} processing complete: {entity_info.get('entity_name', 'Unknown')}"
                    )
                else:
                    self.logger.warning(
                        f"No suitable processor found for {content_type} type content"
                    )

            except Exception as e:
                self.logger.error(f"Error processing multimodal content: {str(e)}")
                self.logger.debug("Exception details:", exc_info=True)
                continue

        # Update doc_status to include multimodal chunks in the standard chunks_list
        if multimodal_chunk_ids:
            try:
                # Get current document status
                current_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)

                if current_doc_status:
                    existing_chunks_list = current_doc_status.get("chunks_list", [])
                    existing_chunks_count = current_doc_status.get("chunks_count", 0)

                    # Add multimodal chunks to the standard chunks_list
                    updated_chunks_list = existing_chunks_list + multimodal_chunk_ids
                    updated_chunks_count = existing_chunks_count + len(
                        multimodal_chunk_ids
                    )

                    # Update document status with integrated chunk list
                    await self.lightrag.doc_status.upsert(
                        {
                            doc_id: {
                                **current_doc_status,  # Keep existing fields
                                "chunks_list": updated_chunks_list,  # Integrated chunks list
                                "chunks_count": updated_chunks_count,  # Updated total count
                                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                            }
                        }
                    )

                    # Ensure doc_status update is persisted to disk
                    await self.lightrag.doc_status.index_done_callback()

                    self.logger.info(
                        f"Updated doc_status with {len(multimodal_chunk_ids)} multimodal chunks integrated into chunks_list"
                    )

            except Exception as e:
                self.logger.warning(
                    f"Error updating doc_status with multimodal chunks: {e}"
                )

        # Batch merge all multimodal content results (similar to text content processing)
        if all_chunk_results:
            from lightrag.operate import merge_nodes_and_edges
            from lightrag.kg.shared_storage import (
                get_namespace_data,
                get_pipeline_status_lock,
            )

            # Get pipeline status and lock from shared storage
            pipeline_status = await get_namespace_data("pipeline_status")
            pipeline_status_lock = get_pipeline_status_lock()

            await merge_nodes_and_edges(
                chunk_results=all_chunk_results,
                knowledge_graph_inst=self.lightrag.chunk_entity_relation_graph,
                entity_vdb=self.lightrag.entities_vdb,
                relationships_vdb=self.lightrag.relationships_vdb,
                global_config=self.lightrag.__dict__,
                full_entities_storage=self.lightrag.full_entities,
                full_relations_storage=self.lightrag.full_relations,
                doc_id=doc_id,
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=self.lightrag.llm_response_cache,
                current_file_number=1,
                total_files=1,
                file_path=file_name,
            )

            await self.lightrag._insert_done()

        self.logger.info("Individual multimodal content processing complete")

        # Mark multimodal content as processed
        await self._mark_multimodal_processing_complete(doc_id)

    async def _process_multimodal_content_batch_type_aware(
            self, multimodal_items: List[Dict[str, Any]], file_path: str, doc_id: str
    ):
        """
        类型感知的批量处理，根据内容类型选择正确的处理器。
        这是处理不同模态类型的修正实现。

        参数说明：
            multimodal_items: 包含不同类型的多模态项目列表。
            file_path: 用于引用溯源的文件路径。
            doc_id: 用于确保正确关联的文档 ID。
        """
        if not multimodal_items:
            self.logger.debug("No multimodal content to process")
            return

        # Get existing chunks count for proper order indexing
        try:
            existing_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            existing_chunks_count = (
                existing_doc_status.get("chunks_count", 0) if existing_doc_status else 0
            )
        except Exception:
            existing_chunks_count = 0

        # Use LightRAG's concurrency control
        semaphore = asyncio.Semaphore(getattr(self.lightrag, "max_parallel_insert", 2))

        # Progress tracking variables
        total_items = len(multimodal_items)
        completed_count = 0
        progress_lock = asyncio.Lock()

        # Log processing start
        self.logger.info(f"Starting to process {total_items} multimodal content items")

        # Stage 1: Concurrent generation of descriptions using correct processors for each type
        async def process_single_item_with_correct_processor(
                item: Dict[str, Any], index: int, file_path: str
        ):
            """Process single item using the correct processor for its type"""
            nonlocal completed_count
            async with semaphore:
                try:
                    content_type = item.get("type", "unknown")

                    # Select the correct processor based on content type
                    processor = get_processor_for_type(
                        self.modal_processors, content_type
                    )

                    if not processor:
                        self.logger.warning(
                            f"No processor found for type: {content_type}"
                        )
                        return None

                    item_info = {
                        "page_idx": item.get("page_idx", 0),
                        "index": index,
                        "type": content_type,
                    }

                    # Call the correct processor's description generation method
                    (
                        description,
                        entity_info,
                    ) = await processor.generate_description_only(
                        modal_content=item,
                        content_type=content_type,
                        item_info=item_info,
                        entity_name=None,  # Let LLM auto-generate
                    )

                    # Update progress (non-blocking)
                    async with progress_lock:
                        completed_count += 1
                        if (
                                completed_count % max(1, total_items // 10) == 0
                                or completed_count == total_items
                        ):
                            progress_percent = (completed_count / total_items) * 100
                            self.logger.info(
                                f"Multimodal chunk generation progress: {completed_count}/{total_items} ({progress_percent:.1f}%)"
                            )

                    return {
                        "index": index,
                        "content_type": content_type,
                        "description": description,
                        "entity_info": entity_info,
                        "original_item": item,
                        "item_info": item_info,
                        "chunk_order_index": existing_chunks_count + index,
                        "processor": processor,  # Keep reference to the processor used
                        "file_path": file_path,  # Add file_path to the result
                    }

                except Exception as e:
                    # Update progress even on error (non-blocking)
                    async with progress_lock:
                        completed_count += 1
                        if (
                                completed_count % max(1, total_items // 10) == 0
                                or completed_count == total_items
                        ):
                            progress_percent = (completed_count / total_items) * 100
                            self.logger.info(
                                f"Multimodal chunk generation progress: {completed_count}/{total_items} ({progress_percent:.1f}%)"
                            )

                    self.logger.error(
                        f"Error generating description for {content_type} item {index}: {e}"
                    )
                    return None

        # Process all items concurrently with correct processors
        tasks = [
            asyncio.create_task(
                process_single_item_with_correct_processor(item, i, file_path)
            )
            for i, item in enumerate(multimodal_items)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        multimodal_data_list = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Task failed: {result}")
                continue
            if result is not None:
                multimodal_data_list.append(result)

        if not multimodal_data_list:
            self.logger.warning("No valid multimodal descriptions generated")
            return

        self.logger.info(
            f"Generated descriptions for {len(multimodal_data_list)}/{len(multimodal_items)} multimodal items using correct processors"
        )

        # Stage 2: Convert to LightRAG chunks format
        lightrag_chunks = self._convert_to_lightrag_chunks_type_aware(
            multimodal_data_list, file_path, doc_id
        )

        # Stage 3: Store chunks to LightRAG storage
        await self._store_chunks_to_lightrag_storage_type_aware(lightrag_chunks)

        # Stage 3.5: Store multimodal main entities to entities_vdb and full_entities
        await self._store_multimodal_main_entities(
            multimodal_data_list, lightrag_chunks, file_path, doc_id
        )

        # Track chunk IDs for doc_status update
        chunk_ids = list(lightrag_chunks.keys())

        # Stage 4: Use LightRAG's batch entity relation extraction
        chunk_results = await self._batch_extract_entities_lightrag_style_type_aware(
            lightrag_chunks
        )

        # Stage 5: Add belongs_to relations (multimodal-specific)
        enhanced_chunk_results = await self._batch_add_belongs_to_relations_type_aware(
            chunk_results, multimodal_data_list
        )

        # Stage 6: Use LightRAG's batch merge
        await self._batch_merge_lightrag_style_type_aware(
            enhanced_chunk_results, file_path, doc_id
        )

        # Stage 7: Update doc_status with integrated chunks_list
        await self._update_doc_status_with_chunks_type_aware(doc_id, chunk_ids)

    def _convert_to_lightrag_chunks_type_aware(
            self, multimodal_data_list: List[Dict[str, Any]], file_path: str, doc_id: str
    ) -> Dict[str, Any]:
        """Convert multimodal data to LightRAG standard chunks format"""

        chunks = {}

        for data in multimodal_data_list:
            description = data["description"]
            entity_info = data["entity_info"]
            chunk_order_index = data["chunk_order_index"]
            content_type = data["content_type"]
            original_item = data["original_item"]

            # Apply the appropriate chunk template based on content type
            formatted_chunk_content = self._apply_chunk_template(
                content_type, original_item, description
            )

            # Generate chunk_id
            chunk_id = compute_mdhash_id(formatted_chunk_content, prefix="chunk-")

            # Calculate tokens
            tokens = len(self.lightrag.tokenizer.encode(formatted_chunk_content))

            # Use full path or basename based on config
            file_ref = self._get_file_reference(file_path)

            # Build LightRAG standard chunk format
            chunks[chunk_id] = {
                "content": formatted_chunk_content,  # Now uses the templated content
                "tokens": tokens,
                "full_doc_id": doc_id,
                "chunk_order_index": chunk_order_index,
                "file_path": file_ref,
                "llm_cache_list": [],  # LightRAG will populate this field
                # Multimodal-specific metadata
                "is_multimodal": True,
                "modal_entity_name": entity_info["entity_name"],
                "original_type": data["content_type"],
                "page_idx": data["item_info"].get("page_idx", 0),
            }

        self.logger.debug(
            f"Converted {len(chunks)} multimodal items to multimodal chunks format"
        )
        return chunks

    def _apply_chunk_template(
            self, content_type: str, original_item: Dict[str, Any], description: str
    ) -> str:
        """
        Apply the appropriate chunk template based on content type

        Args:
            content_type: Type of content (image, table, equation, generic)
            original_item: Original multimodal item data
            description: Enhanced description generated by the processor

        Returns:
            Formatted chunk content using the appropriate template
        """
        from raganything.prompt import PROMPTS

        try:
            if content_type == "image":
                image_path = original_item.get("img_path", "")
                captions = original_item.get(
                    "image_caption", original_item.get("img_caption", [])
                )
                footnotes = original_item.get(
                    "image_footnote", original_item.get("img_footnote", [])
                )

                return PROMPTS["image_chunk"].format(
                    image_path=image_path,
                    captions=", ".join(captions) if captions else "None",
                    footnotes=", ".join(footnotes) if footnotes else "None",
                    enhanced_caption=description,
                )

            elif content_type == "table":
                table_img_path = original_item.get("img_path", "")
                table_caption = original_item.get("table_caption", [])
                table_body = original_item.get("table_body", "")
                table_footnote = original_item.get("table_footnote", [])

                return PROMPTS["table_chunk"].format(
                    table_img_path=table_img_path,
                    table_caption=", ".join(table_caption) if table_caption else "None",
                    table_body=table_body,
                    table_footnote=", ".join(table_footnote)
                    if table_footnote
                    else "None",
                    enhanced_caption=description,
                )

            elif content_type == "equation":
                equation_text = original_item.get("text", "")
                equation_format = original_item.get("text_format", "")

                return PROMPTS["equation_chunk"].format(
                    equation_text=equation_text,
                    equation_format=equation_format,
                    enhanced_caption=description,
                )

            else:  # generic or unknown types
                content = str(original_item.get("content", original_item))

                return PROMPTS["generic_chunk"].format(
                    content_type=content_type.title(),
                    content=content,
                    enhanced_caption=description,
                )

        except Exception as e:
            self.logger.warning(
                f"Error applying chunk template for {content_type}: {e}"
            )
            # Fallback to just the description if template fails
            return description

    async def _store_chunks_to_lightrag_storage_type_aware(
            self, chunks: Dict[str, Any]
    ):
        """Store chunks to storage"""
        try:
            # Store in text_chunks storage (required for extract_entities)
            await self.lightrag.text_chunks.upsert(chunks)

            # Store in chunks vector database for retrieval
            await self.lightrag.chunks_vdb.upsert(chunks)

            self.logger.debug(f"Stored {len(chunks)} multimodal chunks to storage")

        except Exception as e:
            self.logger.error(f"Error storing chunks to storage: {e}")
            raise

    async def _store_multimodal_main_entities(
            self,
            multimodal_data_list: List[Dict[str, Any]],
            lightrag_chunks: Dict[str, Any],
            file_path: str,
            doc_id: str = None,
    ):
        """
        Store multimodal main entities to entities_vdb and full_entities.
        This ensures that entities like "TableName (table)" are properly indexed.

        Args:
            multimodal_data_list: List of processed multimodal data with entity info
            lightrag_chunks: Chunks in LightRAG format (already formatted with templates)
            file_path: File path for the entities
            doc_id: Document ID for full_entities storage
        """
        if not multimodal_data_list:
            return

        # Create entities_vdb entries for all multimodal main entities
        entities_to_store = {}

        # Use full path or basename based on config
        file_ref = self._get_file_reference(file_path)

        for data in multimodal_data_list:
            entity_info = data["entity_info"]
            entity_name = entity_info["entity_name"]
            description = data["description"]
            content_type = data["content_type"]
            original_item = data["original_item"]

            # Apply the same chunk template to get the formatted content
            formatted_chunk_content = self._apply_chunk_template(
                content_type, original_item, description
            )

            # Generate chunk_id using the formatted content (same as in _convert_to_lightrag_chunks)
            chunk_id = compute_mdhash_id(formatted_chunk_content, prefix="chunk-")

            # Generate entity_id using LightRAG's standard format
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")

            # Create entity data in LightRAG format
            entity_data = {
                "entity_name": entity_name,
                "entity_type": entity_info.get("entity_type", content_type),
                "content": entity_info.get("summary", description),
                "source_id": chunk_id,
                "file_path": file_ref,
            }

            entities_to_store[entity_id] = entity_data

        if entities_to_store:
            try:
                # Store entities in knowledge graph
                for entity_id, entity_data in entities_to_store.items():
                    entity_name = entity_data["entity_name"]

                    # Create node data for knowledge graph
                    node_data = {
                        "entity_id": entity_name,
                        "entity_type": entity_data["entity_type"],
                        "description": entity_data["content"],
                        "source_id": entity_data["source_id"],
                        "file_path": entity_data["file_path"],
                        "created_at": int(time.time()),
                    }

                    # Store in knowledge graph
                    await self.lightrag.chunk_entity_relation_graph.upsert_node(
                        entity_name, node_data
                    )

                # Store in entities_vdb
                await self.lightrag.entities_vdb.upsert(entities_to_store)
                await self.lightrag.entities_vdb.index_done_callback()

                # NEW: Store multimodal main entities in full_entities storage
                if doc_id and self.lightrag.full_entities:
                    await self._store_multimodal_entities_to_full_entities(
                        entities_to_store, doc_id
                    )

                self.logger.debug(
                    f"Stored {len(entities_to_store)} multimodal main entities to knowledge graph, entities_vdb, and full_entities"
                )

            except Exception as e:
                self.logger.error(f"Error storing multimodal main entities: {e}")
                raise

    async def _store_multimodal_entities_to_full_entities(
            self, entities_to_store: Dict[str, Any], doc_id: str
    ):
        """
        Store multimodal main entities to full_entities storage.

        Args:
            entities_to_store: Dictionary of entities to store
            doc_id: Document ID for grouping entities
        """
        try:
            # Get current full_entities data for this document
            current_doc_entities = await self.lightrag.full_entities.get_by_id(doc_id)

            if current_doc_entities is None:
                # Create new document entry
                entity_names = [
                    entity_data["entity_name"]
                    for entity_data in entities_to_store.values()
                ]
                doc_entities_data = {
                    "entity_names": entity_names,
                    "count": len(entity_names),
                    "update_time": int(time.time()),
                }
            else:
                # Update existing document entry while preserving any existing
                # metadata fields stored by the text pipeline.
                existing_entity_names = list(
                    current_doc_entities.get("entity_names", [])
                )
                seen_entity_names = set(existing_entity_names)

                for entity_data in entities_to_store.values():
                    entity_name = entity_data["entity_name"]
                    if entity_name not in seen_entity_names:
                        existing_entity_names.append(entity_name)
                        seen_entity_names.add(entity_name)

                doc_entities_data = {
                    **current_doc_entities,
                    "entity_names": existing_entity_names,
                    "count": len(existing_entity_names),
                    "update_time": int(time.time()),
                }

            # Store updated data
            await self.lightrag.full_entities.upsert({doc_id: doc_entities_data})
            await self.lightrag.full_entities.index_done_callback()

            self.logger.debug(
                f"Added {len(entities_to_store)} multimodal main entities to full_entities for doc {doc_id}"
            )

        except Exception as e:
            self.logger.error(
                f"Error storing multimodal entities to full_entities: {e}"
            )
            raise

    async def _batch_extract_entities_lightrag_style_type_aware(
            self, lightrag_chunks: Dict[str, Any]
    ) -> List[Tuple]:
        """Use LightRAG's extract_entities for batch entity relation extraction"""
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_pipeline_status_lock,
        )
        from lightrag.operate import extract_entities

        # Get pipeline status (consistent with LightRAG)
        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status_lock = get_pipeline_status_lock()

        # Directly use LightRAG's extract_entities
        chunk_results = await extract_entities(
            chunks=lightrag_chunks,
            global_config=self.lightrag.__dict__,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            llm_response_cache=self.lightrag.llm_response_cache,
            text_chunks_storage=self.lightrag.text_chunks,
        )

        self.logger.info(
            f"Extracted entities from {len(lightrag_chunks)} multimodal chunks"
        )
        return chunk_results

    async def _batch_add_belongs_to_relations_type_aware(
            self, chunk_results: List[Tuple], multimodal_data_list: List[Dict[str, Any]]
    ) -> List[Tuple]:
        """Add belongs_to relations for multimodal entities"""
        # Create mapping from chunk_id to modal_entity_name
        chunk_to_modal_entity = {}
        chunk_to_file_path = {}

        for data in multimodal_data_list:
            description = data["description"]
            content_type = data["content_type"]
            original_item = data["original_item"]

            # Use the same template formatting as in _convert_to_lightrag_chunks_type_aware
            formatted_chunk_content = self._apply_chunk_template(
                content_type, original_item, description
            )
            chunk_id = compute_mdhash_id(formatted_chunk_content, prefix="chunk-")

            chunk_to_modal_entity[chunk_id] = data["entity_info"]["entity_name"]
            chunk_to_file_path[chunk_id] = data.get("file_path", "multimodal_content")

        enhanced_chunk_results = []
        belongs_to_count = 0

        for maybe_nodes, maybe_edges in chunk_results:
            # Find corresponding modal_entity_name for this chunk
            chunk_id = None
            for nodes_dict in maybe_nodes.values():
                if nodes_dict:
                    chunk_id = nodes_dict[0].get("source_id")
                    break

            if chunk_id and chunk_id in chunk_to_modal_entity:
                modal_entity_name = chunk_to_modal_entity[chunk_id]
                file_path = chunk_to_file_path.get(chunk_id, "multimodal_content")

                # Add belongs_to relations for all extracted entities
                for entity_name in maybe_nodes.keys():
                    if entity_name != modal_entity_name:  # Avoid self-relation
                        belongs_to_relation = {
                            "src_id": entity_name,
                            "tgt_id": modal_entity_name,
                            "description": f"Entity {entity_name} belongs to {modal_entity_name}",
                            "keywords": "belongs_to,part_of,contained_in",
                            "source_id": chunk_id,
                            "weight": 10.0,
                            "file_path": file_path,
                        }

                        # Add to maybe_edges
                        edge_key = (entity_name, modal_entity_name)
                        if edge_key not in maybe_edges:
                            maybe_edges[edge_key] = []
                        maybe_edges[edge_key].append(belongs_to_relation)
                        belongs_to_count += 1

            enhanced_chunk_results.append((maybe_nodes, maybe_edges))

        self.logger.info(
            f"Added {belongs_to_count} belongs_to relations for multimodal entities"
        )
        return enhanced_chunk_results

    async def _batch_merge_lightrag_style_type_aware(
            self, enhanced_chunk_results: List[Tuple], file_path: str, doc_id: str = None
    ):
        """Use LightRAG's merge_nodes_and_edges for batch merge"""
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_pipeline_status_lock,
        )
        from lightrag.operate import merge_nodes_and_edges

        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status_lock = get_pipeline_status_lock()

        # Use full path or basename based on config
        file_ref = self._get_file_reference(file_path)

        await merge_nodes_and_edges(
            chunk_results=enhanced_chunk_results,
            knowledge_graph_inst=self.lightrag.chunk_entity_relation_graph,
            entity_vdb=self.lightrag.entities_vdb,
            relationships_vdb=self.lightrag.relationships_vdb,
            global_config=self.lightrag.__dict__,
            full_entities_storage=self.lightrag.full_entities,
            full_relations_storage=self.lightrag.full_relations,
            doc_id=doc_id,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            llm_response_cache=self.lightrag.llm_response_cache,
            current_file_number=1,
            total_files=1,
            file_path=file_ref,
        )

        await self.lightrag._insert_done()

    async def _update_doc_status_with_chunks_type_aware(
            self, doc_id: str, chunk_ids: List[str]
    ):
        """Update document status with multimodal chunks"""
        try:
            # Get current document status
            current_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)

            if current_doc_status:
                existing_chunks_list = current_doc_status.get("chunks_list", [])
                existing_chunks_count = current_doc_status.get("chunks_count", 0)

                # Add multimodal chunks to the standard chunks_list
                updated_chunks_list = existing_chunks_list + chunk_ids
                updated_chunks_count = existing_chunks_count + len(chunk_ids)

                # Update document status with integrated chunk list
                await self.lightrag.doc_status.upsert(
                    {
                        doc_id: {
                            **current_doc_status,  # Keep existing fields
                            "chunks_list": updated_chunks_list,  # Integrated chunks list
                            "chunks_count": updated_chunks_count,  # Updated total count
                            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                        }
                    }
                )

                # Ensure doc_status update is persisted to disk
                await self.lightrag.doc_status.index_done_callback()

                self.logger.info(
                    f"Updated doc_status: added {len(chunk_ids)} multimodal chunks to standard chunks_list "
                    f"(total chunks: {updated_chunks_count})"
                )

        except Exception as e:
            self.logger.warning(
                f"Error updating doc_status with multimodal chunks: {e}"
            )

    async def _mark_multimodal_processing_complete(self, doc_id: str):
        """Mark multimodal content processing as complete in the document status."""
        try:
            current_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            if current_doc_status:
                await self.lightrag.doc_status.upsert(
                    {
                        doc_id: {
                            **current_doc_status,
                            "multimodal_processed": True,
                            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                        }
                    }
                )
                await self.lightrag.doc_status.index_done_callback()
                self.logger.debug(
                    f"Marked multimodal content processing as complete for document {doc_id}"
                )
        except Exception as e:
            self.logger.warning(
                f"Error marking multimodal processing as complete for document {doc_id}: {e}"
            )

    async def is_document_fully_processed(self, doc_id: str) -> bool:
        """
        Check if a document is fully processed (both text and multimodal content).

        Args:
            doc_id: Document ID to check

        Returns:
            bool: True if both text and multimodal content are processed
        """
        try:
            doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            if not doc_status:
                return False

            text_processed = doc_status.get("status") == DocStatus.PROCESSED
            multimodal_processed = doc_status.get("multimodal_processed", False)

            return text_processed and multimodal_processed

        except Exception as e:
            self.logger.error(
                f"Error checking document processing status for {doc_id}: {e}"
            )
            return False

    async def get_document_processing_status(self, doc_id: str) -> Dict[str, Any]:
        """
        Get detailed processing status for a document.

        Args:
            doc_id: Document ID to check

        Returns:
            Dict with processing status details
        """
        try:
            doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            if not doc_status:
                return {
                    "exists": False,
                    "text_processed": False,
                    "multimodal_processed": False,
                    "fully_processed": False,
                    "chunks_count": 0,
                }

            text_processed = doc_status.get("status") == DocStatus.PROCESSED
            multimodal_processed = doc_status.get("multimodal_processed", False)
            fully_processed = text_processed and multimodal_processed

            return {
                "exists": True,
                "text_processed": text_processed,
                "multimodal_processed": multimodal_processed,
                "fully_processed": fully_processed,
                "chunks_count": doc_status.get("chunks_count", 0),
                "chunks_list": doc_status.get("chunks_list", []),
                "status": doc_status.get("status", ""),
                "updated_at": doc_status.get("updated_at", ""),
                "raw_status": doc_status,
            }

        except Exception as e:
            self.logger.error(
                f"Error getting document processing status for {doc_id}: {e}"
            )
            return {
                "exists": False,
                "error": str(e),
                "text_processed": False,
                "multimodal_processed": False,
                "fully_processed": False,
                "chunks_count": 0,
            }

    async def process_document_complete(
            self,
            file_path: str,
            output_dir: str = None,
            parse_method: str = None,
            display_stats: bool = None,
            split_by_character: str | None = None,
            split_by_character_only: bool = False,
            doc_id: str | None = None,
            file_name: str | None = None,
            **kwargs,
    ):
        """
        完整文档处理流水线

        参数说明：
            file_path: 待处理文件的路径
            output_dir: 输出目录（默认为配置中的 config.parser_output_dir）
            parse_method: 解析方法（默认为配置中的 config.parse_method，如 'auto', 'ocr'）
            display_stats: 是否显示内容统计信息（默认为配置中的 config.display_content_stats）
            split_by_character: 可选的特定分隔符，用于对提取出的文本进行切分
            split_by_character_only: 如果设为 True，则仅使用指定的字符进行切分，忽略默认切分规则
            doc_id: 可选的文档 ID。如果不提供，系统会根据文件内容自动生成唯一的 ID
            **kwargs: 解析器的额外参数。例如：lang（语言）、device（设备，如 cpu/cuda）、start_page/end_page（起始/结束页码）、
            formula（公式识别开关）、table（表格识别开关）、backend（后端引擎）、source（来源标识）。
        """

        callback_manager = getattr(self, "callback_manager", None)  # 获取回调管理器，用于监控进度
        doc_start_time = time.time()  # 记录开始时间
        stage = "parse"  # 当前阶段标识：解析阶段

        try:
            # 确保底层 LightRAG 引擎已初始化（如向量数据库连接等）
            await self._ensure_lightrag_initialized()

            # 如果调用时没传参数，则使用配置文件中的默认值
            if output_dir is None:
                # 默认值./output
                output_dir = self.config.parser_output_dir
            if parse_method is None:
                # 默认值auto
                parse_method = self.config.parse_method
            if display_stats is None:
                # 默认值True
                display_stats = self.config.display_content_stats

            self.logger.info(f"正在启动完整文档处理流程: {file_path}")

            # 第一步：解析文档
            # 调用具体的 parse_document 方法，将文件转为统一的列表格式，实现细节在 parser.py 中
            # content_list: MinerU 解析出的内容列表
            # content_based_doc_id: 基于内容生成的文档 ID
            content_list, content_based_doc_id = await self.parse_document(
                file_path, output_dir, parse_method, display_stats, **kwargs
            )

            # 确定文档唯一标识 ID（如果没传，则使用基于内容的哈希 ID，防止重复入库）
            if doc_id is None:
                doc_id = content_based_doc_id

            # 第二步：内容分类
            # 将解析出的混合内容拆分为：
            # text_content: 拼接后的完整文本字符串
            # multimodal_items: 包含图片、表格、公式等非纯文本项的列表
            text_content, multimodal_items = separate_content(content_list)

            # 第2.5步：上下文感知设置
            # 为多模态处理设置内容来源。这是为了让 AI 知道图片出现在哪一页、哪段文字附近
            # 这样 AI 就能根据上下文理解图片的内容，而不是孤立地看
            if hasattr(self, "set_content_source_for_context") and multimodal_items:
                self.logger.info(
                    "正在为多模态处理设置上下文来源..."
                )
                self.set_content_source_for_context(
                    content_list, self.config.content_format
                )

            # 第三步：纯文本内容入库
            stage = "text_insert"  # 当前阶段标识：文本入库阶段
            if text_content.strip():
                if file_name is None:
                    # 获取引用文件名（全路径或仅文件名，取决于RagAnythingConfig.use_full_path）
                    file_name = self._get_file_reference(file_path)

                # 触发“开始插入文本”的回调钩子
                if callback_manager is not None:
                    callback_manager.dispatch(
                        "on_text_insert_start",
                        file_path=file_name,
                        text_length=len(text_content),
                        doc_id=doc_id,
                    )
                insert_start = time.time()
                # 将文本切片并转化为向量，存入向量数据库
                await insert_text_content(
                    self.lightrag,
                    input=text_content,
                    file_paths=file_name,
                    split_by_character=split_by_character,
                    split_by_character_only=split_by_character_only,
                    ids=doc_id,
                )
                # 触发“文本插入完成”的回调，记录耗时
                if callback_manager is not None:
                    insert_duration = time.time() - insert_start
                    callback_manager.dispatch(
                        "on_text_insert_complete",
                        file_path=file_name,
                        duration_seconds=insert_duration,
                        doc_id=doc_id,
                    )
            else:
                # 如果文本内容为空，则确定文件引用名称
                if file_name is None:
                    file_name = self._get_file_reference(file_path)

            # 第四步：多模态内容处理
            stage = "multimodal"  # 当前阶段标识：多模态处理阶段
            if multimodal_items:
                # 处理多模态内容
                await self._process_multimodal_content(
                    multimodal_items, file_name, doc_id
                )
            else:
                # 如果文档中没有多模态内容，则标记多模态处理已完成
                # 这确保文档状态正确地反映了所有处理的完成情况
                await self._mark_multimodal_processing_complete(doc_id)
                self.logger.debug(
                    f"文档 {doc_id}, "
                    "中未发现多模态内容，已标记处理完成",
                )

        # 异常捕获：如果处理失败，通知回调管理器具体的失败阶段（parse/text_insert/multimodal）
        except Exception as exc:
            if callback_manager is not None:
                callback_manager.dispatch(
                    "on_document_error",
                    file_path=str(file_path),
                    doc_id=doc_id,
                    stage=stage,
                    error=exc,
                )
            raise  # 重新抛出异常供上层处理

        self.logger.info(f"文档 {file_path} 处理全部完成！")
        # 记录总耗时并触发完成回调
        if callback_manager is not None:
            duration = time.time() - doc_start_time
            callback_manager.dispatch(
                "on_document_complete",
                file_path=str(file_path),
                doc_id=doc_id,
                duration_seconds=duration,
            )

    async def process_document_complete_lightrag_api(
            self,
            file_path: str,
            output_dir: str = None,
            parse_method: str = None,
            display_stats: bool = None,
            split_by_character: str | None = None,
            split_by_character_only: bool = False,
            doc_id: str | None = None,
            scheme_name: str | None = None,
            parser: str | None = None,
            **kwargs,
    ):
        """
        API exclusively for LightRAG calls: Complete document processing workflow

        Args:
            file_path: Path to the file to process
            output_dir: output directory (defaults to config.parser_output_dir)
            parse_method: Parse method (defaults to config.parse_method)
            display_stats: Whether to display content statistics (defaults to config.display_content_stats)
            split_by_character: Optional character to split the text by
            split_by_character_only: If True, split only by the specified character
            doc_id: Optional document ID, if not provided will be generated from content
            **kwargs: Additional parameters for parser (e.g., lang, device, start_page, end_page, formula, table, backend, source)
        """
        # Use full path or basename based on config
        file_name = self._get_file_reference(file_path)
        doc_pre_id = f"doc-pre-{file_name}"
        pipeline_status = None
        pipeline_status_lock = None

        if parser:
            self.config.parser = parser

        current_doc_status = await self.lightrag.doc_status.get_by_id(doc_pre_id)

        try:
            # Ensure LightRAG is initialized
            result = await self._ensure_lightrag_initialized()
            if not result["success"]:
                await self.lightrag.doc_status.upsert(
                    {
                        doc_pre_id: {
                            **current_doc_status,
                            "status": DocStatus.FAILED,
                            "error_msg": result["error"],
                        }
                    }
                )
                return False

            # Use config defaults if not provided
            if output_dir is None:
                output_dir = self.config.parser_output_dir
            if parse_method is None:
                parse_method = self.config.parse_method
            if display_stats is None:
                display_stats = self.config.display_content_stats

            self.logger.info(f"Starting complete document processing: {file_path}")

            # Initialize doc status
            current_doc_status = await self.lightrag.doc_status.get_by_id(doc_pre_id)
            if not current_doc_status:
                await self.lightrag.doc_status.upsert(
                    {
                        doc_pre_id: {
                            "status": DocStatus.READY,
                            "content": "",
                            "error_msg": "",
                            "content_summary": "",
                            "multimodal_content": [],
                            "scheme_name": scheme_name,
                            "content_length": 0,
                            "created_at": "",
                            "updated_at": "",
                            "file_path": file_name,
                        }
                    }
                )
                current_doc_status = await self.lightrag.doc_status.get_by_id(
                    doc_pre_id
                )

            from lightrag.kg.shared_storage import (
                get_namespace_data,
                get_pipeline_status_lock,
            )

            pipeline_status = await get_namespace_data("pipeline_status")
            pipeline_status_lock = get_pipeline_status_lock()

            # Set processing status
            async with pipeline_status_lock:
                pipeline_status.update({"scan_disabled": True})
                pipeline_status["history_messages"].append("Now is not allowed to scan")

            await self.lightrag.doc_status.upsert(
                {
                    doc_pre_id: {
                        **current_doc_status,
                        "status": DocStatus.HANDLING,
                        "error_msg": "",
                    }
                }
            )

            content_list = []
            content_based_doc_id = ""

            try:
                # Step 1: Parse document
                content_list, content_based_doc_id = await self.parse_document(
                    file_path, output_dir, parse_method, display_stats, **kwargs
                )
            except MineruExecutionError as e:
                error_message = e.error_msg
                if isinstance(e.error_msg, list):
                    error_message = "\n".join(e.error_msg)
                await self.lightrag.doc_status.upsert(
                    {
                        doc_pre_id: {
                            **current_doc_status,
                            "status": DocStatus.FAILED,
                            "error_msg": error_message,
                        }
                    }
                )
                self.logger.info(
                    f"Error processing document {file_path}: MineruExecutionError"
                )
                return False
            except Exception as e:
                await self.lightrag.doc_status.upsert(
                    {
                        doc_pre_id: {
                            **current_doc_status,
                            "status": DocStatus.FAILED,
                            "error_msg": str(e),
                        }
                    }
                )
                self.logger.info(f"Error processing document {file_path}: {str(e)}")
                return False

            # Use provided doc_id or fall back to content-based doc_id
            if doc_id is None:
                doc_id = content_based_doc_id

            # Step 2: Separate text and multimodal content
            text_content, multimodal_items = separate_content(content_list)

            # Step 2.5: Set content source for context extraction in multimodal processing
            if hasattr(self, "set_content_source_for_context") and multimodal_items:
                self.logger.info(
                    "Setting content source for context-aware multimodal processing..."
                )
                self.set_content_source_for_context(
                    content_list, self.config.content_format
                )

            # Step 3: Insert pure text content and multimodal content with all parameters
            if text_content.strip():
                await insert_text_content_with_multimodal_content(
                    self.lightrag,
                    input=text_content,
                    multimodal_content=multimodal_items,
                    file_paths=file_name,
                    split_by_character=split_by_character,
                    split_by_character_only=split_by_character_only,
                    ids=doc_id,
                    scheme_name=scheme_name,
                )

            self.logger.info(f"Document {file_path} processing completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {str(e)}")
            self.logger.debug("Exception details:", exc_info=True)

            # Update doc status to Failed
            await self.lightrag.doc_status.upsert(
                {
                    doc_pre_id: {
                        **current_doc_status,
                        "status": DocStatus.FAILED,
                        "error_msg": str(e),
                    }
                }
            )
            await self.lightrag.doc_status.index_done_callback()

            # Update pipeline status
            if pipeline_status_lock and pipeline_status:
                try:
                    async with pipeline_status_lock:
                        pipeline_status.update({"scan_disabled": False})
                        error_msg = (
                            f"RAGAnything processing failed for {file_name}: {str(e)}"
                        )
                        pipeline_status["latest_message"] = error_msg
                        pipeline_status["history_messages"].append(error_msg)
                        pipeline_status["history_messages"].append(
                            "Now is allowed to scan"
                        )
                except Exception as pipeline_update_error:
                    self.logger.error(
                        f"Failed to update pipeline status: {pipeline_update_error}"
                    )

            return False

        finally:
            async with pipeline_status_lock:
                pipeline_status.update({"scan_disabled": False})
                pipeline_status["latest_message"] = (
                    f"RAGAnything processing completed for {file_name}"
                )
                pipeline_status["history_messages"].append(
                    f"RAGAnything processing completed for {file_name}"
                )
                pipeline_status["history_messages"].append("Now is allowed to scan")

    async def insert_content_list(
            self,
            content_list: List[Dict[str, Any]],
            file_path: str = "unknown_document",
            split_by_character: str | None = None,
            split_by_character_only: bool = False,
            doc_id: str | None = None,
            display_stats: bool = None,
    ):
        """
        Insert content list directly without document parsing

        Args:
            content_list: Pre-parsed content list containing text and multimodal items.
                         Each item should be a dictionary with the following structure:
                         - Text: {"type": "text", "text": "content", "page_idx": 0}
                         - Image: {"type": "image", "img_path": "/absolute/path/to/image.jpg",
                                  "image_caption": ["caption"], "image_footnote": ["note"], "page_idx": 1}
                         - Table: {"type": "table", "table_body": "markdown table",
                                  "table_caption": ["caption"], "table_footnote": ["note"], "page_idx": 2}
                         - Equation: {"type": "equation", "latex": "LaTeX formula",
                                     "text": "description", "page_idx": 3}
                         - Generic: {"type": "custom_type", "content": "any content", "page_idx": 4}
            file_path: Reference file path/name for citation (defaults to "unknown_document")
            split_by_character: Optional character to split the text by
            split_by_character_only: If True, split only by the specified character
            doc_id: Optional document ID, if not provided will be generated from content
            display_stats: Whether to display content statistics (defaults to config.display_content_stats)

        Note:
            - img_path must be an absolute path to the image file
            - page_idx represents the page number where the content appears (0-based indexing)
            - Items are processed in the order they appear in the list
        """
        callback_manager = getattr(self, "callback_manager", None)
        doc_start_time = time.time()

        # Ensure LightRAG is initialized
        await self._ensure_lightrag_initialized()

        # Use config defaults if not provided
        if display_stats is None:
            display_stats = self.config.display_content_stats

        self.logger.info(
            f"Starting direct content list insertion for: {file_path} ({len(content_list)} items)"
        )

        # Generate doc_id based on content if not provided
        if doc_id is None:
            doc_id = self._generate_content_based_doc_id(content_list)

        # Display content statistics if requested
        if display_stats:
            self.logger.info("\nContent Information:")
            self.logger.info(f"* Total blocks in content_list: {len(content_list)}")

            # Count elements by type
            block_types: Dict[str, int] = {}
            for block in content_list:
                if isinstance(block, dict):
                    block_type = block.get("type", "unknown")
                    if isinstance(block_type, str):
                        block_types[block_type] = block_types.get(block_type, 0) + 1

            self.logger.info("* Content block types:")
            for block_type, count in block_types.items():
                self.logger.info(f"  - {block_type}: {count}")

        # Step 1: Separate text and multimodal content
        text_content, multimodal_items = separate_content(content_list)

        # Step 1.5: Set content source for context extraction in multimodal processing
        if hasattr(self, "set_content_source_for_context") and multimodal_items:
            self.logger.info(
                "Setting content source for context-aware multimodal processing..."
            )
            self.set_content_source_for_context(
                content_list, self.config.content_format
            )

        # Step 2: Insert pure text content with all parameters
        if text_content.strip():
            # Use full path or basename based on config
            file_ref = self._get_file_reference(file_path)
            if callback_manager is not None:
                callback_manager.dispatch(
                    "on_text_insert_start",
                    file_path=file_ref,
                    text_length=len(text_content),
                    doc_id=doc_id,
                )
            insert_start = time.time()
            await insert_text_content(
                self.lightrag,
                input=text_content,
                file_paths=file_ref,
                split_by_character=split_by_character,
                split_by_character_only=split_by_character_only,
                ids=doc_id,
            )
            if callback_manager is not None:
                insert_duration = time.time() - insert_start
                callback_manager.dispatch(
                    "on_text_insert_complete",
                    file_path=file_ref,
                    duration_seconds=insert_duration,
                    doc_id=doc_id,
                )
        else:
            # Determine file reference even if no text content
            file_ref = self._get_file_reference(file_path)

        # Step 3: Process multimodal content (using specialized processors)
        if multimodal_items:
            await self._process_multimodal_content(multimodal_items, file_ref, doc_id)
        else:
            # If no multimodal content, mark multimodal processing as complete
            # This ensures the document status properly reflects completion of all processing
            await self._mark_multimodal_processing_complete(doc_id)
            self.logger.debug(
                f"No multimodal content found in document {doc_id}, marked multimodal processing as complete"
            )

        self.logger.info(f"Content list insertion complete for: {file_path}")
        if callback_manager is not None:
            duration = time.time() - doc_start_time
            callback_manager.dispatch(
                "on_document_complete",
                file_path=file_path,
                doc_id=doc_id,
                duration_seconds=duration,
            )
