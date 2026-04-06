"""
RAGAnything 的查询功能

包含所有针对文本查询和多模态查询的相关方法
"""

import json # 处理json数据，特别是在生成缓存键时需要将查询参数序列化为字符串
import hashlib # 生成查询参数的哈希值，用于创建唯一的缓存键
import re # 正则表达式，用于在查询文本中识别和处理图片路径
import time
from typing import Dict, List, Any # 类型提示，增强代码可读性和维护性
from pathlib import Path
from lightrag import QueryParam # LightRAG 的查询参数类，用于配置查询行为
from lightrag.utils import always_get_an_event_loop # 确保在异步环境中有一个事件循环可用，特别是在某些环境（如 Jupyter Notebook）中可能没有默认事件循环时使用
from raganything.prompt import PROMPTS
from raganything.utils import (
    get_processor_for_type,
    encode_image_to_base64,
    validate_image_file,
)


class QueryMixin:
    """QueryMixin 类：包含 RAGAnything 的查询功能"""

    def _generate_multimodal_cache_key(
            self, query: str, multimodal_content: List[Dict[str, Any]], mode: str, **kwargs
    ) -> str:
        """
        Generate cache key for multimodal query

        Args:
            query: Base query text
            multimodal_content: List of multimodal content
            mode: Query mode
            **kwargs: Additional parameters

        Returns:
            str: Cache key hash
        """

        """
        生成多模态查询的缓存键

        参数说明：
            query: 基础查询文本
            multimodal_content: 多模态内容列表，每个元素包含内容类型和相关信息
            mode: 查询模式（如 "local", "global", "hybrid", "naive", "mix", "bypass"）
            **kwargs: 其他查询参数，将传递给 QueryParam 配置对象

        返回：
            str: 生成的缓存键哈希值，用于唯一标识具有相同查询参数的查询结果
        """
        # Create a normalized representation of the query parameters
        # 创建一个规范化的查询参数表示，确保相同的查询内容即使在格式上有细微差异（如空格、字段顺序等）也能生成相同的缓存键
        cache_data = {
            "query": query.strip(), # 去除查询文本的前后空格，避免因为多余空格导致缓存键不同
            "mode": mode,
        }

        # Normalize multimodal content for stable caching
        # 规范化多模态内容，确保即使在格式上有差异（如字段顺序、文件路径等）也能生成相同的缓存键
        normalized_content = []
        if multimodal_content: # 如果提供了多模态内容，则进行规范化处理
            for item in multimodal_content:
                if isinstance(item, dict): # 只处理字典类型的内容项，其他类型直接添加到规范化列表中
                    normalized_item = {}
                    for key, value in item.items():
                        # For file paths, use basename to make cache more portable
                        # 对于文件路径，使用 basename（文件名）来生成缓存键，使其更具可移植性，避免因为不同环境中的绝对路径差异导致缓存键不同
                        if key in [
                            "img_path",
                            "image_path",
                            "file_path",
                        ] and isinstance(value, str):
                            normalized_item[key] = Path(value).name
                        # For large content, create a hash instead of storing directly
                        # 对于大内容，创建哈希值而不是直接存储
                        elif (
                                key in ["table_data", "table_body"]
                                and isinstance(value, str)
                                and len(value) > 200
                        ):
                            normalized_item[f"{key}_hash"] = hashlib.md5(
                                value.encode()
                            ).hexdigest()
                        else:
                            normalized_item[key] = value
                    normalized_content.append(normalized_item)
                else:
                    normalized_content.append(item)

        cache_data["multimodal_content"] = normalized_content

        # Add relevant kwargs to cache data
        # 将相关的查询参数添加到缓存数据中，这些参数会影响查询结果，因此需要包含在缓存键的生成中。只选择对查询结果有实际影响的参数，避免因为无关参数的差异导致缓存键不同
        relevant_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
               in [
                   "stream",
                   "response_type",
                   "top_k",
                   "max_tokens",
                   "temperature",
                   # "only_need_context",
                   # "only_need_prompt",
               ]
        }
        cache_data.update(relevant_kwargs)

        # Generate hash from the cache data
        # 从缓存数据生成哈希值
        cache_str = json.dumps(cache_data, sort_keys=True, ensure_ascii=False) # 将缓存数据转换为字符串，使用 sort_keys=True 确保字典的键顺序一致，使用 ensure_ascii=False 保持非 ASCII 字符的原样输出
        cache_hash = hashlib.md5(cache_str.encode()).hexdigest() # 生成 MD5 哈希值，作为缓存键的一部分

        return f"multimodal_query:{cache_hash}"

    async def aquery(
            self, query: str, mode: str = "mix", system_prompt: str | None = None, **kwargs
    ) -> str:
        """
        纯文本查询 - 直接调用 LightRAG 的查询功能

        参数说明：
            query: 查询文本
            mode: 查询模式 ("local", "global", "hybrid", "naive", "mix", "bypass")
            system_prompt: 可选的系统提示词
            **kwargs: 其他查询参数，将传递给 QueryParam
                - vlm_enhanced: bool 类型，当存在 vision_model_func 时默认为 True
                如果为 True，将解析检索到的上下文中的图片路径,
                并将其替换为 Base64 编码的图像，以便进行视觉模型（VLM）处理

        返回：
            str: 查询结果字符串
        """

        # 依赖检查：如果没有知识库实例，就没法查。必须先处理文档或提供实例
        if self.lightrag is None:
            raise ValueError(
                "没有可用的 LightRAG 实例。请先处理文档或提供预初始化的实例."
            )

        # 视觉增强判断逻辑
        # 从 kwargs 中取出 vlm_enhanced 参数（如果用户没传，则为 None）
        vlm_enhanced = kwargs.pop("vlm_enhanced", None)

        # 如果用户没指定，则自动判断：只要系统配置了视觉模型函数，就默认开启增强模式
        if vlm_enhanced is None:
            vlm_enhanced = (
                    hasattr(self, "vision_model_func")
                    and self.vision_model_func is not None
            )

        # 路由分发：如果开启了视觉增强且模型可用，跳转到【视觉增强查询】分支
        if (
                vlm_enhanced
                and hasattr(self, "vision_model_func")
                and self.vision_model_func
        ):
            # 这会去读图，并将图片转成 Base64 发给多模态大模型
            return await self.aquery_vlm_enhanced(
                query, mode=mode, system_prompt=system_prompt, **kwargs
            )
        # 如果用户强行要求增强模式，但系统没配视觉模型，则打印警告并降级处理
        elif vlm_enhanced and (
                not hasattr(self, "vision_model_func") or not self.vision_model_func
        ):
            self.logger.warning(
                "请求了 VLM 视觉增强查询，但 vision_model_func 不可用，正在回退到普通查询模式"
            )

        # 回调与性能监控
        callback_manager = getattr(self, "callback_manager", None)
        query_start_time = time.time()

        if callback_manager is not None:
            callback_manager.dispatch(
                "on_query_start",
                query=query,
                mode=mode,
            )

        # 构建标准 LightRAG 查询参数
        query_param = QueryParam(mode=mode, **kwargs)

        self.logger.info(f"正在执行文本查询: {query[:100]}...")
        self.logger.info(f"查询模式: {mode}")

        try:
            # 执行 LightRAG 的异步查询
            # 这是纯文本检索路径，适合回答“文档中提到了哪些技术指标？”这类问题
            result = await self.lightrag.aquery(
                query, param=query_param, system_prompt=system_prompt
            )
        except Exception as exc:
            # 异常捕获并触发回调
            if callback_manager is not None:
                callback_manager.dispatch(
                    "on_query_error",
                    query=query,
                    mode=mode,
                    error=exc,
                )
            raise

        self.logger.info("文本查询完成")
        if callback_manager is not None:
            duration = time.time() - query_start_time
            result_len = len(result) if isinstance(result, str) else 0
            callback_manager.dispatch(
                "on_query_complete",
                query=query,
                mode=mode,
                duration_seconds=duration,
                result_length=result_len,
            )
        return result

    async def aquery_with_multimodal(
            self,
            query: str,
            multimodal_content: List[Dict[str, Any]] = None,
            mode: str = "mix",
            **kwargs,
    ) -> str:
        """
        多模态查询 - 结合文本和多模态内容进行提问

        参数说明：
        query: 基础查询文本（用户的提问）。
        multimodal_content: 多模态内容列表，每个元素包含：
            - type: 内容类型（如 "image"、"table"、"equation" 等）。
            - 其他字段取决于类型（例如 img_path 图片路径、table_data 表格数据、latex 公式等）。
        mode: 查询模式（"local", "global", "hybrid", "naive", "mix", "bypass"）。
        **kwargs: 其他查询参数，将传递给 QueryParam 配置对象。

        返回：
            str: 查询结果（AI 的回答）。

        代码示例：
            # 1. 纯文本查询（不带附件）
                result = await rag.query_with_multimodal("什么是机器学习？")

            # 2. 图片查询（带本地图片路径）
                result = await rag.query_with_multimodal(
                    "分析这张图片中的内容",
                     multimodal_content=[{
                        "type": "image",
                        "img_path": "./image.jpg"
                    }]
                )

            # 3. 表格查询（带原始 CSV 字符串）
                result = await rag.query_with_multimodal(
                    "分析这个表格中的数据趋势",
                    multimodal_content=[{
                        "type": "table",
                        "table_data": "姓名,年龄\n爱丽丝,25\n鲍勃,30"
                    }]
                )
        """
        # 启动检查：确保底层 LightRAG 引擎已经初始化
        await self._ensure_lightrag_initialized()

        self.logger.info(f"正在执行多模态查询: {query[:100]}...")
        self.logger.info(f"查询模式: {mode}")

        # 如果没有提供任何多模态内容，则退回到普通的纯文本查询
        if not multimodal_content:
            self.logger.info("未提供多模态内容，执行普通文本查询")
            return await self.aquery(query, mode=mode, **kwargs)

        # 根据问题、多模态内容、模式等生成一个唯一的哈希值（ID）
        # 这样可以检查是否已经处理过完全相同的查询
        cache_key = self._generate_multimodal_cache_key(
            query, multimodal_content, mode, **kwargs
        )

        # 检查缓存：如果已经处理过完全相同的查询，则直接返回缓存结果
        cached_result = None
        if (
                hasattr(self, "lightrag")
                and self.lightrag
                and hasattr(self.lightrag, "llm_response_cache")
                and self.lightrag.llm_response_cache
        ):
            if self.lightrag.llm_response_cache.global_config.get(
                    "enable_llm_cache", True
            ):
                try:
                    cached_result = await self.lightrag.llm_response_cache.get_by_id(
                        cache_key
                    )
                    if cached_result and isinstance(cached_result, dict):
                        result_content = cached_result.get("return")
                        if result_content:
                            self.logger.info(
                                f"多模态查询命中缓存: {cache_key[:16]}..."
                            )
                            return result_content
                except Exception as e:
                    self.logger.debug(f"访问多模态查询缓存出错: {e}")

        # 处理多模态内容，将其转化为“增强后的查询文本”
        enhanced_query = await self._process_multimodal_query_content(
            query, multimodal_content
        )

        self.logger.info(
            f"已生成增强查询，长度: {len(enhanced_query)} 字符"
        )

        # 执行查询
        result = await self.aquery(enhanced_query, mode=mode, **kwargs)

        # 写入缓存
        if (
                hasattr(self, "lightrag")
                and self.lightrag
                and hasattr(self.lightrag, "llm_response_cache")
                and self.lightrag.llm_response_cache
        ):
            if self.lightrag.llm_response_cache.global_config.get(
                    "enable_llm_cache", True
            ):
                try:
                    # 构建缓存条目：记录原始问题、模态数量等信息
                    cache_entry = {
                        "return": result,
                        "cache_type": "multimodal_query",
                        "original_query": query,
                        "multimodal_content_count": len(multimodal_content),
                        "mode": mode,
                    }

                    await self.lightrag.llm_response_cache.upsert(
                        {cache_key: cache_entry}
                    )
                    self.logger.info(
                        f"已将多模态查询结果存入缓存: {cache_key[:16]}..."
                    )
                except Exception as e:
                    self.logger.debug(f"保存多模态查询缓存时出错: {e}")

        # 确保缓存已写入磁盘，防止程序意外崩溃导致数据丢失
        if (
                hasattr(self, "lightrag")
                and self.lightrag
                and hasattr(self.lightrag, "llm_response_cache")
                and self.lightrag.llm_response_cache
        ):
            try:
                await self.lightrag.llm_response_cache.index_done_callback()
            except Exception as e:
                self.logger.debug(f"持久化多模态查询缓存时出错: {e}")

        self.logger.info("多模态查询已完成")
        return result

    async def aquery_vlm_enhanced(
            self,
            query: str,
            mode: str = "mix",
            system_prompt: str | None = None,
            extra_safe_dirs: List[str] = None,
            **kwargs,
    ) -> str:
        """
        VLM enhanced query - replaces image paths in retrieved context with base64 encoded images for VLM processing

        Args:
            query: User query
            mode: Underlying LightRAG query mode
            system_prompt: Optional system prompt to include
            extra_safe_dirs: Optional list of additional safe directories to allow images from
            **kwargs: Other query parameters

        Returns:
            str: VLM query result
        """
        # Ensure VLM is available
        if not hasattr(self, "vision_model_func") or not self.vision_model_func:
            raise ValueError(
                "VLM enhanced query requires vision_model_func. "
                "Please provide a vision model function when initializing RAGAnything."
            )

        # Ensure LightRAG is initialized
        await self._ensure_lightrag_initialized()

        self.logger.info(f"Executing VLM enhanced query: {query[:100]}...")

        # Clear previous image cache
        if hasattr(self, "_current_images_base64"):
            delattr(self, "_current_images_base64")

        # 1. Get original retrieval prompt (without generating final answer)
        query_param = QueryParam(mode=mode, only_need_prompt=True, **kwargs)
        raw_prompt = await self.lightrag.aquery(query, param=query_param)

        self.logger.debug("Retrieved raw prompt from LightRAG")

        # 2. Extract and process image paths
        enhanced_prompt, images_found = await self._process_image_paths_for_vlm(
            raw_prompt, extra_safe_dirs=extra_safe_dirs
        )

        if not images_found:
            self.logger.info("No valid images found, falling back to normal query")
            # Fallback to normal query
            query_param = QueryParam(mode=mode, **kwargs)
            return await self.lightrag.aquery(
                query, param=query_param, system_prompt=system_prompt
            )

        self.logger.info(f"Processed {images_found} images for VLM")

        # 3. Build VLM message format
        messages = self._build_vlm_messages_with_images(
            enhanced_prompt, query, system_prompt
        )

        # 4. Call VLM for question answering
        result = await self._call_vlm_with_multimodal_content(messages)

        self.logger.info("VLM enhanced query completed")
        return result

    async def _process_multimodal_query_content(
            self, base_query: str, multimodal_content: List[Dict[str, Any]]
    ) -> str:
        """
        Process multimodal query content to generate enhanced query text

        Args:
            base_query: Base query text
            multimodal_content: List of multimodal content

        Returns:
            str: Enhanced query text
        """
        self.logger.info("Starting multimodal query content processing...")

        enhanced_parts = [f"User query: {base_query}"]

        for i, content in enumerate(multimodal_content):
            content_type = content.get("type", "unknown")
            self.logger.info(
                f"Processing {i + 1}/{len(multimodal_content)} multimodal content: {content_type}"
            )

            try:
                # Get appropriate processor
                processor = get_processor_for_type(self.modal_processors, content_type)

                if processor:
                    # Generate content description
                    description = await self._generate_query_content_description(
                        processor, content, content_type
                    )
                    enhanced_parts.append(
                        f"\nRelated {content_type} content: {description}"
                    )
                else:
                    # If no appropriate processor, use basic description
                    basic_desc = str(content)[:200]
                    enhanced_parts.append(
                        f"\nRelated {content_type} content: {basic_desc}"
                    )

            except Exception as e:
                self.logger.error(f"Error processing multimodal content: {str(e)}")
                # Continue processing other content
                continue

        enhanced_query = "\n".join(enhanced_parts)
        enhanced_query += PROMPTS["QUERY_ENHANCEMENT_SUFFIX"]

        self.logger.info("Multimodal query content processing completed")
        return enhanced_query

    async def _generate_query_content_description(
            self, processor, content: Dict[str, Any], content_type: str
    ) -> str:
        """
        Generate content description for query

        Args:
            processor: Multimodal processor
            content: Content data
            content_type: Content type

        Returns:
            str: Content description
        """
        try:
            if content_type == "image":
                return await self._describe_image_for_query(processor, content)
            elif content_type == "table":
                return await self._describe_table_for_query(processor, content)
            elif content_type == "equation":
                return await self._describe_equation_for_query(processor, content)
            else:
                return await self._describe_generic_for_query(
                    processor, content, content_type
                )

        except Exception as e:
            self.logger.error(f"Error generating {content_type} description: {str(e)}")
            return f"{content_type} content: {str(content)[:100]}"

    async def _describe_image_for_query(
            self, processor, content: Dict[str, Any]
    ) -> str:
        """Generate image description for query"""
        image_path = content.get("img_path")
        captions = content.get("image_caption", content.get("img_caption", []))
        footnotes = content.get("image_footnote", content.get("img_footnote", []))

        if image_path and Path(image_path).exists():
            # If image exists, use vision model to generate description
            image_base64 = processor._encode_image_to_base64(image_path)
            if image_base64:
                prompt = PROMPTS["QUERY_IMAGE_DESCRIPTION"]
                description = await processor.modal_caption_func(
                    prompt,
                    image_data=image_base64,
                    system_prompt=PROMPTS["QUERY_IMAGE_ANALYST_SYSTEM"],
                )
                return description

        # If image doesn't exist or processing failed, use existing information
        parts = []
        if image_path:
            parts.append(f"Image path: {image_path}")
        if captions:
            parts.append(f"Image captions: {', '.join(captions)}")
        if footnotes:
            parts.append(f"Image footnotes: {', '.join(footnotes)}")

        return "; ".join(parts) if parts else "Image content information incomplete"

    async def _describe_table_for_query(
            self, processor, content: Dict[str, Any]
    ) -> str:
        """Generate table description for query"""
        table_data = content.get("table_data", "")
        table_caption = content.get("table_caption", "")

        prompt = PROMPTS["QUERY_TABLE_ANALYSIS"].format(
            table_data=table_data, table_caption=table_caption
        )

        description = await processor.modal_caption_func(
            prompt, system_prompt=PROMPTS["QUERY_TABLE_ANALYST_SYSTEM"]
        )

        return description

    async def _describe_equation_for_query(
            self, processor, content: Dict[str, Any]
    ) -> str:
        """Generate equation description for query"""
        latex = content.get("latex", "")
        equation_caption = content.get("equation_caption", "")

        prompt = PROMPTS["QUERY_EQUATION_ANALYSIS"].format(
            latex=latex, equation_caption=equation_caption
        )

        description = await processor.modal_caption_func(
            prompt, system_prompt=PROMPTS["QUERY_EQUATION_ANALYST_SYSTEM"]
        )

        return description

    async def _describe_generic_for_query(
            self, processor, content: Dict[str, Any], content_type: str
    ) -> str:
        """Generate generic content description for query"""
        content_str = str(content)

        prompt = PROMPTS["QUERY_GENERIC_ANALYSIS"].format(
            content_type=content_type, content_str=content_str
        )

        description = await processor.modal_caption_func(
            prompt,
            system_prompt=PROMPTS["QUERY_GENERIC_ANALYST_SYSTEM"].format(
                content_type=content_type
            ),
        )

        return description

    async def _process_image_paths_for_vlm(
            self, prompt: str, extra_safe_dirs: List[str] = None
    ) -> tuple[str, int]:
        """
        Process image paths in prompt, keeping original paths and adding VLM markers

        Args:
            prompt: Original prompt
            extra_safe_dirs: Optional list of additional safe directories

        Returns:
            tuple: (processed prompt, image count)
        """
        enhanced_prompt = prompt
        images_processed = 0

        # Initialize image cache
        self._current_images_base64 = []

        # Enhanced regex pattern for matching image paths
        # Matches only the path ending with image file extensions
        image_path_pattern = (
            r"Image Path:\s*([^\r\n]*?\.(?:jpg|jpeg|png|gif|bmp|webp|tiff|tif))"
        )

        # First, let's see what matches we find
        matches = re.findall(image_path_pattern, prompt)
        self.logger.info(f"Found {len(matches)} image path matches in prompt")

        def replace_image_path(match):
            nonlocal images_processed

            image_path = match.group(1).strip()
            self.logger.debug(f"Processing image path: '{image_path}'")

            # Validate path format (basic check)
            if not image_path or len(image_path) < 3:
                self.logger.warning(f"Invalid image path format: {image_path}")
                return match.group(0)  # Keep original

            # Use utility function to validate image file
            is_valid = validate_image_file(image_path)

            # Security check: only allow images from the workspace or output directories
            # to prevent indirect prompt injection from reading arbitrary system files.
            if is_valid:
                abs_image_path = Path(image_path).resolve()
                # Check if it's in the current working directory or subdirectories
                try:
                    is_in_cwd = abs_image_path.is_relative_to(Path.cwd())
                except ValueError:
                    is_in_cwd = False

                # If a config is available, check against working_dir and parser_output_dir
                is_in_safe_dir = is_in_cwd
                if hasattr(self, "config") and self.config:
                    try:
                        is_in_working = abs_image_path.is_relative_to(
                            Path(self.config.working_dir).resolve()
                        )
                        is_in_output = abs_image_path.is_relative_to(
                            Path(self.config.parser_output_dir).resolve()
                        )
                        is_in_safe_dir = is_in_safe_dir or is_in_working or is_in_output
                    except Exception:
                        pass

                # Check against extra safe directories if provided
                if not is_in_safe_dir and extra_safe_dirs:
                    for safe_dir in extra_safe_dirs:
                        try:
                            if abs_image_path.is_relative_to(Path(safe_dir).resolve()):
                                is_in_safe_dir = True
                                break
                        except Exception:
                            continue

                if not is_in_safe_dir:
                    self.logger.warning(
                        f"Blocking image path outside safe directories: {image_path}"
                    )
                    is_valid = False

            if not is_valid:
                self.logger.warning(
                    f"Image validation failed or path unsafe for: {image_path}"
                )
                return match.group(0)  # Keep original if validation fails

            try:
                # Encode image to base64 using utility function
                self.logger.debug(f"Attempting to encode image: {image_path}")
                image_base64 = encode_image_to_base64(image_path)
                if image_base64:
                    images_processed += 1
                    # Save base64 to instance variable for later use
                    self._current_images_base64.append(image_base64)

                    # Keep original path info and add VLM marker
                    result = f"Image Path: {image_path}\n[VLM_IMAGE_{images_processed}]"
                    self.logger.debug(
                        f"Successfully processed image {images_processed}: {image_path}"
                    )
                    return result
                else:
                    self.logger.error(f"Failed to encode image: {image_path}")
                    return match.group(0)  # Keep original if encoding failed

            except Exception as e:
                self.logger.error(f"Failed to process image {image_path}: {e}")
                return match.group(0)  # Keep original

        # Execute replacement
        enhanced_prompt = re.sub(
            image_path_pattern, replace_image_path, enhanced_prompt
        )

        return enhanced_prompt, images_processed

    def _build_vlm_messages_with_images(
            self, enhanced_prompt: str, user_query: str, system_prompt: str
    ) -> List[Dict]:
        """
        Build VLM message format, using markers to correspond images with text positions

        Args:
            enhanced_prompt: Enhanced prompt with image markers
            user_query: User query

        Returns:
            List[Dict]: VLM message format
        """
        images_base64 = getattr(self, "_current_images_base64", [])

        if not images_base64:
            # Pure text mode
            return [
                {
                    "role": "user",
                    "content": f"Context:\n{enhanced_prompt}\n\nUser Question: {user_query}",
                }
            ]

        # Build multimodal content
        content_parts = []

        # Split text at image markers and insert images
        text_parts = enhanced_prompt.split("[VLM_IMAGE_")

        for i, text_part in enumerate(text_parts):
            if i == 0:
                # First text part
                if text_part.strip():
                    content_parts.append({"type": "text", "text": text_part})
            else:
                # Find marker number and insert corresponding image
                marker_match = re.match(r"(\d+)\](.*)", text_part, re.DOTALL)
                if marker_match:
                    image_num = (
                            int(marker_match.group(1)) - 1
                    )  # Convert to 0-based index
                    remaining_text = marker_match.group(2)

                    # Insert corresponding image
                    if 0 <= image_num < len(images_base64):
                        content_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{images_base64[image_num]}"
                                },
                            }
                        )

                    # Insert remaining text
                    if remaining_text.strip():
                        content_parts.append({"type": "text", "text": remaining_text})

        # Add user question
        content_parts.append(
            {
                "type": "text",
                "text": f"\n\nUser Question: {user_query}\n\nPlease answer based on the context and images provided.",
            }
        )
        base_system_prompt = "You are a helpful assistant that can analyze both text and image content to provide comprehensive answers."

        if system_prompt:
            full_system_prompt = base_system_prompt + " " + system_prompt
        else:
            full_system_prompt = base_system_prompt

        return [
            {
                "role": "system",
                "content": full_system_prompt,
            },
            {
                "role": "user",
                "content": content_parts,
            },
        ]

    async def _call_vlm_with_multimodal_content(self, messages: List[Dict]) -> str:
        """
        Call VLM to process multimodal content

        Args:
            messages: VLM message format

        Returns:
            str: VLM response result
        """
        try:
            user_message = messages[1]
            content = user_message["content"]
            system_prompt = messages[0]["content"]

            if isinstance(content, str):
                # Pure text mode
                result = await self.vision_model_func(
                    content, system_prompt=system_prompt
                )
            else:
                # Multimodal mode - pass complete messages directly to VLM
                result = await self.vision_model_func(
                    "",  # Empty prompt since we're using messages format
                    messages=messages,
                )

            return result

        except Exception as e:
            self.logger.error(f"VLM call failed: {e}")
            raise

    # Synchronous versions of query methods
    def query(self, query: str, mode: str = "mix", **kwargs) -> str:
        """
        Synchronous version of pure text query

        Args:
            query: Query text
            mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
            **kwargs: Other query parameters, will be passed to QueryParam
                - vlm_enhanced: bool, default True when vision_model_func is available.
                  If True, will parse image paths in retrieved context and replace them
                  with base64 encoded images for VLM processing.

        Returns:
            str: Query result
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, mode=mode, **kwargs))

    def query_with_multimodal(
            self,
            query: str,
            multimodal_content: List[Dict[str, Any]] = None,
            mode: str = "mix",
            **kwargs,
    ) -> str:
        """
        Synchronous version of multimodal query

        Args:
            query: Base query text
            multimodal_content: List of multimodal content, each element contains:
                - type: Content type ("image", "table", "equation", etc.)
                - Other fields depend on type (e.g., img_path, table_data, latex, etc.)
            mode: Query mode ("local", "global", "hybrid", "naive", "mix", "bypass")
            **kwargs: Other query parameters, will be passed to QueryParam

        Returns:
            str: Query result
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aquery_with_multimodal(query, multimodal_content, mode=mode, **kwargs)
        )
