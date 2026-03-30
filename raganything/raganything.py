"""
完整的文档解析 + 多模态内容插入流水线（Pipeline）

本脚本集成了以下功能：

1.文档解析（使用可配置的解析器）

2.纯文本内容入库（插入到 LightRAG）

3.多模态内容专项处理（使用不同的专用处理器）
"""

import os
from typing import Dict, Any, Optional, Callable
import sys
import asyncio
import atexit
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# 将项目根目录添加到 Python 路径（PYTHONPATH）中
sys.path.insert(0, str(Path(__file__).parent.parent))

# 在导入 LightRAG 之前从 .env 文件中加载环境变量
# 这在离线环境中对于 TIKTOKEN_CACHE_DIR 的正确工作至关重要
# 操作系统环境变量优先于 .env 文件
load_dotenv(dotenv_path=".env", override=False)

from lightrag import LightRAG
from lightrag.utils import logger

# 导入配置和模块
from raganything.config import RAGAnythingConfig
from raganything.query import QueryMixin
from raganything.processor import ProcessorMixin
from raganything.batch import BatchMixin
from raganything.utils import get_processor_supports
from raganything.parser import MineruParser, SUPPORTED_PARSERS, get_parser
from raganything.callbacks import CallbackManager

# 导入专用的多模态处理器
from raganything.modalprocessors import (
    ImageModalProcessor,
    TableModalProcessor,
    EquationModalProcessor,
    GenericModalProcessor,
    ContextExtractor,
    ContextConfig,
)


@dataclass
class RAGAnything(QueryMixin, ProcessorMixin, BatchMixin):
    """多模态文档处理流水线 - 完整的文档解析与入库流水线"""

    # 核心组件
    # ---
    lightrag: Optional[LightRAG] = field(default=None)
    """可选参数：预先初始化的 LightRAG 实例"""

    llm_model_func: Optional[Callable] = field(default=None)
    """用于文本分析的大语言模型（LLM）函数"""

    vision_model_func: Optional[Callable] = field(default=None)
    """用于图像分析的视觉模型（VLM）函数."""

    embedding_func: Optional[Callable] = field(default=None)
    """用于文本向量化的嵌入（Embedding）函数."""

    config: Optional[RAGAnythingConfig] = field(default=None)
    """配置对象，如果为 None，将通过环境变量自动创建."""

    # LightRAG 详细配置
    # ---
    lightrag_kwargs: Dict[str, Any] = field(default_factory=dict)
    """
    当未直接提供 lightrag 实例时，用于初始化 LightRAG 的附加关键字参数
    允许传递所有 LightRAG 配置参数，例如：
    - 各种存储引擎：键值对存储 (kv)、向量存储 (vector)、图谱存储 (graph)、文档状态存储 (doc_status)
    - 检索控制：top_k 检索数量、分片 top_k、实体/关系/总 Token 最大限制
    - 相似度阈值、相关分片数量
    - 分片设置：分片 Token 大小、重叠 Token 大小、分词器、tiktoken 模型名
    - 嵌入优化：批量处理数量、最大异步并发数、Embedding 缓存配置
    - 模型参数：模型名称、Token 上限、最大异步并发、模型私有参数 (kwargs)
    - 其他：重排序函数 (rerank)、向量库类参数、是否启用 LLM 缓存
    - 并发控制：最大并行插入数、最大图节点数、附加参数等。
    """

    # 内部状态
    # ---
    modal_processors: Dict[str, Any] = field(default_factory=dict, init=False)
    """多模态处理器字典（存储图片、表格等专家处理器的实例）."""

    context_extractor: Optional[ContextExtractor] = field(default=None, init=False)
    """上下文提取器，用于为多模态处理器提供（图片/表格）周围的文本背景."""

    parse_cache: Optional[Any] = field(default=None, init=False)
    """解析结果缓存存储（利用 LightRAG 的 KV 存储能力）."""

    callback_manager: CallbackManager = field(
        default_factory=CallbackManager, init=False, repr=False
    )
    """处理回调管理器（用于监控系统状态、收集运行指标的可选钩子）."""

    _parser_installation_checked: bool = field(default=False, init=False)
    """标记位：记录是否已检查过解析器（如 MinerU）的安装情况."""

    def __post_init__(self):
        """Post-initialization setup following LightRAG pattern"""
        # Initialize configuration if not provided
        # 如果没有提供配置对象，则从环境变量创建一个默认配置对象
        if self.config is None:
            self.config = RAGAnythingConfig()

        # Set working directory
        # 将配置中的工作目录设置为实例属性，供后续使用（如 LightRAG 初始化、文件处理等）
        self.working_dir = self.config.working_dir

        # Set up logger (use existing logger, don't configure it)
        # 使用现有的 logger 实例，不在这里配置它（例如：不设置日志级别或处理器），以便用户可以在外部完全控制日志输出
        self.logger = logger

        # Set up document parser
        # 根据config中的parser字段获取对应的解析器实例
        self.doc_parser = get_parser(self.config.parser)

        # Register close method for cleanup
        # 注册 close 方法，在对象销毁时自动调用以清理资源
        atexit.register(self.close)

        # Create working directory if needed
        # 如果工作目录不存在，则创建它，并记录日志
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
            self.logger.info(f"Created working directory: {self.working_dir}")

        # Log configuration info
        self.logger.info("RAGAnything initialized with config:")
        self.logger.info(f"  Working directory: {self.config.working_dir}")
        self.logger.info(f"  Parser: {self.config.parser}")
        self.logger.info(f"  Parse method: {self.config.parse_method}")
        self.logger.info(
            f"  Multimodal processing - Image: {self.config.enable_image_processing}, "
            f"Table: {self.config.enable_table_processing}, "
            f"Equation: {self.config.enable_equation_processing}"
        )
        self.logger.info(f"  Max concurrent files: {self.config.max_concurrent_files}")

    def close(self):
        """
        对象销毁时清理资源
        处理以下三种常见场景：
        1.在运行中的异步上下文内（例如 FastAPI 关闭时） -> 调度异步任务
        2.线程中没有事件循环（典型的 atexit 进程退出时） -> 使用 asyncio.run() 创建一个
        3.事件循环存在但已关闭/正在关闭（atexit 竞态条件） -> 创建一个新的事件循环
        """
        try:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                # Case 1: We're inside a running event loop, schedule cleanup task
                loop.create_task(self.finalize_storages())
            else:
                # Case 2/3: No running loop. Clean up any stale loop reference
                # so asyncio.run() can create a fresh one (Python 3.10+ raises
                # RuntimeError if a loop is already set for the thread).
                if loop is not None:
                    try:
                        loop.close()
                    except Exception:
                        pass
                    asyncio.set_event_loop(None)
                asyncio.run(self.finalize_storages())
        except Exception:
            # Silently ignore during interpreter shutdown - the event loop and
            # resources are being torn down anyway, and printing may fail if
            # stdout/stderr are already closed. This avoids the noisy
            # "There is no current event loop in thread 'MainThread'" warning
            # that confused users (#135).
            pass

    def _create_context_config(self) -> ContextConfig:
        """从 RAGAnything 配置中创建上下文配置"""
        return ContextConfig(
            context_window=self.config.context_window,
            context_mode=self.config.context_mode,
            max_context_tokens=self.config.max_context_tokens,
            include_headers=self.config.include_headers,
            include_captions=self.config.include_captions,
            filter_content_types=self.config.context_filter_content_types,
        )

    def _create_context_extractor(self) -> ContextExtractor:
        """从 LightRAG 实例中获取分词器，并创建上下文提取器"""
        if self.lightrag is None:
            raise ValueError(
                "LightRAG must be initialized before creating context extractor"
            )

        context_config = self._create_context_config()
        return ContextExtractor(
            config=context_config, tokenizer=self.lightrag.tokenizer
        )

    def _initialize_processors(self):
        """使用适当的模型函数初始化多模态处理器"""
        # 依赖检查：必须先有 LightRAG 实例（底层数据库和基础能力），才能创建处理器
        if self.lightrag is None:
            raise ValueError(
                "在创建处理器之前，必须先初始化 LightRAG 实例"
            )

        # 创建上下文提取器：负责从文档中搜集背景信息
        self.context_extractor = self._create_context_extractor()

        # 根据配置创建不同的多模态处理器
        self.modal_processors = {}

        # 根据配置决定是否开启【图片处理器】
        if self.config.enable_image_processing:
            self.modal_processors["image"] = ImageModalProcessor(
                lightrag=self.lightrag,
                modal_caption_func=self.vision_model_func or self.llm_model_func,
                context_extractor=self.context_extractor,
            )

        # 根据配置决定是否开启【表格处理器】
        if self.config.enable_table_processing:
            self.modal_processors["table"] = TableModalProcessor(
                lightrag=self.lightrag,
                modal_caption_func=self.llm_model_func,
                context_extractor=self.context_extractor,
            )

        # 根据配置决定是否开启【公式处理器】
        if self.config.enable_equation_processing:
            self.modal_processors["equation"] = EquationModalProcessor(
                lightrag=self.lightrag,
                modal_caption_func=self.llm_model_func,
                context_extractor=self.context_extractor,
            )

        # 始终包含一个“通用处理器”，用于处理那些无法归类的特殊内容块
        self.modal_processors["generic"] = GenericModalProcessor(
            lightrag=self.lightrag,
            modal_caption_func=self.llm_model_func,
            context_extractor=self.context_extractor,
        )

        self.logger.info("多模态处理器初始化完成，并已支持上下文提取")
        self.logger.info(f"可用处理器列表: {list(self.modal_processors.keys())}")
        self.logger.info(f"上下文配置信息: {self._create_context_config()}")

    def update_config(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.debug(f"Updated config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown config parameter: {key}")

    async def _ensure_lightrag_initialized(self):
        """确保 LightRAG 实例已初始化，如果尚未初始化则创建一个"""
        try:
            # Check parser installation first
            if not self._parser_installation_checked:
                if not self.doc_parser.check_installation():
                    error_msg = (
                        f"Parser '{self.config.parser}' is not properly installed. "
                        "Please install it using 'pip install' or 'uv pip install'."
                    )
                    self.logger.error(error_msg)
                    return {"success": False, "error": error_msg}

                self._parser_installation_checked = True
                self.logger.info(f"Parser '{self.config.parser}' installation verified")

            if self.lightrag is not None:
                # LightRAG was pre-provided, but we need to ensure it's properly initialized
                # Inherit model functions from LightRAG if not explicitly provided
                if self.llm_model_func is None and hasattr(
                        self.lightrag, "llm_model_func"
                ):
                    self.llm_model_func = self.lightrag.llm_model_func
                    self.logger.debug("Inherited llm_model_func from LightRAG instance")

                if self.embedding_func is None and hasattr(
                        self.lightrag, "embedding_func"
                ):
                    self.embedding_func = self.lightrag.embedding_func
                    self.logger.debug("Inherited embedding_func from LightRAG instance")

                try:
                    # Ensure LightRAG storages are initialized
                    if (
                            not hasattr(self.lightrag, "_storages_status")
                            or self.lightrag._storages_status.name != "INITIALIZED"
                    ):
                        self.logger.info(
                            "Initializing storages for pre-provided LightRAG instance"
                        )
                        await self.lightrag.initialize_storages()
                        from lightrag.kg.shared_storage import (
                            initialize_pipeline_status,
                        )

                        await initialize_pipeline_status()

                    # Initialize parse cache if not already done
                    if self.parse_cache is None:
                        self.logger.info(
                            "Initializing parse cache for pre-provided LightRAG instance"
                        )
                        self.parse_cache = (
                            self.lightrag.key_string_value_json_storage_cls(
                                namespace="parse_cache",
                                workspace=self.lightrag.workspace,
                                global_config=self.lightrag.__dict__,
                                embedding_func=self.embedding_func,
                            )
                        )
                        await self.parse_cache.initialize()

                    # Initialize processors if not already done
                    if not self.modal_processors:
                        self._initialize_processors()

                    return {"success": True}

                except Exception as e:
                    error_msg = (
                        f"Failed to initialize pre-provided LightRAG instance: {str(e)}"
                    )
                    self.logger.error(error_msg, exc_info=True)
                    return {"success": False, "error": error_msg}

            # Validate required functions for creating new LightRAG instance
            if self.llm_model_func is None:
                error_msg = "llm_model_func must be provided when LightRAG is not pre-initialized"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}

            if self.embedding_func is None:
                error_msg = "embedding_func must be provided when LightRAG is not pre-initialized"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}

            from lightrag.kg.shared_storage import initialize_pipeline_status

            # Prepare LightRAG initialization parameters
            lightrag_params = {
                "working_dir": self.working_dir,
                "llm_model_func": self.llm_model_func,
                "embedding_func": self.embedding_func,
            }

            # Merge user-provided lightrag_kwargs, which can override defaults
            lightrag_params.update(self.lightrag_kwargs)

            # Log the parameters being used for initialization (excluding sensitive data)
            log_params = {
                k: v
                for k, v in lightrag_params.items()
                if not callable(v)
                   and k not in ["llm_model_kwargs", "vector_db_storage_cls_kwargs"]
            }
            self.logger.info(f"Initializing LightRAG with parameters: {log_params}")

            try:
                # Create LightRAG instance with merged parameters
                self.lightrag = LightRAG(**lightrag_params)
                await self.lightrag.initialize_storages()
                await initialize_pipeline_status()

                # Initialize parse cache storage using LightRAG's KV storage
                self.parse_cache = self.lightrag.key_string_value_json_storage_cls(
                    namespace="parse_cache",
                    workspace=self.lightrag.workspace,
                    global_config=self.lightrag.__dict__,
                    embedding_func=self.embedding_func,
                )
                await self.parse_cache.initialize()

                # Initialize processors after LightRAG is ready
                self._initialize_processors()

                self.logger.info(
                    "LightRAG, parse cache, and multimodal processors initialized"
                )
                return {"success": True}

            except Exception as e:
                error_msg = f"Failed to initialize LightRAG instance: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                return {"success": False, "error": error_msg}

        except Exception as e:
            error_msg = f"Unexpected error during LightRAG initialization: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {"success": False, "error": error_msg}

    async def finalize_storages(self):
        """
        完成所有存储的收尾工作，包括解析缓存和 LightRAG 存储。

        本方法应在系统关闭时调用，以便正确清理资源并持久化任何缓存数据。它将同时完成解析缓存和 LightRAG 内部存储的收尾.

        代码示例:
            try:
                rag_anything = RAGAnything(...)
                await rag_anything.process_file("document.pdf")
                # ... other operations ...
            finally:
                # Always finalize storages to clean up resources
                if rag_anything:
                    await rag_anything.finalize_storages()

        注意：
        1.当对象被销毁时，__del__ 方法会自动调用此方法
        2.在生产环境中，建议手动调用
        3。为了提升性能，所有收尾任务都会并发运行
        """
        try:
            tasks = []

            # Finalize parse cache if it exists
            if self.parse_cache is not None:
                tasks.append(self.parse_cache.finalize())
                self.logger.debug("Scheduled parse cache finalization")

            # Finalize LightRAG storages if LightRAG is initialized
            if self.lightrag is not None:
                tasks.append(self.lightrag.finalize_storages())
                self.logger.debug("Scheduled LightRAG storages finalization")

            # Run all finalization tasks concurrently
            if tasks:
                await asyncio.gather(*tasks)
                self.logger.info("Successfully finalized all RAGAnything storages")
            else:
                self.logger.debug("No storages to finalize")

        except Exception as e:
            self.logger.error(f"Error during storage finalization: {e}")
            raise

    def check_parser_installation(self) -> bool:
        """
        检查已配置的解析器是否已正确安装。
        返回：
        bool: 如果配置的解析器已正确安装，则返回 True
        """
        return self.doc_parser.check_installation()

    def verify_parser_installation_once(self) -> bool:
        if not self._parser_installation_checked:
            if not self.doc_parser.check_installation():
                raise RuntimeError(
                    f"Parser '{self.config.parser}' is not properly installed. "
                    "Please install it using pip install or uv pip install."
                )
            self._parser_installation_checked = True
            self.logger.info(f"Parser '{self.config.parser}' installation verified")
        return True

    def get_config_info(self) -> Dict[str, Any]:
        """获取当前配置信息"""
        config_info = {
            "directory": {
                "working_dir": self.config.working_dir,
                "parser_output_dir": self.config.parser_output_dir,
            },
            "parsing": {
                "parser": self.config.parser,
                "parse_method": self.config.parse_method,
                "display_content_stats": self.config.display_content_stats,
            },
            "multimodal_processing": {
                "enable_image_processing": self.config.enable_image_processing,
                "enable_table_processing": self.config.enable_table_processing,
                "enable_equation_processing": self.config.enable_equation_processing,
            },
            "context_extraction": {
                "context_window": self.config.context_window,
                "context_mode": self.config.context_mode,
                "max_context_tokens": self.config.max_context_tokens,
                "include_headers": self.config.include_headers,
                "include_captions": self.config.include_captions,
                "filter_content_types": self.config.context_filter_content_types,
            },
            "batch_processing": {
                "max_concurrent_files": self.config.max_concurrent_files,
                "supported_file_extensions": self.config.supported_file_extensions,
                "recursive_folder_processing": self.config.recursive_folder_processing,
            },
            "logging": {
                "note": "Logging fields have been removed - configure logging externally",
            },
        }

        # Add LightRAG configuration if available
        if self.lightrag_kwargs:
            # Filter out sensitive data and callable objects for display
            safe_kwargs = {
                k: v
                for k, v in self.lightrag_kwargs.items()
                if not callable(v)
                   and k not in ["llm_model_kwargs", "vector_db_storage_cls_kwargs"]
            }
            config_info["lightrag_config"] = {
                "custom_parameters": safe_kwargs,
                "note": "LightRAG will be initialized with these additional parameters",
            }
        else:
            config_info["lightrag_config"] = {
                "custom_parameters": {},
                "note": "Using default LightRAG parameters",
            }

        return config_info

    def set_content_source_for_context(
            self, content_source, content_format: str = "auto"
    ):
        """
        为所有多模态处理器设置用于提取上下文的内容源。

        参数说明：
            content_source: 用于提取上下文的源内容（例如：MinerU 解析出的内容列表 content_list）。
            content_format: 内容源的格式（可选："minerU", "text_chunks", "auto"）。
        """

        # 安全检查：如果没有初始化任何多模态处理器（如图片处理器、表格处理器），则无法设置
        if not self.modal_processors:
            self.logger.warning(
                "多模态处理器尚未初始化。内容源将在处理器创建时再行设置"
            )
            return

        # 遍历字典中所有的处理器
        for processor_name, processor in self.modal_processors.items():
            try:
                # 调用每个处理器内部的 set_content_source 方法，把整份文档的信息同步过去
                processor.set_content_source(content_source, content_format)
                self.logger.debug(f"已为 {processor_name} 处理器设置内容上下文源")
            except Exception as e:
                # 如果某个处理器（比如公式处理器）同步失败，记录错误但不中断整体流程
                self.logger.error(
                    f"为 {processor_name}设置内容源失败: {e}"
                )

        self.logger.info(
            f"上下文提取的内容源已设置完成 (格式: {content_format})"
        )

    def update_context_config(self, **context_kwargs):
        """
        更新上下文提取配置
        参数说明：
        **context_kwargs: 待更新的上下文配置参数
        （例如：context_window 上下文窗口大小、context_mode 上下文模式、max_context_tokens 最大上下文 Token 数等）
        """
        # Update the main config
        for key, value in context_kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.debug(f"Updated context config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown context config parameter: {key}")

        # Recreate context extractor with new config if processors are initialized
        if self.lightrag and self.modal_processors:
            try:
                self.context_extractor = self._create_context_extractor()
                # Update all processors with new context extractor
                for processor_name, processor in self.modal_processors.items():
                    processor.context_extractor = self.context_extractor

                self.logger.info(
                    "Context configuration updated and applied to all processors"
                )
                self.logger.info(
                    f"New context configuration: {self._create_context_config()}"
                )
            except Exception as e:
                self.logger.error(f"Failed to update context configuration: {e}")

    def get_processor_info(self) -> Dict[str, Any]:
        """获取处理器信息"""
        base_info = {
            "mineru_installed": MineruParser.check_installation(MineruParser()),
            "parser_installation": {
                parser_name: get_parser(parser_name).check_installation()
                for parser_name in SUPPORTED_PARSERS
            },
            "config": self.get_config_info(),
            "models": {
                "llm_model": "External function"
                if self.llm_model_func
                else "Not provided",
                "vision_model": "External function"
                if self.vision_model_func
                else "Not provided",
                "embedding_model": "External function"
                if self.embedding_func
                else "Not provided",
            },
        }

        if not self.modal_processors:
            base_info["status"] = "Not initialized"
            base_info["processors"] = {}
        else:
            base_info["status"] = "Initialized"
            base_info["processors"] = {}

            for proc_type, processor in self.modal_processors.items():
                base_info["processors"][proc_type] = {
                    "class": processor.__class__.__name__,
                    "supports": get_processor_supports(proc_type),
                    "enabled": True,
                }

        return base_info
