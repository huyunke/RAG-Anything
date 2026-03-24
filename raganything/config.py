"""
RAGAnything 配置类
包含支持环境变量的配置数据类（Dataclasses）。
"""

from dataclasses import dataclass, field
from typing import List
from lightrag.utils import get_env_value


@dataclass
class RAGAnythingConfig:
    """RAGAnything 配置类，支持通过环境变量进行参数设置"""

    # 目录配置
    # ---
    working_dir: str = field(default=get_env_value("WORKING_DIR", "./rag_storage", str))
    """RAG 存储和缓存文件的存放目录。"""

    # 解析器配置
    # ---
    parse_method: str = field(default=get_env_value("PARSE_METHOD", "auto", str))
    """文档解析的默认方法：'auto'（自动）、'ocr'（光学字符识别）或 'txt'（纯文本）。"""

    parser_output_dir: str = field(default=get_env_value("OUTPUT_DIR", "./output", str))
    """解析后内容的默认输出目录。"""

    parser: str = field(default=get_env_value("PARSER", "mineru", str))
    """解析器选择：'mineru'、'docling' 或 'paddleocr'。"""

    display_content_stats: bool = field(
        default=get_env_value("DISPLAY_CONTENT_STATS", True, bool)
    )
    """是否在解析过程中显示内容统计信息。"""

    # 多模态处理配置
    # ---
    enable_image_processing: bool = field(
        default=get_env_value("ENABLE_IMAGE_PROCESSING", True, bool)
    )
    """是否启用图像内容处理。"""

    enable_table_processing: bool = field(
        default=get_env_value("ENABLE_TABLE_PROCESSING", True, bool)
    )
    """是否启用表格内容处理。"""

    enable_equation_processing: bool = field(
        default=get_env_value("ENABLE_EQUATION_PROCESSING", True, bool)
    )
    """是否启用数学公式处理。"""

    # 批量处理配置
    # ---
    max_concurrent_files: int = field(
        default=get_env_value("MAX_CONCURRENT_FILES", 1, int)
    )
    """最大同时并行处理的文件数量。"""

    supported_file_extensions: List[str] = field(
        default_factory=lambda: get_env_value(
            "SUPPORTED_FILE_EXTENSIONS",
            ".pdf,.jpg,.jpeg,.png,.bmp,.tiff,.tif,.gif,.webp,.doc,.docx,.ppt,.pptx,.xls,.xlsx,.txt,.md",
            str,
        ).split(",")
    )
    """批量处理模式下支持的文件扩展名列表。"""

    recursive_folder_processing: bool = field(
        default=get_env_value("RECURSIVE_FOLDER_PROCESSING", True, bool)
    )
    """在批量模式下是否递归处理子文件夹。"""

    # 上下文提取配置
    # ---
    context_window: int = field(default=get_env_value("CONTEXT_WINDOW", 1, int))
    """在当前条目前后包含的页面或块的数量（用于提供上下文）。"""

    context_mode: str = field(default=get_env_value("CONTEXT_MODE", "page", str))
    """上下文提取模式：'page' 表示基于页面，'chunk' 表示基于数据块。"""

    max_context_tokens: int = field(
        default=get_env_value("MAX_CONTEXT_TOKENS", 2000, int)
    )
    """提取的上下文内容的最大 Token 数量。"""

    include_headers: bool = field(default=get_env_value("INCLUDE_HEADERS", True, bool))
    """在上下文中是否包含文档页眉和标题。"""

    include_captions: bool = field(
        default=get_env_value("INCLUDE_CAPTIONS", True, bool)
    )
    """在上下文中是否包含图片和表格的说明文字（Caption）。"""

    context_filter_content_types: List[str] = field(
        default_factory=lambda: get_env_value(
            "CONTEXT_FILTER_CONTENT_TYPES", "text", str
        ).split(",")
    )
    """上下文提取中包含的内容类型（例如：'text'、'image'、'table'）。"""

    content_format: str = field(default=get_env_value("CONTENT_FORMAT", "minerU", str))
    """处理文档时，上下文提取的默认内容格式。"""

    # 路径处理配置
    # ---
    use_full_path: bool = field(default=get_env_value("USE_FULL_PATH", False, bool))
    """在 LightRAG 文件引用中，是否使用全路径（True）或仅使用文件名（False）。"""

    def __post_init__(self):
        """初始化后的设置，用于保持向下兼容性"""
        # 支持旧的环境变量名称
        legacy_parse_method = get_env_value("MINERU_PARSE_METHOD", None, str)
        if legacy_parse_method and not get_env_value("PARSE_METHOD", None, str):
            self.parse_method = legacy_parse_method
            import warnings

            warnings.warn(
                "MINERU_PARSE_METHOD 已弃用。请改用 PARSE_METHOD。",
                DeprecationWarning,
                stacklevel=2,
            )

    @property
    def mineru_parse_method(self) -> str:
        """
        针对旧代码的向下兼容属性。

        .. 已弃用::
           请改用 `parse_method`。此属性将在未来版本中移除。
        """
        import warnings

        warnings.warn(
            "mineru_parse_method 已弃用。请改用 parse_method。",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.parse_method

    @mineru_parse_method.setter
    def mineru_parse_method(self, value: str):
        """向下兼容的 Setter 方法"""
        import warnings

        warnings.warn(
            "mineru_parse_method 已弃用。请改用 parse_method。",
            DeprecationWarning,
            stacklevel=2,
        )
        self.parse_method = value
