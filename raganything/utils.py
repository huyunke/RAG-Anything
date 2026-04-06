"""
RAGAnything 工具函数库

包含用于内容分离、文本插入及其他用途的辅助函数。
"""

import base64 # 用于图像编码
from typing import Dict, List, Any, Tuple # 用于类型注解
from pathlib import Path # 用于文件路径处理
from lightrag.utils import logger # 引入 LightRAG 的日志工具


def separate_content(
        content_list: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    分离文本内容和多模态内容
    参数说明：
        content_list: 来自 MinerU 解析后的内容列表
    返回：
        (text_content, multimodal_items):
            text_content: 拼接后的完整文本字符串
            multimodal_items: 包含图片、表格、公式等非纯文本项的列表
    """
    text_parts = []
    multimodal_items = []

    # 核心分拣循环：遍历每一个内容块
    # content_list是一个列表，列表中的每一个元素都是一个字典
    for item in content_list:
        # 获取当前块的类型，如果没有类型字段，默认视为 "text"
        content_type = item.get("type", "text")

        if content_type == "text":
            # 处理文本内容
            text = item.get("text", "")
            # 只有当文本不是纯空格或空字符串时才收集
            if text.strip():
                text_parts.append(text)
        else:
            # 处理多模态内容（如：image, table, equation 等）
            # 直接将整个字典对象存入多模态列表，保留其路径、坐标、属性等原始信息
            multimodal_items.append(item)

    # 将所有收集到的文本片段用两个换行符连接起来，形成一篇完整的 Markdown 文档
    text_content = "\n\n".join(text_parts)

    logger.info("内容分离完成：")
    logger.info(f"  - 文本总长度:{len(text_content)} 个字符")
    logger.info(f"  - 多模态项总数: {len(multimodal_items)}")

    # 统计多模态项的具体分布（例如：有多少张图，多少个表格）
    modal_types = {}
    for item in multimodal_items:
        modal_type = item.get("type", "unknown")
        modal_types[modal_type] = modal_types.get(modal_type, 0) + 1

    if modal_types:
        logger.info(f"  - 多模态类型分布情况: {modal_types}")

    return text_content, multimodal_items


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image file to base64 string

    Args:
        image_path: Path to the image file

    Returns:
        str: Base64 encoded string, empty string if encoding fails
    """

    """
    将图像文件编码为 Base64 字符串

    参数说明：
        image_path: 图像文件的路径

    返回：
        str: Base64 编码后的字符串，如果编码失败则返回空字符串
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return ""


def validate_image_file(image_path: str, max_size_mb: int = 50) -> bool:
    """
    Validate if a file is a valid image file

    Args:
        image_path: Path to the image file
        max_size_mb: Maximum file size in MB

    Returns:
        bool: True if valid, False otherwise
    """

    """
    验证问价是否为有效的图像文件

    参数：
        image_path: 图像文件的路径
        max_size_mb: 文件最大大小（单位：MB）

    返回：
        bool: 如果文件有效则返回 True，否则返回 False
    """
    try:
        path = Path(image_path)

        # 详细日志记录 - 包括输入路径、解析后的路径对象以及存在性检查结果
        logger.debug(f"Validating image path: {image_path}")
        logger.debug(f"Resolved path object: {path}")
        logger.debug(f"Path exists check: {path.exists()}")

        # Check if file exists and is not a symlink (for security)
        # 检查文件是否存在，并且不是符号链接（出于安全考虑）
        if not path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return False

        if path.is_symlink():
            logger.warning(f"Blocking symlink for security: {image_path}")
            return False

        # Check file extension
        # 定义允许的图像文件扩展名列表，支持常见的图像格式
        image_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp",
            ".tiff",
            ".tif",
        ]

        path_lower = str(path).lower() # 将路径转换为小写以进行不区分大小写的扩展名检查
        has_valid_extension = any(path_lower.endswith(ext) for ext in image_extensions) # 检查文件扩展名是否在允许的列表中
        logger.debug(
            f"File extension check - path: {path_lower}, valid: {has_valid_extension}"
        )

        # 如果文件扩展名不合法，记录警告日志并返回 False
        if not has_valid_extension:
            logger.warning(f"File does not appear to be an image: {image_path}")
            return False

        # Check file size
        # 检查文件大小，获取文件的实际大小并与最大允许大小进行比较
        file_size = path.stat().st_size
        max_size = max_size_mb * 1024 * 1024
        logger.debug(
            f"File size check - size: {file_size} bytes, max: {max_size} bytes"
        )

        if file_size > max_size:
            logger.warning(f"Image file too large ({file_size} bytes): {image_path}")
            return False

        logger.debug(f"Image validation successful: {image_path}")
        return True

    except Exception as e:
        logger.error(f"Error validating image file {image_path}: {e}")
        return False


async def insert_text_content(
        lightrag,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
):
    """
    将纯文本内容插入到 LightRAG 中

    参数说明：
        lightrag: LightRAG 实例（底层引擎）
        input: 单个文档字符串或文档字符串列表（即你要存入的知识）
        split_by_character: 如果不为 None，则按此字符（如换行符）切分字符串；
                           如果切片后的长度超过了配置的 chunk_token_size（分片 Token 上限），
                           它会根据 Token 大小再次进行细分
        split_by_character_only: 如果为 True，则仅按照 split_by_character 字符切分
                                不再进行二次 Token 切分。如果 split_by_character 为 None，此参数无效
        ids: 单个文档 ID 或唯一文档 ID 列表。如果不提供，系统将自动生成基于 MD5 哈希的内容 ID
        file_paths: 单个文件路径或路径列表，主要用于 AI 生成回答时的“引用溯源”
    """
    logger.info("正在开始向 LightRAG 插入文本内容...")

    # 调用 LightRAG 的异步插入方法 (ainsert)，并传入所有控制参数
    # 这是真正的“入库”动作，涉及文本分段、向量化（Embedding）和存储
    await lightrag.ainsert(
        input=input,
        file_paths=file_paths,
        split_by_character=split_by_character,
        split_by_character_only=split_by_character_only,
        ids=ids,
    )

    logger.info("文本内容插入完成")


async def insert_text_content_with_multimodal_content(
        lightrag,
        input: str | list[str],
        multimodal_content: list[dict[str, any]] | None = None,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
        scheme_name: str | None = None,
):
    """
    Insert pure text content into LightRAG

    Args:
        lightrag: LightRAG instance
        input: Single document string or list of document strings
        multimodal_content: Multimodal content list (optional)
        split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
        chunk_token_size, it will be split again by token size.
        split_by_character_only: if split_by_character_only is True, split the string by character only, when
        split_by_character is None, this parameter is ignored.
        ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
        file_paths: single string of the file path or list of file paths, used for citation
        scheme_name: scheme name (optional)
    """

    """
    将纯文本内容插入到 LightRAG 中，并支持多模态内容的关联

    参数说明：
    lightrag: LightRAG 实例（底层引擎）
    input: 单个文档字符串或文档字符串列表（即你要存入的知识）
    multimodal_content: 多模态内容列表（可选）
    split_by_character: 如果不为 None，则按此字符（如换行符）切分字符串；如果切片后的长度超过了配置的 chunk_token_size（分片 Token 上限），它会根据 Token 大小再次进行细分
    split_by_character_only: 如果为 True，则仅按照 split_by_character 字符切分，不再进行二次 Token 切分。如果 split_by_character 为 None，此参数无效
    ids: 单个文档 ID 或唯一文档 ID 列表。如果不提供，系统将自动生成基于 MD5 哈希的内容 ID
    file_paths: 单个文件路径或路径列表，主要用于 AI 生成回答时的“引用溯源”
    scheme_name: 方案名称（可选）
    """
    logger.info("Starting text content insertion into LightRAG...")

    # Use LightRAG's insert method with all parameters
    # 使用 LightRAG 的异步插入方法 (ainsert)，并传入所有控制参数。这是真正的“入库”动作，涉及文本分段、向量化（Embedding）和存储
    try:
        await lightrag.ainsert(
            input=input,
            multimodal_content=multimodal_content,
            file_paths=file_paths,
            split_by_character=split_by_character,
            split_by_character_only=split_by_character_only,
            ids=ids,
            scheme_name=scheme_name,
        )
    except Exception as e:
        logger.info(f"Error: {e}")
        logger.info(
            "If the error is caused by the ainsert function not having a multimodal content parameter, please update the raganything branch of lightrag"
        )

    logger.info("Text content insertion complete")


def get_processor_for_type(modal_processors: Dict[str, Any], content_type: str):
    """
    Get appropriate processor based on content type

    Args:
        modal_processors: Dictionary of available processors
        content_type: Content type

    Returns:
        Corresponding processor instance
    """

    """
    根据内容类型获取适当的处理器

    参数说明：
        modal_processors: 可用处理器的字典
        content_type: 内容类型

    返回：
        对应的处理器实例
    """
    # Direct mapping to corresponding processor
    # 根据内容类型直接映射到对应的处理器，例如：如果内容类型是 "image"，则返回 modal_processors 中的 "image" 处理器实例
    if content_type == "image":
        return modal_processors.get("image")
    elif content_type == "table":
        return modal_processors.get("table")
    elif content_type == "equation":
        return modal_processors.get("equation")
    else:
        # For other types, use generic processor
        # 对于其他类型，使用通用处理器（generic processor），它可以处理一些基本的文本分析和结构化处理任务
        return modal_processors.get("generic")

# 获取处理器支持的功能列表，基于处理器类型返回相应的功能描述列表
def get_processor_supports(proc_type: str) -> List[str]:
    """Get processor supported features"""
    supports_map = {
        "image": [
            "Image content analysis",
            "Visual understanding",
            "Image description generation",
            "Image entity extraction",
        ],
        "table": [
            "Table structure analysis",
            "Data statistics",
            "Trend identification",
            "Table entity extraction",
        ],
        "equation": [
            "Mathematical formula parsing",
            "Variable identification",
            "Formula meaning explanation",
            "Formula entity extraction",
        ],
        "generic": [
            "General content analysis",
            "Structured processing",
            "Entity extraction",
        ],
    }
    return supports_map.get(proc_type, ["Basic processing"])
