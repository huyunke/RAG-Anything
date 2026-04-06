#!/usr/bin/env python
"""
Example script demonstrating direct content list insertion with RAGAnything

This example shows how to:
1. Create a simple content list with different content types
2. Insert content list directly without document parsing using insert_content_list() method
3. Perform pure text queries using aquery() method
4. Perform multimodal queries with specific multimodal content using aquery_with_multimodal() method
5. Handle different types of multimodal content in the inserted knowledge base
"""

"""
示例脚本演示了如何使用 RAGAnything 直接插入内容列表并进行查询

这个示例展示了以下内容：
1. 创建一个包含不同内容类型的简单内容列表
2. 使用 insert_content_list() 方法直接插入内容列表，无需文档解析
3. 使用 aquery() 方法执行纯文本查询
4. 使用 aquery_with_multimodal() 方法执行包含特定多模态内容
5. 处理插入的知识库中不同类型的多模态内容
"""

import os # 用于文件路径操作和环境变量管理
import argparse # 用于解析命令行参数
import asyncio # 用于异步处理和并发执行
import logging # 用于日志记录
import logging.config # 用于配置日志记录
from pathlib import Path # 用于处理文件路径

# Add project root directory to Python path
# 将项目根目录添加到 Python 路径（PYTHONPATH）中
import sys

sys.path.append(str(Path(__file__).parent.parent)) # 添加项目根目录到 Python 路径

from lightrag.llm.openai import openai_complete_if_cache, openai_embed # 从 lightrag 包中导入 OpenAI 相关函数
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug # 从 lightrag 包中导入必要的函数和类
from raganything import RAGAnything, RAGAnythingConfig

from dotenv import load_dotenv # 用于从 .env 文件加载环境变量

load_dotenv(dotenv_path=".env", override=False) # 加载环境变量，优先使用系统环境变量，.env 文件中的变量仅在系统环境变量未设置时生效

# 应用的日志配置函数，设置日志记录的格式、处理器和级别
def configure_logging():
    """Configure logging for the application"""
    # Get log directory path from environment variable or use current directory
    # 从环境变量获取日志目录路径，如果未设置则使用当前目录
    log_dir = os.getenv("LOG_DIR", os.getcwd()) # 获取日志目录路径，默认为当前工作目录
    log_file_path = os.path.abspath( # 获取日志文件的绝对路径
        os.path.join(log_dir, "insert_content_list_example.log")
    )

    print(f"\nInsert Content List example log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    # 从环境变量获取日志文件的最大大小和备份数量
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    # 用一个字典配置日志记录，定义格式化器、处理器和日志器
    logging.config.dictConfig(
        {
            "version": 1, # 日志配置版本
            "disable_existing_loggers": False, # 是否禁用现有日志器
            "formatters": { # 定义日志格式化器
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": { # 定义日志处理器
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": { # 定义文件处理器，使用 RotatingFileHandler 来管理日志文件大小和备份
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": { # 定义日志器，指定使用的处理器和日志级别
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    # 将日志器级别设置为 INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    # 根据环境变量设置是否启用详细调试日志
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")

def create_sample_content_list():
    """
    Create a simple content list for testing insert_content_list functionality

    Returns:
        List[Dict]: Sample content list with various content types

    Note:
        - img_path should be absolute path to the image file
        - page_idx represents the page number where the content appears (0-based)
    """

    """
    创建一个简单的内容列表，用于测试 insert_content_list 功能

    返回:
        List[Dict]: 包含各种内容类型的示例内容列表

    注意:
        - img_path 应当是图像文件的绝对路径
        - page_idx 表示内容出现的页码（从0开始）
    """
    content_list = [ # 具体内容，存放在一个列表中，每个元素是一个字典，表示不同类型的内容
        # Introduction text
        { # 定义一个文本内容，包含文本内容和所在页码
            "type": "text",
            "text": "Welcome to the RAGAnything System Documentation. This guide covers the advanced multimodal document processing capabilities and features of our comprehensive RAG system.",
            "page_idx": 0,  # Page number where this content appears
        },
        # System architecture image
        { # 定义一个图像内容，包含图像路径、标题、脚注和所在页码
            "type": "image",
            "img_path": "/absolute/path/to/system_architecture.jpg",  # IMPORTANT: Use absolute path to image file
            "image_caption": ["Figure 1: RAGAnything System Architecture"],
            "image_footnote": [
                "The architecture shows the complete pipeline from document parsing to multimodal query processing"
            ],
            "page_idx": 1,  # Page number where this image appears
        },
        # Performance comparison table
        # 定义一个表格内容，包含表格数据、标题、脚注和所在页码
        {
            "type": "table",
            "table_body": """| System | Accuracy | Processing Speed | Memory Usage |
                            |--------|----------|------------------|--------------|
                            | RAGAnything | 95.2% | 120ms | 2.1GB |
                            | Traditional RAG | 87.3% | 180ms | 3.2GB |
                            | Baseline System | 82.1% | 220ms | 4.1GB |
                            | Simple Retrieval | 76.5% | 95ms | 1.8GB |""",
            "table_caption": [
                "Table 1: Performance Comparison of Different RAG Systems"
            ],
            "table_footnote": [
                "All tests conducted on the same hardware with identical test datasets"
            ],
            "page_idx": 2,  # Page number where this table appears
        },
        # Mathematical formula
        # 定义一个方程内容，包含 LaTeX 公式、文本描述和所在页码
        {
            "type": "equation",
            "latex": "Relevance(d, q) = \\sum_{i=1}^{n} w_i \\cdot sim(t_i^d, t_i^q) \\cdot \\alpha_i",
            "text": "Document relevance scoring formula where w_i are term weights, sim() is similarity function, and α_i are modality importance factors",
            "page_idx": 3,  # Page number where this equation appears
        },
        # Feature description
        # 定义另一个文本内容，描述系统支持的内容模态和处理方式，包含文本内容和所在页码
        {
            "type": "text",
            "text": "The system supports multiple content modalities including text, images, tables, and mathematical equations. Each modality is processed using specialized processors optimized for that content type.",
            "page_idx": 4,  # Page number where this content appears
        },
        # Technical specifications table
        # 定义另一个表格内容，包含系统的技术规格，包含表格数据、标题、脚注和所在页码
        {
            "type": "table",
            "table_body": """| Feature | Specification |
                            |---------|---------------|
                            | Supported Formats | PDF, DOCX, PPTX, XLSX, Images |
                            | Max Document Size | 100MB |
                            | Concurrent Processing | Up to 8 documents |
                            | Query Response Time | <200ms average |
                            | Knowledge Graph Nodes | Up to 1M entities |""",
            "table_caption": ["Table 2: Technical Specifications"],
            "table_footnote": [
                "Specifications may vary based on hardware configuration"
            ],
            "page_idx": 5,  # Page number where this table appears
        },
        # Conclusion
        # 定义最后一个文本内容，作为结论，包含文本内容和所在页码
        {
            "type": "text",
            "text": "RAGAnything represents a significant advancement in multimodal document processing, providing comprehensive solutions for complex knowledge extraction and retrieval tasks.",
            "page_idx": 6,  # Page number where this content appears
        },
    ]

    return content_list

# 主异步函数，演示内容列表的插入和查询
async def demo_insert_content_list(
    api_key: str, # OpenAI的API密钥
    base_url: str = None, # 可选的API基础URL
    working_dir: str = None, # RAG存储的工作目录
):
    """
    Demonstrate content list insertion and querying with RAGAnything

    Args:
        api_key: OpenAI API key
        base_url: Optional base URL for API
        working_dir: Working directory for RAG storage
    """

    """
    展示使用 RAGAnything 进行内容列表插入和查询的示例

    参数:
        api_key: OpenAI API 密钥
        base_url: 可选的 API 基础 URL
        working_dir: RAG 存储的工作目录
    """
    try:
        # Create RAGAnything configuration
        # 创建 RAGAnything 配置，启用多模态内容处理和显示内容统计
        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_storage", # RAG 存储的工作目录，默认为 "./rag_storage"
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
            display_content_stats=True,  # Show content statistics
        )

        # Define LLM model function
        # 定义大预言模型函数，使用 openai_complete_if_cache 来调用 OpenAI API，并支持系统提示和历史消息
        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                "gpt-4o-mini",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        # Define vision model function for image processing
        # 定义视觉模型函数，用于处理图像输入，如果提供了图像数据，则调用 openai_complete_if_cache 进行多模态处理，否则调用 llm_model_func 进行纯文本处理
        def vision_model_func(
            prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs
        ):
            if image_data:
                return openai_complete_if_cache(
                    "gpt-4o",
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt}
                        if system_prompt
                        else None,
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    },
                                },
                            ],
                        }
                        if image_data
                        else {"role": "user", "content": prompt},
                    ],
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            else:
                return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        # Define embedding function - using environment variables for configuration
        # 定义嵌入函数，使用环境变量配置嵌入维度和模型名称，调用 openai_embed 进行文本嵌入
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "3072")) # 从环境变量获取嵌入维度，默认为 3072
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large") # 从环境变量获取嵌入模型名称，默认为 "text-embedding-3-large"

        # 定义一个 EmbeddingFunc 对象，指定嵌入维度、最大 token 大小和实际的嵌入函数
        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=lambda texts: openai_embed.func(
                texts,
                model=embedding_model,
                api_key=api_key,
                base_url=base_url,
            ),
        )

        # Initialize RAGAnything
        # 初始化 RAGAnything 实例，传入配置、LLM 模型函数、视觉模型函数和嵌入函数
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        # Create sample content list
        # 创建示例内容列表，调用 create_sample_content_list() 函数生成一个包含不同内容类型的列表
        logger.info("Creating sample content list...")
        content_list = create_sample_content_list()
        logger.info(f"Created content list with {len(content_list)} items")

        # Insert content list directly
        # 直接插入内容列表，调用 rag.insert_content_list() 方法，将内容列表插入到 RAGAnything 中，并指定参考文件名、文档 ID 和显示内容统计
        logger.info("\nInserting content list into RAGAnything...")
        await rag.insert_content_list(
            content_list=content_list,
            file_path="raganything_documentation.pdf",  # Reference file name for citation
            split_by_character=None,  # Optional text splitting
            split_by_character_only=False,  # Optional text splitting mode
            doc_id="demo-doc-001",  # Custom document ID
            display_stats=True,  # Show content statistics
        )
        logger.info("Content list insertion completed!")

        # Example queries - demonstrating different query approaches
        # 示例查询 - 演示不同的查询方法
        logger.info("\nQuerying inserted content:")

        # 1. Pure text queries using aquery()
        # 1. 使用 aquery() 方法进行纯文本查询
        text_queries = [
            "What is RAGAnything and what are its main features?",
            "How does RAGAnything compare to traditional RAG systems?",
            "What are the technical specifications of the system?",
        ]

        for query in text_queries:
            logger.info(f"\n[Text Query]: {query}")
            result = await rag.aquery(query, mode="hybrid") # 使用 aquery() 方法进行查询，mode="hybrid" 表示同时考虑文本和多模态内容
            logger.info(f"Answer: {result}")

        # 2. Multimodal query with specific multimodal content using aquery_with_multimodal()
        # 2. 使用 aquery_with_multimodal() 方法进行多模态查询，提供特定的多模态内容（如表格数据）来辅助查询
        logger.info(
            "\n[Multimodal Query]: Analyzing new performance data against existing benchmarks"
        )
        multimodal_result = await rag.aquery_with_multimodal(
            "Compare this new performance data with the existing benchmark results in the documentation",
            multimodal_content=[
                {
                    "type": "table",
                    "table_data": """Method,Accuracy,Speed,Memory
                                New_Approach,97.1%,110ms,1.9GB
                                Enhanced_RAG,91.4%,140ms,2.5GB""",
                    "table_caption": "Latest experimental results",
                }
            ],
            mode="hybrid",
        )
        logger.info(f"Answer: {multimodal_result}")

        # 3. Another multimodal query with equation content
        # 3. 另一个包含方程内容的多模态查询，提供一个数学公式来分析其与文档中相关评分机制的关系
        logger.info("\n[Multimodal Query]: Mathematical formula analysis")
        equation_result = await rag.aquery_with_multimodal(
            "How does this similarity formula relate to the relevance scoring mentioned in the documentation?",
            multimodal_content=[
                {
                    "type": "equation",
                    "latex": "sim(a, b) = \\frac{a \\cdot b}{||a|| \\times ||b||} + \\beta \\cdot context\\_weight",
                    "equation_caption": "Enhanced cosine similarity with context weighting",
                }
            ],
            mode="hybrid",
        )
        logger.info(f"Answer: {equation_result}")

        # 4. Insert another content list with different document ID
        # 4. 插入另一个内容列表，使用不同的文档 ID，展示如何在同一 RAGAnything 实例中管理多个文档的内容
        logger.info("\nInserting additional content list...")
        additional_content = [
            {
                "type": "text",
                "text": "This is additional documentation about advanced features and configuration options.",
                "page_idx": 0,  # Page number where this content appears
            },
            {
                "type": "table",
                "table_body": """| Configuration | Default Value | Range |
                                    |---------------|---------------|-------|
                                    | Chunk Size | 512 tokens | 128-2048 |
                                    | Context Window | 4096 tokens | 1024-8192 |
                                    | Batch Size | 32 | 1-128 |""",
                "table_caption": ["Advanced Configuration Parameters"],
                "page_idx": 1,  # Page number where this table appears
            },
        ]

        # 调用 insert_content_list() 方法插入新的内容列表，指定不同的文档 ID 来区分不同的知识库内容
        await rag.insert_content_list(
            content_list=additional_content,
            file_path="advanced_configuration.pdf",
            doc_id="demo-doc-002",  # Different document ID
        )

        # Query combined knowledge base
        # 查询组合知识库，展示如何在同一 RAGAnything 实例中查询多个文档的内容，验证不同文档的内容是否正确插入并可用于查询
        logger.info("\n[Combined Query]: What configuration options are available?")
        combined_result = await rag.aquery(
            "What configuration options are available and what are their default values?",
            mode="hybrid",
        )
        logger.info(f"Answer: {combined_result}")

    except Exception as e:
        logger.error(f"Error in content list insertion demo: {str(e)}")
        import traceback # 导入 traceback 模块，用于获取详细的错误堆栈信息

        logger.error(traceback.format_exc())

# 主函数，解析命令行参数并运行示例
def main():
    """Main function to run the example"""
    parser = argparse.ArgumentParser(description="Insert Content List Example")
    parser.add_argument( # 添加--working_dir 参数，用于指定 RAG 存储的工作目录，默认为 "./rag_storage"
        "--working_dir", "-w", default="./rag_storage", help="Working directory path"
    )
    parser.add_argument( # 添加--api-key 参数，用于指定 OpenAI API 密钥，默认为环境变量 LLM_BINDING_API_KEY 的值
        "--api-key",
        default=os.getenv("LLM_BINDING_API_KEY"),
        help="OpenAI API key (defaults to LLM_BINDING_API_KEY env var)",
    )
    parser.add_argument( # 添加--base-url 参数，用于指定 API 的基础 URL，默认为环境变量 LLM_BINDING_HOST 的值
        "--base-url",
        default=os.getenv("LLM_BINDING_HOST"),
        help="Optional base URL for API",
    )

    args = parser.parse_args()

    # Check if API key is provided
    # 检查是否提供了 API 密钥，如果没有提供，则记录错误日志并提示用户设置环境变量或使用命令行选项来提供 API 密钥
    if not args.api_key:
        logger.error("Error: OpenAI API key is required")
        logger.error("Set api key environment variable or use --api-key option")
        return

    # Run the demo
    # 运行示例，调用 asyncio.run() 来执行主异步函数 demo_insert_content_list()，传入解析后的命令行参数
    asyncio.run(
        demo_insert_content_list(
            args.api_key,
            args.base_url,
            args.working_dir,
        )
    )


if __name__ == "__main__":
    # Configure logging first
    configure_logging()

    print("RAGAnything Insert Content List Example")
    print("=" * 45)
    print("Demonstrating direct content list insertion without document parsing")
    print("=" * 45)

    main()
