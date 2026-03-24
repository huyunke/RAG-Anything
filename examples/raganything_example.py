#!/usr/bin/env python
"""
RAG-Anything 解析器集成示例脚本

本示例展示了如何：
1. 使用可配置的解析器处理文档：演示如何通过 RAG-Anything 调用不同的解析引擎来处理和准备文档数据。
2. 执行纯文本查询 (aquery() 方法)：展示如何利用异步查询方法对文档内容进行基础的文本提问。
3. 执行特定多模态内容的查询 (aquery_with_multimodal() 方法)：演示如何针对文档中包含的非文本信息（如图片、视觉内容）进行多模态问答。
4. 在查询中处理不同类型的多模态内容（表格、公式）：专门展示系统如何识别并理解文档中复杂的表格数据和数学公式，并据此给出准确回复。
"""

import os
import argparse
import asyncio
import logging
import logging.config
from pathlib import Path

# 将项目根目录添加到 Python 搜索路径中
import sys

sys.path.append(str(Path(__file__).parent.parent))

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig

from dotenv import load_dotenv

# 加载 .env 环境变量文件
load_dotenv(dotenv_path=".env", override=False)

# 日志配置
def configure_logging():
    """为应用程序配置日志系统"""
    # 从环境变量获取日志目录路径，或者使用当前目录
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "raganything_example.log"))

    print(f"\nRAGAnything 示例日志文件路径: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # 从环境变量获取日志文件最大大小和备份数量
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # 默认 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # 默认保留 5 个备份

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # 设置日志级别为 INFO
    logger.setLevel(logging.INFO)
    # 如果需要，启用详细调试模式
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")

# RAG 核心流程
async def process_with_rag(
    file_path: str,
    output_dir: str,
    api_key: str,
    base_url: str = None,
    working_dir: str = None,
    parser: str = None,
):
    """
    使用 RAGAnything 处理文档

    参数:
        file_path: 文档路径
        output_dir: RAG 结果输出目录
        api_key: OpenAI API 密钥
        base_url: 可选的 API 基础地址
        working_dir: RAG 存储工作目录
        parser: 解析器选择
    """
    try:
        # 创建 RAGAnything 配置
        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_storage",
            parser=parser,  # 解析器选择：mineru, docling, 或 paddleocr
            parse_method="auto",  # 解析方法：auto (自动), ocr, 或 txt
            enable_image_processing=True,   # 启用图像处理
            enable_table_processing=True,   # 启用表格处理
            enable_equation_processing=True, # 启用公式处理
        )

        # 定义 LLM 文本模型函数
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

        # 定义用于图像处理的视觉模型函数
        def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            image_data=None,
            messages=None,
            **kwargs,
        ):
            # 如果提供了 messages 格式（用于多模态 VLM 增强查询），直接使用它
            if messages:
                return openai_complete_if_cache(
                    "gpt-4o",
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            # 传统的单图格式
            elif image_data:
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
            # 纯文本格式
            else:
                return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        # 定义 Embedding 嵌入函数 - 使用环境变量进行配置
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "3072"))
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

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

        # 使用新的数据类结构初始化 RAGAnything
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        # 处理文档（完整流水线：解析、分片、向量化存储）
        await rag.process_document_complete(
            file_path=file_path, output_dir=output_dir, parse_method="auto"
        )

        # 示例查询 - 展示不同的查询方式
        logger.info("\n正在查询已处理的文档:")

        # 1. 使用 aquery() 进行纯文本查询
        text_queries = [
            "这份文档的主要内容是什么？",
            "讨论的关键主题有哪些？",
        ]

        for query in text_queries:
            logger.info(f"\n[文本查询]: {query}")
            result = await rag.aquery(query, mode="hybrid")
            logger.info(f"回答: {result}")

        # 2. 使用 aquery_with_multimodal() 进行带有特定多模态内容的查询
        logger.info(
            "\n[多模态查询]: 结合文档上下文分析性能数据"
        )
        multimodal_result = await rag.aquery_with_multimodal(
            "请将此性能数据与文档中提到的任何类似结果进行比较",
            multimodal_content=[
                {
                    "type": "table",
                    "table_data": """方法,准确率,处理时间
                                RAGAnything,95.2%,120ms
                                Traditional_RAG,87.3%,180ms
                                Baseline,82.1%,200ms""",
                    "table_caption": "性能对比结果表",
                }
            ],
            mode="hybrid",
        )
        logger.info(f"回答: {multimodal_result}")

        # 3. 另一个带有公式内容的多模态查询
        logger.info("\n[多模态查询]: 数学公式分析")
        equation_result = await rag.aquery_with_multimodal(
            "解释这个公式，并将其与文档中的任何数学概念联系起来",
            multimodal_content=[
                {
                    "type": "equation",
                    "latex": "F1 = 2 \\cdot \\frac{precision \\cdot recall}{precision + recall}",
                    "equation_caption": "F1-score 计算公式",
                }
            ],
            mode="hybrid",
        )
        logger.info(f"回答: {equation_result}")

    except Exception as e:
        logger.error(f"RAG 处理出错: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


def main():
    """运行示例的主函数"""
    parser = argparse.ArgumentParser(description="MinerU RAG 示例")
    parser.add_argument("file_path", help="要处理的文档路径")
    parser.add_argument(
        "--working_dir", "-w", default="./rag_storage", help="工作目录路径"
    )
    parser.add_argument(
        "--output", "-o", default="./output", help="输出目录路径"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_BINDING_API_KEY"),
        help="OpenAI API 密钥 (默认为 LLM_BINDING_API_KEY 环境变量)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BINDING_HOST"),
        help="可选的 API 基础地址",
    )
    parser.add_argument(
        "--parser",
        default=os.getenv("PARSER", "mineru"),
        help=(
            "解析器选择。内置选项：mineru, docling, paddleocr。"
            "当把 RAGAnything 作为库使用时，也接受在同一个 Python 进程中"
            "通过 register_parser() 注册的自定义解析器。"
            "本示例脚本不会执行任何自动插件发现。"
        ),
    )

    args = parser.parse_args()

    # 检查是否提供了 API 密钥
    if not args.api_key:
        logger.error("错误: 需要 OpenAI API 密钥")
        logger.error("请设置 API 密钥环境变量或使用 --api-key 选项")
        return

    # 如果指定了输出目录，则创建它
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # 启动 RAG 处理流程
    asyncio.run(
        process_with_rag(
            args.file_path,
            args.output,
            args.api_key,
            args.base_url,
            args.working_dir,
            args.parser,
        )
    )


if __name__ == "__main__":
    # 首先配置日志
    configure_logging()

    print("RAGAnything 示例")
    print("=" * 30)
    print("正在使用多模态 RAG 流水线处理文档")
    print("=" * 30)

    main()
