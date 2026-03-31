#!/usr/bin/env python3
"""
RAG-Anything 文本格式解析测试脚本

本脚本演示了如何使用 MinerU 解析多种文本格式，包括 TXT 和 MD 文件。

依赖项：

ReportLab 库（用于 PDF 转换）

RAG-Anything 软件包

使用方法:
    python text_format_test.py --file path/to/text/document.md
"""

import argparse
import asyncio
import sys
from pathlib import Path
from raganything import RAGAnything


def check_reportlab_installation():
    """检查 ReportLab 是否已安装并可用"""
    try:
        import reportlab

        print(
            f"✅ 找到 ReportLab: 版本 {reportlab.Version if hasattr(reportlab, 'Version') else 'Unknown'}"
        )
        return True
    except ImportError:
        print("❌ 未找到 ReportLab。请安装该库：")
        print("  pip install reportlab")
        return False


async def test_text_format_parsing(file_path: str):
    """使用 MinerU 测试文本格式解析能力"""

    print(f"🧪 正在测试文本格式解析: {file_path}")

    # 1. 检查文件是否存在以及是否为支持的文本格式
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return False

    supported_extensions = {".txt", ".md"}
    if file_path.suffix.lower() not in supported_extensions:
        print(f"❌ 不支持的文件格式: {file_path.suffix}")
        print(f"   当前支持的格式: {', '.join(supported_extensions)}")
        return False

    print(f"📄 文件格式: {file_path.suffix.upper()}")
    print(f"📏 文件大小: {file_path.stat().st_size / 1024:.1f} KB")

    # Display text file info
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"📝 文本长度: {len(content)} characters")
        print(f"📋 行数: {len(content.splitlines())}")
    except UnicodeDecodeError:
        print(
            "⚠️  文本编码: 非 UTF-8 (处理过程中将尝试多种编码识别)"
        )

    # 3. 初始化 RAGAnything (此处仅使用其解析模块)
    rag = RAGAnything()

    try:
        # 4. 调用 MinerU 执行核心解析
        # parse_method="auto" 表示让系统自动决定是用 OCR 还是直接提取
        print("\n🔄 正在调用 MinerU 解析文本...")
        content_list, md_content = await rag.parse_document(
            file_path=str(file_path),
            output_dir="./test_output",
            parse_method="auto",
            display_stats=True,
        )

        print("✅ 解析成功!")
        print(f"   📊 内容块数量: {len(content_list)}")
        print(f"   📝 生成的 Markdown 长度: {len(md_content)} 个字符")

        # 5.分析内容类型（统计解析出了哪些元素）
        content_types = {}
        for item in content_list:
            if isinstance(item, dict):
                content_type = item.get("type", "unknown")
                content_types[content_type] = content_types.get(content_type, 0) + 1

        if content_types:
            print("   📋 内容分布情况:")
            for content_type, count in sorted(content_types.items()):
                print(f"      • {content_type}: {count}")

        # 6. 展示提取出的文本预览
        if md_content.strip():
            print("\n📄 Extracted text preview (first 500 characters):")
            preview = md_content.strip()[:500]
            print(f"   {preview}{'...' if len(md_content) > 500 else ''}")
        else:
            print("\n📄 No text extracted from the document")

        # 7. 详细检查文本块 (Text Blocks)
        text_items = [
            item
            for item in content_list
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if text_items:
            print("\n📝 发现文本块:")
            for i, item in enumerate(text_items[:3], 1):
                text_content = item.get("text", "")
                if text_content.strip():
                    preview = text_content.strip()[:200]
                    print(
                        f"   {i}. {preview}{'...' if len(text_content) > 200 else ''}"
                    )

        # 8. 检查文本中是否识别到了表格
        table_items = [
            item
            for item in content_list
            if isinstance(item, dict) and item.get("type") == "table"
        ]
        if table_items:
            print(f"\n📊 文档中发现 {len(table_items)} 个表格:")
            for i, item in enumerate(table_items, 1):
                table_body = item.get("table_body", "")
                row_count = len(table_body.split("\n"))
                print(f"   {i}. 包含约 {row_count} 行的表格")

        # 9. 检查图片（TXT 不可能有，但 MD 可能包含外链图片引用）
        image_items = [
            item
            for item in content_list
            if isinstance(item, dict) and item.get("type") == "image"
        ]
        if image_items:
            print(f"\n🖼️  发现 {len(image_items)} 张图片:")
            for i, item in enumerate(image_items, 1):
                print(f"   {i}. 图片路径: {item.get('img_path', 'N/A')}")

        print("\n🎉 文本格式解析测试圆满完成!")
        print("📁 输出文件已保存至: ./test_output")
        return True

    except Exception as e:
        print(f"\n❌ 文本格式解析失败: {str(e)}")
        import traceback

        print(f"   完整错误堆栈: {traceback.format_exc()}")
        return False


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description="使用 MinerU 测试文本格式解析")
    parser.add_argument("--file", help="待测试的文本文件路径")
    parser.add_argument(
        "--check-reportlab",
        action="store_true",
        help="仅检查 ReportLab 安装情况",
    )

    args = parser.parse_args()

    # 首先执行环境检查
    print("🔧 正在检查 ReportLab 安装状态...")
    if not check_reportlab_installation():
        return 1

    if args.check_reportlab:
        print("✅ ReportLab 环境检查通过!")
        return 0

    # 如果不是只查依赖，则必须提供 --file 参数
    if not args.file:
        print("❌ 错误: 未使用 --check-reportlab 时，必须提供 --file 参数")
        parser.print_help()
        return 1

    # 执行异步解析测试
    try:
        success = asyncio.run(test_text_format_parsing(args.file))
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n⏹️ 用户手动中断测试")
        return 1
    except Exception as e:
        print(f"\n❌ 发生了意外错误: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
