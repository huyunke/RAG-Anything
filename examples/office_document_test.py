#!/usr/bin/env python3
"""
用于 RAG-Anything 的办公文档解析测试脚本

该脚本演示如何使用 MinerU 解析各种办公文档格式，
包括 DOC、DOCX、PPT、PPTX、XLS 和 XLSX 文件。

依赖要求：
- 系统已安装 LibreOffice
- RAG-Anything 包

使用方法：
    python office_document_test.py --file path/to/office/document.docx
"""

import argparse
import asyncio
import sys
from pathlib import Path
from raganything import RAGAnything


def check_libreoffice_installation():
    """检查 LibreOffice 是否已安装并可用"""
    import subprocess

    # 尝试不同的命令名称（libreoffice 或 soffice）来定位 LibreOffice
    for cmd in ["libreoffice", "soffice"]:
        try:
            result = subprocess.run(
                [cmd, "--version"], capture_output=True, check=True, timeout=10
            )
            print(f"✅ LibreOffice 已找到: {result.stdout.decode().strip()}")
            return True
        except (
                subprocess.CalledProcessError,
                FileNotFoundError,
                subprocess.TimeoutExpired,
        ):
            continue

    print("❌ 未找到 LibreOffice。请安装 LibreOffice：")
    print("  - Windows: 从 https://www.libreoffice.org/download/download/ 下载")
    print("  - macOS: brew install --cask libreoffice")
    print("  - Ubuntu/Debian: sudo apt-get install libreoffice")
    print("  - CentOS/RHEL: sudo yum install libreoffice")
    return False


async def test_office_document_parsing(file_path: str):
    """
    使用 MinerU 测试办公文档解析功能
    Args:
        file_path: 要测试的办公文档文件路径

    Returns:
        bool: 解析成功返回 True，失败返回 False
    """

    print(f"🧪 正在测试办公文档解析: {file_path}")

    # 检查文件是否存在且为支持的办公文档格式
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return False

    # 定义支持的办公文档扩展名集合
    supported_extensions = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
    if file_path.suffix.lower() not in supported_extensions:
        print(f"❌ 不支持的文件格式: {file_path.suffix}")
        print(f"   支持的格式: {', '.join(supported_extensions)}")
        return False

    # 打印文件基本信息
    print(f"📄 文件格式: {file_path.suffix.upper()}")
    print(f"📏 文件大小: {file_path.stat().st_size / 1024:.1f} KB")

    # 初始化 RAGAnything 实例（仅用于解析功能）
    rag = RAGAnything()

    try:
        # 使用 MinerU 测试文档解析
        print("\n🔄 正在使用 MinerU 测试文档解析...")

        content_list, md_content = await rag.parse_document(
            file_path=str(file_path),
            output_dir="./test_output",
            parse_method="auto",
            display_stats=True,
        )

        print("✅ 解析成功!")
        print(f"   📊 内容块数量: {len(content_list)}")
        print(f"   📝 Markdown 内容长度: {len(md_content)} 字符")

        # 统计不同类型的内容块数量
        content_types = {}
        for item in content_list:
            if isinstance(item, dict):
                content_type = item.get("type", "unknown")
                content_types[content_type] = content_types.get(content_type, 0) + 1

        if content_types:
            print("   📋 内容分布统计:")
            for content_type, count in sorted(content_types.items()):
                print(f"      • {content_type}: {count}")

        # 显示部分解析内容预览
        if md_content.strip():
            print("\n📄 解析内容预览（前 500 字符）:")
            preview = md_content.strip()[:500]
            print(f"   {preview}{'...' if len(md_content) > 500 else ''}")

        # 显示一些结构化的文本内容示例
        # 过滤出类型为 "text" 的内容块
        text_items = [
            item
            for item in content_list
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if text_items:
            print("\n📝 文本块示例:")
            for i, item in enumerate(text_items[:3], 1):
                text_content = item.get("text", "")
                if text_content.strip():
                    preview = text_content.strip()[:200]
                    print(
                        f"   {i}. {preview}{'...' if len(text_content) > 200 else ''}"
                    )

        # 检查文档中是否包含图片
        image_items = [
            item
            for item in content_list
            if isinstance(item, dict) and item.get("type") == "image"
        ]
        if image_items:
            print(f"\n🖼️  找到 {len(image_items)} 张图片:")
            for i, item in enumerate(image_items, 1):
                print(f"   {i}. 图片路径: {item.get('img_path', 'N/A')}")

        # 检查文档中是否包含表格
        table_items = [
            item
            for item in content_list
            if isinstance(item, dict) and item.get("type") == "table"
        ]
        if table_items:
            print(f"\n📊 找到{len(table_items)} 个表格:")
            for i, item in enumerate(table_items, 1):
                table_body = item.get("table_body", "")
                row_count = len(table_body.split("\n"))
                print(f"   {i}. 表格，约 {row_count} 行")

        print("\n🎉 办公文档解析测试成功完成！")
        print("📁 输出文件已保存到: ./test_output")
        return True

    except Exception as e:
        print(f"\n❌ 办公文档解析失败: {str(e)}")
        import traceback

        print(f"   完整错误信息: {traceback.format_exc()}")
        return False


def main():
    """主函数 - 处理命令行参数并执行测试"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="使用 MinerU 测试办公文档解析功能"
    )
    parser.add_argument("--file", help="要测试的办公文档文件路径")
    parser.add_argument(
        "--check-libreoffice",
        action="store_true",
        help="Only check LibreOffice installation",
    )

    args = parser.parse_args()

    # 首先检查 LibreOffice 是否已安装（解析办公文档的必需依赖）
    print("🔧 正在检查 LibreOffice 安装状态...")
    if not check_libreoffice_installation():
        return 1  # 未安装则返回非零退出码

    if args.check_libreoffice:
        print("✅ LibreOffice installation check passed!")
        return 0

    # 如果只是检查依赖，到这里就可以退出了
    if args.check_libreoffice:
        print("✅ LibreOffice 安装检查通过！")
        return 0

    # 如果不是仅检查依赖模式，则必须提供 --file 参数
    if not args.file:
        print(
            "❌ 错误: 当不使用 --check-libreoffice 时，必须提供 --file 参数"
        )
        parser.print_help()  # 显示帮助信息
        return 1

    # 运行解析测试（异步函数需要用 asyncio.run 执行）
    try:
        success = asyncio.run(test_office_document_parsing(args.file))
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
        return 1
    except Exception as e:
        print(f"\n❌ 未预期的错误: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())  # 将退出码传递给系统
