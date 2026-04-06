#!/usr/bin/env python3
"""
Image Format Parsing Test Script for RAG-Anything

This script demonstrates how to parse various image formats
using MinerU, including JPG, PNG, BMP, TIFF, GIF, and WebP files.

Requirements:
- PIL/Pillow library for format conversion
- RAG-Anything package

Usage:
    python image_format_test.py --file path/to/image.bmp
"""

"""
RAG-Anything的图片格式解析测试脚本

这个脚本演示了如何使用MinerU解析各种图片格式，包括JPG、PNG、BMP、TIFF、GIF和WebP文件

要求：
- PIL/Pillow库用于格式转换
- RAG-Anything包

使用方法：
    python image_format_test.py --file path/to/image.bmp
"""

import argparse # 用于解析命令行参数的模块
import asyncio # 用于处理异步操作的模块
import sys # 用于访问系统相关功能的模块
from pathlib import Path # 用于处理文件路径的模块
from raganything import RAGAnything

# 检查PIL/Pillow库是否安装和可用
def check_pillow_installation():
    """Check if PIL/Pillow is installed and available"""
    try: # 尝试导入PIL库，如果成功则打印版本信息并返回True，如果导入失败则提示用户安装Pillow并返回False
        from PIL import Image

        print(
            f"✅ PIL/Pillow found: PIL version {Image.__version__ if hasattr(Image, '__version__') else 'Unknown'}"
        )
        return True
    except ImportError:
        print("❌ PIL/Pillow not found. Please install Pillow:")
        print("  pip install Pillow")
        return False

# 获取图片的详细信息，包括格式、模式、尺寸和是否具有透明度
def get_image_info(image_path: Path):
    """Get detailed image information"""
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            return {
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
                "has_transparency": img.mode in ("RGBA", "LA")
                or "transparency" in img.info,
            }
    except Exception as e:
        return {"error": str(e)}

# 使用minerU测试图片格式解析功能（这是一个异步函数，可以多线程执行）
async def test_image_format_parsing(file_path: str):
    """Test image format parsing with MinerU"""

    print(f"🧪 Testing image format parsing: {file_path}")

    # Check if file exists and is a supported image format
    # 验证文件是否存在，并检查文件扩展名是否在支持的图片格式列表中，如果文件不存在或格式不受支持，则打印错误信息并返回False
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"❌ File does not exist: {file_path}")
        return False

    # 列出支持的图片格式扩展名，如果文件的扩展名不在支持的列表中，则提示用户并返回False
    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".tif",
        ".gif",
        ".webp",
    }
    if file_path.suffix.lower() not in supported_extensions:
        print(f"❌ Unsupported file format: {file_path.suffix}")
        print(f"   Supported formats: {', '.join(supported_extensions)}")
        return False

    print(f"📸 File format: {file_path.suffix.upper()}")
    print(f"📏 File size: {file_path.stat().st_size / 1024:.1f} KB") # 输出文件大小，单位为KB

    # Get detailed image information
    # 获取图片的详细信息，包括格式、模式、尺寸和是否具有透明度，如果获取信息成功则打印这些信息，否则打印错误信息
    img_info = get_image_info(file_path)
    if "error" not in img_info:
        print("🖼️  Image info:")
        print(f"   • Format: {img_info['format']}")
        print(f"   • Mode: {img_info['mode']}")
        print(f"   • Size: {img_info['size'][0]}x{img_info['size'][1]}")
        print(f"   • Has transparency: {img_info['has_transparency']}")

    # Check format compatibility with MinerU
    # 检查文件格式是否与MinerU兼容，MinerU原生支持JPG、JPEG和PNG格式，如果文件格式不在这些格式中，则提示用户将其转换为PNG以获得兼容性
    mineru_native_formats = {".jpg", ".jpeg", ".png"}
    needs_conversion = file_path.suffix.lower() not in mineru_native_formats

    if needs_conversion:
        print(
            f"ℹ️  Format {file_path.suffix.upper()} will be converted to PNG for MinerU compatibility"
        )
    else:
        print(f"✅ Format {file_path.suffix.upper()} is natively supported by MinerU")

    # Initialize RAGAnything (only for parsing functionality)
    # 初始化RAGAnything实例，用于调用MinerU的解析功能，注意这里我们只使用解析功能，不涉及其他功能
    rag = RAGAnything()

    try:
        # Test image parsing with MinerU
        # 使用MinerU测试图片解析功能
        print("\n🔄 Testing image parsing with MinerU...")
        content_list, md_content = await rag.parse_document(
            file_path=str(file_path),
            output_dir="./test_output",
            parse_method="ocr",  # Images use OCR method # 图片使用OCR方法进行解析
            display_stats=True,
        )

        print("✅ Parsing successful!")
        print(f"   📊 Content blocks: {len(content_list)}")
        print(f"   📝 Markdown length: {len(md_content)} characters")

        # Analyze content types
        # 分析内容类型，统计不同类型的内容块数量，例如文本、图片、表格等，并打印这些统计信息
        content_types = {}
        for item in content_list:
            if isinstance(item, dict): # 如果内容块是一个字典，则获取其类型并统计数量
                content_type = item.get("type", "unknown")
                content_types[content_type] = content_types.get(content_type, 0) + 1

        # 如果存在内容类型统计信息，则打印每种内容类型的数量分布，例如文本块、图片块、表格块等
        if content_types:
            print("   📋 Content distribution:")
            for content_type, count in sorted(content_types.items()):
                print(f"      • {content_type}: {count}")

        # Display extracted text (if any)
        # 显示提取的文本内容（如果有的话），如果Markdown内容不为空，则打印前500个字符的预览，否则提示没有从图片中提取到文本
        if md_content.strip():
            print("\n📄 Extracted text preview (first 500 characters):")
            preview = md_content.strip()[:500]
            print(f"   {preview}{'...' if len(md_content) > 500 else ''}")
        else:
            print("\n📄 No text extracted from the image")

        # Display image processing results
        # 显示图片处理结果，检查内容块中是否有类型为“image”的块，如果有则打印这些块的数量和相关信息，例如图片路径和标题等
        image_items = [
            item
            for item in content_list
            if isinstance(item, dict) and item.get("type") == "image"
        ]
        if image_items:
            print(f"\n🖼️  Found {len(image_items)} processed image(s):")
            for i, item in enumerate(image_items, 1):
                print(f"   {i}. Image path: {item.get('img_path', 'N/A')}")
                caption = item.get("image_caption", item.get("img_caption", []))
                if caption:
                    print(f"      Caption: {caption[0] if caption else 'N/A'}")

        # Display text blocks (OCR results)
        # 显示文本块（OCR结果），检查内容块中是否有类型为“text”的块，如果有则打印这些块的数量和相关信息，例如文本内容的预览等
        text_items = [
            item
            for item in content_list
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if text_items:
            print("\n📝 OCR text blocks found:")
            for i, item in enumerate(text_items, 1):
                text_content = item.get("text", "")
                if text_content.strip():
                    preview = text_content.strip()[:200]
                    print(
                        f"   {i}. {preview}{'...' if len(text_content) > 200 else ''}"
                    )

        # Check for any tables detected in the image
        # 检查图片中是否检测到任何表格，检查内容块中是否有类型为“table”的块，如果有则打印这些块的数量和相关信息，例如表格内容的预览等
        table_items = [
            item
            for item in content_list
            if isinstance(item, dict) and item.get("type") == "table"
        ]
        if table_items:
            print(f"\n📊 Found {len(table_items)} table(s) in image:")
            for i, item in enumerate(table_items, 1):
                print(f"   {i}. Table detected with content")

        print("\n🎉 Image format parsing test completed successfully!")
        print("📁 Output files saved to: ./test_output")
        return True

    except Exception as e:
        print(f"\n❌ Image format parsing failed: {str(e)}")
        import traceback

        print(f"   Full error: {traceback.format_exc()}")
        return False

# 主函数，用于解析命令行参数并执行测试逻辑，根据用户提供的参数决定是仅检查PIL/Pillow安装还是进行图片格式解析测试，如果在执行过程中发生异常则捕获并打印错误信息
def main():
    """Main function"""
    parser = argparse.ArgumentParser( # 创建一个命令行参数解析器，用于测试图片格式解析功能
        description="Test image format parsing with MinerU"
    )
    parser.add_argument("--file", help="Path to the image file to test") # 添加一个参数用于指定要测试的图片文件路径
    parser.add_argument(
        "--check-pillow", action="store_true", help="Only check PIL/Pillow installation" # 添加一个参数用于仅检查PIL/Pillow库的安装情况，如果用户提供了这个参数则只进行安装检查而不执行图片解析测试
    )

    args = parser.parse_args() # 解析命令行参数

    # Check PIL/Pillow installation
    # 检查PIL/Pillow库的安装情况，如果检查失败则返回1表示错误，如果用户仅要求检查安装则在检查通过后返回0表示成功，否则继续执行图片格式解析测试
    print("🔧 Checking PIL/Pillow installation...")
    if not check_pillow_installation():
        return 1

    if args.check_pillow:
        print("✅ PIL/Pillow installation check passed!")
        return 0

    # If not just checking dependencies, file argument is required
    # 如果不是仅检查依赖项，则需要提供文件参数，如果用户没有提供文件参数则打印错误信息并显示帮助信息，然后返回1表示错误
    if not args.file:
        print("❌ Error: --file argument is required when not using --check-pillow")
        parser.print_help()
        return 1

    # Run the parsing test
    # 运行解析测试
    try:
        success = asyncio.run(test_image_format_parsing(args.file))
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
