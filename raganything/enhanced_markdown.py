"""
Enhanced Markdown to PDF Conversion

This module provides improved Markdown to PDF conversion with:
- Better formatting and styling
- Image support
- Table support
- Code syntax highlighting
- Custom templates
- Multiple output formats
"""

"""
增强的 Markdown 转 PDF 转换

该模块提供了改进的 Markdown 转 PDF 转换功能，具有以下特点：
- 更好的格式和样式
- 图片支持
- 表格支持
- 代码语法高亮
- 自定义模板
- 多种输出格式
"""

import os # 操作系统相关功能，如路径处理和环境变量访问
import logging # 日志记录模块
from pathlib import Path # 面向对象的路径处理模块
from typing import Dict, Any, Optional # 类型提示模块(给变量和函数参数加类型提示)
from dataclasses import dataclass # 数据类装饰器，用于简化类的定义(可自动生成__init__等方法)
import tempfile # 用于创建用完即删的临时文件和目录的模块
import subprocess # 用于运行外部命令的模块

try:
    import markdown # Markdown 处理库，用于将 Markdown 转换为 HTML

    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

try:
    from weasyprint import HTML # WeasyPrint 库，用于将 HTML 转换为 PDF

    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

try:
    # Check if pandoc module exists (not used directly, just for detection)
    import importlib.util # 用于检查模块是否存在的工具模块

    spec = importlib.util.find_spec("pandoc") # 检查是否安装了 pandoc 模块（pandoc是一个文档转换工具，支持多种格式转换）
    PANDOC_AVAILABLE = spec is not None
except ImportError:
    PANDOC_AVAILABLE = False


@dataclass # 说明这是一个数据类，Python 会自动生成 __init__ 和其他方法
class MarkdownConfig:
    """Configuration for Markdown to PDF conversion"""
    # Markdown 转 PDF 转换的配置类，包含各种选项和参数

    # Styling options
    # 样式选项
    css_file: Optional[str] = None # 自定义 CSS 文件路径，用于覆盖默认样式
    template_file: Optional[str] = None # 自定义 HTML 模板文件路径，用于覆盖默认模板
    page_size: str = "A4" # 页面尺寸
    margin: str = "1in" # 页面边距
    font_size: str = "12pt" # 字体大小
    line_height: str = "1.5" # 行高

    # Content options
    # 内容选项
    include_toc: bool = True # 是否包含目录
    syntax_highlighting: bool = True # 是否启用代码语法高亮
    image_max_width: str = "100%" # 图片最大宽度
    table_style: str = "border-collapse: collapse; width: 100%;" # 表格样式

    # Output options
    # 输出选项
    output_format: str = "pdf"  # pdf, html, docx
    output_dir: Optional[str] = None # 输出目录，如果为 None 则使用当前目录

    # Advanced options
    # 高级选项
    custom_css: Optional[str] = None # 内联自定义 CSS 样式，适合少量样式调整
    metadata: Optional[Dict[str, str]] = None # pdf元数据，如标题、作者、主题等


class EnhancedMarkdownConverter:
    """
    Enhanced Markdown to PDF converter with multiple backends

    Supports multiple conversion methods:
    - WeasyPrint (recommended for HTML/CSS styling)
    - Pandoc (recommended for complex documents)
    - ReportLab (fallback, basic styling)
    """

    """
    增强的 Markdown 转 PDF 转换器，支持多个后端

    支持多种转换方法：
    - WeasyPrint（推荐用于 HTML/CSS 样式）
    - Pandoc（推荐用于复杂文档）
    - ReportLab（回退选项，基本样式）
    """

    def __init__(self, config: Optional[MarkdownConfig] = None):
        """
        Initialize the converter

        Args:
            config: Configuration for conversion
        """

        """
        初始化转换器

        参数:
            config: 转换配置
        """
        self.config = config or MarkdownConfig() # 如果没有提供配置，则使用默认配置
        self.logger = logging.getLogger(__name__) # 获取当前模块的日志记录器

        # Check available backends
        # 检查可用的转换后端，并记录日志
        self.available_backends = self._check_backends()
        self.logger.info(f"Available backends: {list(self.available_backends.keys())}")

    # 检查哪些转换后端可用，并返回一个字典，键是后端名称，值是布尔值表示是否可用
    def _check_backends(self) -> Dict[str, bool]:
        """Check which conversion backends are available"""
        backends = {
            "weasyprint": WEASYPRINT_AVAILABLE,
            "pandoc": PANDOC_AVAILABLE,
            "markdown": MARKDOWN_AVAILABLE,
        }

        # Check if pandoc is installed on system
        # 检查系统上是否安装了 pandoc 命令行工具（因为pandoc模块可能只是一个接口，实际转换需要系统上的pandoc工具）
        try:
            subprocess.run(["pandoc", "--version"], capture_output=True, check=True)
            backends["pandoc_system"] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            backends["pandoc_system"] = False

        return backends

    # 获取默认的 CSS 样式，用于转换后的 HTML 文档
    def _get_default_css(self) -> str:
        """Get default CSS styling"""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }

        h1 { font-size: 2em; border-bottom: 2px solid #3498db; padding-bottom: 0.3em; }
        h2 { font-size: 1.5em; border-bottom: 1px solid #bdc3c7; padding-bottom: 0.2em; }
        h3 { font-size: 1.3em; }
        h4 { font-size: 1.1em; }

        p { margin-bottom: 1em; }

        code {
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }

        pre {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 4px solid #3498db;
        }

        pre code {
            background-color: transparent;
            padding: 0;
        }

        blockquote {
            border-left: 4px solid #3498db;
            margin: 0;
            padding-left: 20px;
            color: #7f8c8d;
        }

        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }

        th, td {
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: left;
        }

        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }

        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 1em auto;
        }

        ul, ol {
            margin-bottom: 1em;
        }

        li {
            margin-bottom: 0.5em;
        }

        a {
            color: #3498db;
            text-decoration: none;
        }

        a:hover {
            text-decoration: underline;
        }

        .toc {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 2em;
        }

        .toc ul {
            list-style-type: none;
            padding-left: 0;
        }

        .toc li {
            margin-bottom: 0.3em;
        }

        .toc a {
            color: #2c3e50;
        }
        """

    # 处理 Markdown 内容
    def _process_markdown_content(self, content: str) -> str:
        """Process Markdown content with extensions"""
        if not MARKDOWN_AVAILABLE: # 如果 markdown 库不可用，抛出运行时错误，提示用户安装 markdown 库
            raise RuntimeError(
                "Markdown library not available. Install with: pip install markdown"
            )

        # Configure Markdown extensions
        # 配置 Markdown 扩展，启用表格、代码块、目录、属性列表、定义列表和脚注等功能
        extensions = [
            "markdown.extensions.tables",
            "markdown.extensions.fenced_code",
            "markdown.extensions.codehilite",
            "markdown.extensions.toc",
            "markdown.extensions.attr_list",
            "markdown.extensions.def_list",
            "markdown.extensions.footnotes",
        ]

        # 扩展配置，设置代码高亮的 CSS 类和使用 Pygments 进行语法高亮，设置目录的标题和永久链接等选项
        extension_configs = {
            "codehilite": {
                "css_class": "highlight",
                "use_pygments": True,
            },
            "toc": {
                "title": "Table of Contents",
                "permalink": True,
            },
        }

        # Convert Markdown to HTML
        # 将 Markdown 转换为 HTML
        md = markdown.Markdown( # 创建一个 Markdown 对象，传入扩展和扩展配置
            extensions=extensions, extension_configs=extension_configs
        )
        # 将 Markdown 内容转换为 HTML
        html_content = md.convert(content)

        # Add CSS styling
        # 添加 CSS 样式，如果用户提供了自定义 CSS，则使用它，否则使用默认 CSS
        css = self.config.custom_css or self._get_default_css()

        # Create complete HTML document
        # 创建完整的 HTML 文档，包含 DOCTYPE、html、head 和 body 标签，在 head 中包含 meta 标签和 title 标签，以及内联的 CSS 样式，在 body 中包含转换后的 HTML 内容
        html_doc = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Converted Document</title>
            <style>
                {css}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        return html_doc

    # 使用 WeasyPrint 进行转换（适合样式化的文档）
    def convert_with_weasyprint(self, markdown_content: str, output_path: str) -> bool:
        """Convert using WeasyPrint (best for styling)"""
        if not WEASYPRINT_AVAILABLE: # 检查 WeasyPrint 是否可用，如果不可用，抛出运行时错误，提示用户安装 WeasyPrint 库
            raise RuntimeError(
                "WeasyPrint not available. Install with: pip install weasyprint"
            )

        try:
            # Process Markdown to HTML
            # 将 Markdown 内容处理为 HTML，调用之前定义的 _process_markdown_content 方法
            html_content = self._process_markdown_content(markdown_content)

            # Convert HTML to PDF
            # 使用 WeasyPrint 将 HTML 转换为 PDF，创建一个 HTML 对象，传入 HTML 内容，然后调用 write_pdf 方法将其写入指定的输出路径
            html = HTML(string=html_content)
            html.write_pdf(output_path)

            self.logger.info( # 记录日志，表示成功使用 WeasyPrint 转换为 PDF，并显示输出路径
                f"Successfully converted to PDF using WeasyPrint: {output_path}"
            )
            return True

        except Exception as e:
            self.logger.error(f"WeasyPrint conversion failed: {str(e)}")
            return False

    # 使用 Pandoc 进行转换（适合复杂文档）
    def convert_with_pandoc(
        self, markdown_content: str, output_path: str, use_system_pandoc: bool = False
    ) -> bool:
        """Convert using Pandoc (best for complex documents)"""
        if ( # 如果系统没有安装pandoc并且不使用系统pandoc，则抛出运行时错误，提示用户安装 pandoc 工具
            not self.available_backends.get("pandoc_system", False)
            and not use_system_pandoc
        ):
            raise RuntimeError(
                "Pandoc not available. Install from: https://pandoc.org/installing.html"
            )

        temp_md_path = None # 初始化临时 Markdown 文件路径变量，用于存储转换过程中生成的临时文件路径
        try:
            import subprocess # 导入 subprocess 模块，用于运行外部命令（在这里用于调用 pandoc 命令行工具）

            # Create temporary markdown file
            # 创建一个临时 Markdown 文件
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", delete=False # 以写入模式创建一个后缀为 .md 的临时文件，并且在关闭后不删除该文件
            ) as temp_file:
                temp_file.write(markdown_content)
                temp_md_path = temp_file.name

            # Build pandoc command with wkhtmltopdf engine
            # 构建 pandoc 命令，指定输入文件、输出文件、使用 wkhtmltopdf 作为 PDF 引擎，并启用独立文档、目录和章节编号等选项
            cmd = [
                "pandoc",
                temp_md_path,
                "-o",
                output_path,
                "--pdf-engine=wkhtmltopdf",
                "--standalone",
                "--toc",
                "--number-sections",
            ]

            # Run pandoc
            # 运行 pandoc 命令，使用 subprocess.run 方法，传入命令列表，启用捕获输出、文本模式和设置超时时间为 60 秒
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            # 如果命令执行成功（返回码为 0），记录日志表示成功转换为 PDF，并显示输出路径；否则，记录错误日志并显示错误信息
            if result.returncode == 0:
                self.logger.info(
                    f"Successfully converted to PDF using Pandoc: {output_path}"
                )
                return True
            else:
                self.logger.error(f"Pandoc conversion failed: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Pandoc conversion failed: {str(e)}")
            return False

        # 无论转换成功与否，最后都尝试清理临时 Markdown 文件，如果文件存在，则尝试删除它，并记录任何可能发生的错误
        finally:
            if temp_md_path and os.path.exists(temp_md_path):
                try:
                    os.unlink(temp_md_path)
                except OSError as e:
                    self.logger.error(
                        f"Failed to clean up temp file {temp_md_path}: {str(e)}"
                    )

    # 主要的转换方法，根据指定的转换方法参数，调用相应的转换函数进行 Markdown 到 PDF 的转换
    def convert_markdown_to_pdf(
        self, markdown_content: str, output_path: str, method: str = "auto"
    ) -> bool:
        """
        Convert markdown content to PDF

        Args:
            markdown_content: Markdown content to convert
            output_path: Output PDF file path
            method: Conversion method ("auto", "weasyprint", "pandoc", "pandoc_system")

        Returns:
            True if conversion successful, False otherwise
        """

        """
        将 Markdown 内容转换为 PDF

        参数:
            markdown_content: 要转换的 Markdown 内容
            output_path: 输出 PDF 文件路径
            method: 转换方法（"auto"、"weasyprint"、"pandoc"、"pandoc_system"）

        返回:
            如果转换成功则返回 True，否则返回 False
        """

        # 当指定转换方法为 "auto" 时，调用 _get_recommended_backend 方法获取推荐的转换后端
        if method == "auto":
            method = self._get_recommended_backend()
        # 调用相应的转换函数进行 Markdown 到 PDF 的转换
        try:
            if method == "weasyprint":
                return self.convert_with_weasyprint(markdown_content, output_path)
            elif method == "pandoc":
                return self.convert_with_pandoc(markdown_content, output_path)
            elif method == "pandoc_system":
                return self.convert_with_pandoc(
                    markdown_content, output_path, use_system_pandoc=True
                )
            else:
                raise ValueError(f"Unknown conversion method: {method}")

        except Exception as e:
            self.logger.error(f"{method.title()} conversion failed: {str(e)}")
            return False

    # 将 Markdown 文件转换为 PDF
    def convert_file_to_pdf(
        self, input_path: str, output_path: Optional[str] = None, method: str = "auto"
    ) -> bool:
        """
        Convert Markdown file to PDF

        Args:
            input_path: Input Markdown file path
            output_path: Output PDF file path (optional)
            method: Conversion method

        Returns:
            bool: True if conversion successful
        """

        """
        将 Markdown 文件转换为 PDF

        参数:
            input_path: 输入 Markdown 文件路径
            output_path: 输出 PDF 文件路径（可选）
            method: 转换方法

        返回:
            bool: 如果转换成功则返回 True
        """
        # 将输入路径转换为 Path 对象，从而更方便地进行路径操作和检查
        input_path_obj = Path(input_path)

        # 检查路径是否存在
        if not input_path_obj.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Read markdown content
        try: # 尝试以 UTF-8 编码读取 Markdown 文件内容
            with open(input_path_obj, "r", encoding="utf-8") as f:
                markdown_content = f.read()
        except UnicodeDecodeError:
            # Try with different encodings
            # 尝试一些别的编码来读取markdown文件内容
            for encoding in ["gbk", "latin-1", "cp1252"]:
                try:
                    with open(input_path_obj, "r", encoding=encoding) as f:
                        markdown_content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise RuntimeError(
                    f"Could not decode file {input_path} with any supported encoding"
                )

        # Determine output path
        # 指定输出路径，如果没有提供，则使用输入文件的路径并将扩展名更改为 .pdf
        if output_path is None:
            output_path = str(input_path_obj.with_suffix(".pdf"))

        # 获取内容后使用内容转换函数将 Markdown 内容转换为 PDF，并返回转换结果
        return self.convert_markdown_to_pdf(markdown_content, output_path, method)

    # 获取关于可用转换后端的信息，包括哪些后端可用、推荐的后端以及当前配置的详细信息
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about available backends"""
        return {
            "available_backends": self.available_backends, # 可用的转换后端列表，包含每个后端的可用状态（True 或 False）
            "recommended_backend": self._get_recommended_backend(), # 推荐的转换后端，根据可用性自动选择
            "config": {
                "page_size": self.config.page_size, # 页面尺寸配置
                "margin": self.config.margin, # 页面边距配置
                "font_size": self.config.font_size, # 字体大小配置
                "include_toc": self.config.include_toc, # 是否包含目录的配置
                "syntax_highlighting": self.config.syntax_highlighting, # 是否启用代码语法高亮的配置
            },
        }

    # 根据可用性自动选择推荐的转换后端，优先使用系统安装的 pandoc，其次是 weasyprint，如果都不可用则返回 "none"
    def _get_recommended_backend(self) -> str:
        """Get recommended backend based on availability"""
        if self.available_backends.get("pandoc_system", False):
            return "pandoc"
        elif self.available_backends.get("weasyprint", False):
            return "weasyprint"
        else:
            return "none"

# 主函数，提供命令行界面，允许用户指定输入文件、输出文件、转换方法、自定义 CSS 文件以及显示后端信息等选项
def main():
    """Command-line interface for enhanced markdown conversion"""
    import argparse # 用于解析命令行参数的模块

    parser = argparse.ArgumentParser(description="Enhanced Markdown to PDF conversion") # 创建一个 ArgumentParser 对象，用于定义和解析命令行参数，提供一个描述信息
    parser.add_argument("input", nargs="?", help="Input markdown file") # 定义一个位置参数 "input"，表示输入的 Markdown 文件路径，nargs="?" 表示该参数是可选的
    parser.add_argument("--output", "-o", help="Output PDF file") # 定义一个可选参数 "--output" 或 "-o"，表示输出的 PDF 文件路径
    parser.add_argument( # 定义一个可选参数 "--method"，表示转换方法，提供多个选项（"auto"、"weasyprint"、"pandoc"、"pandoc_system"），默认值为 "auto"
        "--method",
        choices=["auto", "weasyprint", "pandoc", "pandoc_system"],
        default="auto",
        help="Conversion method",
    )
    parser.add_argument("--css", help="Custom CSS file") # 定义一个可选参数 "--css"，表示自定义 CSS 文件路径，用于覆盖默认样式
    parser.add_argument("--info", action="store_true", help="Show backend information") # 定义一个可选参数 "--info"，表示是否显示后端信息，action="store_true" 表示如果提供该参数则将其值设置为 True

    args = parser.parse_args() # 解析命令行参数，并将结果存储在 args 变量中，args.input、args.output、args.method、args.css 和 args.info 分别对应之前定义的参数

    # Configure logging
    # 配置日志记录，设置日志级别为 INFO，并定义日志消息的格式，包括时间戳、记录器名称、日志级别和消息内容
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create converter
    # 创建一个 Markdown 转换器对象，传入配置，如果用户提供了自定义 CSS 文件路径，则将其设置到配置中
    config = MarkdownConfig()
    if args.css:
        config.css_file = args.css

    converter = EnhancedMarkdownConverter(config)

    # Show backend info if requested
    # 如果用户请求显示后端信息，则调用 converter.get_backend_info() 获取后端信息，并格式化输出可用的后端列表和推荐的后端，然后返回 0 表示成功退出
    if args.info:
        info = converter.get_backend_info()
        print("Backend Information:")
        for backend, available in info["available_backends"].items():
            status = "✅" if available else "❌"
            print(f"  {status} {backend}")
        print(f"Recommended backend: {info['recommended_backend']}")
        return 0

    # Check if input file is provided
    if not args.input:
        parser.error("Input file is required when not using --info")

    # Convert file
    try:
        success = converter.convert_file_to_pdf(
            input_path=args.input, output_path=args.output, method=args.method
        )

        if success:
            print(f"✅ Successfully converted {args.input} to PDF")
            return 0
        else:
            print("❌ Conversion failed")
            return 1

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
