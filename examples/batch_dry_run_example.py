"""
Dry-run batch parsing example.

Lists supported files without running any parser.

Usage:
  - pip install:
      python examples/batch_dry_run_example.py examples/sample_docs --parser mineru
      python examples/batch_dry_run_example.py examples/sample_docs/projects examples/sample_docs/web --parser docling
      python examples/batch_dry_run_example.py examples/sample_docs --parser paddleocr
  - uv install:
      uv run python examples/batch_dry_run_example.py examples/sample_docs --parser mineru --recursive
      uv run python examples/batch_dry_run_example.py examples/sample_docs --parser mineru --no-recursive
"""

"""
批量解析干运行示例(干运行即练习性运行，不实际处理文件)

仅列出可被解析的文件，而不实际运行任何解析器。

使用方法：
    - pip 安装：
        python examples/batch_dry_run_example.py examples/sample_docs --parser mineru(使用mineru解析器)
        python examples/batch_dry_run_example.py examples/sample_docs/projects examples/sample_docs/web --parser docling(使用docling解析器)
        python examples/batch_dry_run_example.py examples/sample_docs --parser paddleocr(使用paddleocr解析器)
    - uv 安装：
        uv run python examples/batch_dry_run_example.py examples/sample_docs --parser mineru --recursive(递归遍历文件夹，包括子文件夹)
        uv run python examples/batch_dry_run_example.py examples/sample_docs --parser mineru --no-recursive(只看当前文件夹的文件，不进入子文件夹)
"""

import argparse # argparse是命令行参数解析器，使python脚本可以接收上面这种形式的命令

from raganything.batch_parser import BatchParser


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run batch parsing example") #创建一个ArgumentParser实例，用于翻译命令行参数
    parser.add_argument("paths", nargs="+", help="File paths or directories to scan") # 添加一个位置参数"paths"，它可以接受一个或多个文件路径或目录路径，nargs="+"表示至少需要一个参数
    parser.add_argument(
        "--parser", # 添加一个可选参数parser，指定要使用的解析器
        choices=["mineru", "docling", "paddleocr"], # 表示只能从这三个参数中选一个
        default="mineru", # 不添加参数的话默认使用mineru解析器
        help="Parser to use for file-type support", # 表示选定的解析器用于处理文件
    )
    parser.add_argument(
        "--output", # 添加一个可选参数output，用于指定结果输出的目录
        default="./batch_output", # 默认输出目录为当前目录下的batch_output文件夹
        help="Output directory (unused in dry-run, but required by API)",
    )
    parser.add_argument(
        "--recursive", # 添加一个可选参数recursive，表示是否递归搜索目录中的文件
        action=argparse.BooleanOptionalAction, # 将参数转换成布尔开关，后面可以不跟值，直接使用--recursive表示True，--no-recursive表示False
        default=True, # 默认开启递归搜索
        help="Search directories recursively",
    )
    args = parser.parse_args() # 将输入的命令行参数解析并打包成args对象，使得程序能够读懂用户输入的参数

    batch_parser = BatchParser(parser_type=args.parser, show_progress=False)
    # 创建一个BatchParser实例，用于批量处理文件，这里使用命令行中指定的解析器类型，并且关闭进度显示

    #此处是核心执行部分，这里的参数均为命令行中指定的参数
    result = batch_parser.process_batch(
        file_paths=args.paths,
        output_dir=args.output,
        recursive=args.recursive,
        dry_run=True, # 表示这是一个干运行，不实际处理文件
    )

    print(result.summary()) # 打印本次处理的汇总信息
    if result.successful_files: # 如果有可处理的文件，则列出这些文件的路径
        print("\nDry run: files that would be processed:")
        for file_path in result.successful_files:
            print(f"  - {file_path}")
    else:
        print("\nDry run: no supported files found.")

    return 0

# 如果文件是被直接运行的，在这里就会退出程序，如果是被别的文件导入的就不会在这里退出
if __name__ == "__main__":
    raise SystemExit(main())
