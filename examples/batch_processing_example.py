#!/usr/bin/env python
"""
Batch Processing Example for RAG-Anything

This example demonstrates how to use the batch processing capabilities
to process multiple documents in parallel for improved throughput.

Features demonstrated:
- Basic batch processing with BatchParser
- Asynchronous batch processing
- Integration with RAG-Anything
- Error handling and progress tracking
- File filtering and directory processing
"""

"""
批量处理演示

这个示例演示了如何使用批量处理功能来并行处理多个文档以提高吞吐量

演示的功能：
- 使用 BatchParser 进行基本批量处理
- 异步批量处理
- 与 RAG-Anything 的集成
- 错误处理和进度跟踪
- 文件过滤和目录处理
"""

import asyncio
import logging
from pathlib import Path
import tempfile
import time

# Add project root directory to Python path
# 将项目根目录添加到 Python 路径中
import sys

sys.path.append(str(Path(__file__).parent.parent))

from raganything import RAGAnything, RAGAnythingConfig
from raganything.batch_parser import BatchParser

# 生成用于批量处理测试的示例文档
def create_sample_documents():
    """Create sample documents for batch processing testing"""
    temp_dir = Path(tempfile.mkdtemp()) # 创建临时目录
    sample_files = []

    # 创建各种文档类型，包括三个文本文档，两个Markdown文档(里面带多个#指的是markdown的标题以及相应内容，这里不做翻译)
    # Create various document types
    documents = {
        "document1.txt": "This is a simple text document for testing batch processing.",
        "document2.txt": "Another text document with different content.",
        "document3.md": """# Markdown Document

## Introduction
This is a markdown document for testing.

### Features
- Markdown formatting
- Code blocks
- Lists

```python
def example():
    return "Hello from markdown"
```
""",
        "report.txt": """Business Report

Executive Summary:
This report demonstrates batch processing capabilities.

Key Findings:
1. Parallel processing improves throughput
2. Progress tracking enhances user experience
3. Error handling ensures reliability

Conclusion:
Batch processing is essential for large-scale document processing.
""",
        "notes.md": """# Meeting Notes

## Date: 2024-01-15

### Attendees
- Alice Johnson
- Bob Smith
- Carol Williams

### Discussion Topics
1. **Batch Processing Implementation**
   - Parallel document processing
   - Progress tracking
   - Error handling strategies

2. **Performance Metrics**
   - Target: 100 documents/hour
   - Memory usage: < 4GB
   - Success rate: > 95%

### Action Items
- [ ] Implement batch processing
- [ ] Add progress bars
- [ ] Test with large document sets
- [ ] Optimize memory usage

### Next Steps
Continue development and testing of batch processing features.
""",
    }

    # Create files
    # 创建测试文件
    for filename, content in documents.items():
        file_path = temp_dir / filename # 拼接得到文件的完整路径(这里的/指的是路径拼接，不是除法)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content) # 写入文件内容
        sample_files.append(str(file_path)) # 将文件路径添加到列表中

    return sample_files, temp_dir # 返回文件路径列表和临时目录路径

# 基础批量处理演示
def demonstrate_basic_batch_processing():
    """Demonstrate basic batch processing functionality"""
    print("\n" + "=" * 60)
    print("BASIC BATCH PROCESSING DEMONSTRATION")
    print("=" * 60)

    # Create sample documents
    # 调用上面的函数创建示例文档
    sample_files, temp_dir = create_sample_documents()

    try:
        # 打印示例文件信息
        print(f"Created {len(sample_files)} sample documents in: {temp_dir}")
        for file_path in sample_files:
            print(f"  - {Path(file_path).name}")

        # Create batch parser
        # 创建批量解析器实例，配置解析器类型、并行工作线程数、进度显示和超时时间等参数
        batch_parser = BatchParser(
            parser_type="mineru",
            max_workers=3,
            show_progress=True,
            timeout_per_file=60,
            skip_installation_check=True,  # 演示时跳过安装校验

        )

        print("\nBatch parser configured:")
        print("  - Parser type: mineru")
        print("  - Max workers: 3")
        print("  - Progress tracking: enabled")
        print("  - Timeout per file: 60 seconds")

        # Check supported extensions
        # 获取并打印可处理的文件名后缀
        supported_extensions = batch_parser.get_supported_extensions()
        print(f"  - Supported extensions: {supported_extensions}")

        # Filter files to supported types
        # 从所有文件中过滤出支持的文件，并打印过滤结果
        supported_files = batch_parser.filter_supported_files(sample_files)
        print("\nFile filtering results:")
        print(f"  - Total files: {len(sample_files)}")
        print(f"  - Supported files: {len(supported_files)}")

        # Process batch
        # 创建输出目录
        output_dir = temp_dir / "batch_output"
        print("\nStarting batch processing...")
        print(f"Output directory: {output_dir}")

        # 执行批量处理并计时
        start_time = time.time()
        result = batch_parser.process_batch(
            file_paths=supported_files,
            output_dir=str(output_dir),
            parse_method="auto",
            recursive=False,
        )
        processing_time = time.time() - start_time

        # Display results
        # 打印批量处理结果摘要、处理时间和成功率等统计信息
        print("\n" + "-" * 40)
        print("BATCH PROCESSING RESULTS")
        print("-" * 40)
        print(result.summary())
        print(f"Total processing time: {processing_time:.2f} seconds")
        print(f"Success rate: {result.success_rate:.1f}%")

        # 打印成功处理的文件和失败的文件列表，以及失败文件的错误信息
        if result.successful_files:
            print("\nSuccessfully processed files:")
            for file_path in result.successful_files:
                print(f"  ✅ {Path(file_path).name}")

        if result.failed_files:
            print("\nFailed files:")
            for file_path in result.failed_files:
                error = result.errors.get(file_path, "Unknown error")
                print(f"  ❌ {Path(file_path).name}: {error}")

        return result

    # 提示处理过程中可能出现的异常，并打印错误信息
    except Exception as e:
        print(f"❌ Batch processing demonstration failed: {str(e)}")
        return None

# 异步批量处理演示，文件处理并行执行以提高性能
async def demonstrate_async_batch_processing():
    """Demonstrate asynchronous batch processing"""
    print("\n" + "=" * 60)
    print("ASYNCHRONOUS BATCH PROCESSING DEMONSTRATION")
    print("=" * 60)

    # Create sample documents
    # 导入上面创建的文件路径和目录
    sample_files, temp_dir = create_sample_documents()

    try:
        print(f"Processing {len(sample_files)} documents asynchronously...")

        # Create batch parser
        # 创建批量解析器实例，配置解析器类型、并行工作线程数、进度显示和跳过安装检查等参数
        batch_parser = BatchParser(
            parser_type="mineru",
            max_workers=2,
            show_progress=True,
            skip_installation_check=True,
        )

        # Process batch asynchronously
        # 创建输出目录
        output_dir = temp_dir / "async_output"

        # 执行异步批量处理并计时(await关键字用于等待异步操作完成，在此期间程序可以继续执行其他任务)
        start_time = time.time()
        result = await batch_parser.process_batch_async(
            file_paths=sample_files,
            output_dir=str(output_dir),
            parse_method="auto",
            recursive=False,
        )
        processing_time = time.time() - start_time

        # Display results
        # 打印异步批量处理结果摘要、处理时间和成功率等统计信息
        print("\n" + "-" * 40)
        print("ASYNC BATCH PROCESSING RESULTS")
        print("-" * 40)
        print(result.summary())
        print(f"Async processing time: {processing_time:.2f} seconds")
        print(f"Success rate: {result.success_rate:.1f}%")

        return result

    # 异常显示
    except Exception as e:
        print(f"❌ Async batch processing demonstration failed: {str(e)}")
        return None

# RAG-Anything集成演示，展示如何将批量处理与RAG-Anything的文档处理管道集成以实现更强大的功能
async def demonstrate_rag_integration():
    """Demonstrate batch processing integration with RAG-Anything"""
    print("\n" + "=" * 60)
    print("RAG-ANYTHING BATCH INTEGRATION DEMONSTRATION")
    print("=" * 60)

    # Create sample documents
    sample_files, temp_dir = create_sample_documents()

    try:
        # Initialize RAG-Anything with temporary storage
        # 初始化RAG-Anything实例，配置临时存储目录和启用各种处理功能，以及设置最大并行文件数
        config = RAGAnythingConfig(
            working_dir=str(temp_dir / "rag_storage"),
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
            max_concurrent_files=2,
        )

        rag = RAGAnything(config=config)

        print("RAG-Anything initialized with batch processing capabilities")

        # Show available batch methods
        # 获取并打印RAG-Anything实例中可用的批处理方法列表
        batch_methods = [method for method in dir(rag) if "batch" in method.lower()]
        print(f"Available batch methods: {batch_methods}")

        # Demonstrate batch processing with RAG integration
        # 演示使用RAG集成的批量处理功能，处理示例文件并输出结果摘要和成功率等统计信息
        print(f"\nProcessing {len(sample_files)} documents with RAG integration...")

        # Use the RAG-integrated batch processing
        # 使用RAG集成的批量处理方法
        try:
            # Process documents in batch
            # 批量处理文档并输出结果摘要和成功率等统计信息
            result = rag.process_documents_batch(
                file_paths=sample_files, # 处理示例文件列表
                output_dir=str(temp_dir / "rag_batch_output"), # 输出目录
                max_workers=2, # 最大并行文件数
                show_progress=True, # 显示处理进度
            )

            print("\n" + "-" * 40)
            print("RAG BATCH PROCESSING RESULTS")
            print("-" * 40)
            print(result.summary()) # 打印处理结果摘要
            print(f"Success rate: {result.success_rate:.1f}%") # 打印成功率

            # Demonstrate batch processing with full RAG integration
            # 演示使用完整RAG集成的批量处理功能，处理示例文件并输出结果摘要、处理时间和成功率等统计信息
            print("\nProcessing documents with full RAG integration...")

            rag_result = await rag.process_documents_with_rag_batch(
                file_paths=sample_files[:2],  # Process subset for demo # 演示时处理文件子集
                output_dir=str(temp_dir / "rag_full_output"),
                max_workers=1,
                show_progress=True,
            )

            print("\n" + "-" * 40)
            print("FULL RAG INTEGRATION RESULTS")
            print("-" * 40)
            print(f"Parse result: {rag_result['parse_result'].summary()}")
            print(
                f"RAG processing time: {rag_result['total_processing_time']:.2f} seconds"
            )
            print(
                f"Successfully processed with RAG: {rag_result['successful_rag_files']}"
            )
            print(f"Failed RAG processing: {rag_result['failed_rag_files']}")

            return rag_result

        except Exception as e:
            print(f"⚠️ RAG integration demo completed with limitations: {str(e)}")
            print(
                "Note: This is expected in environments without full API configuration"
            )
            return None

    except Exception as e:
        print(f"❌ RAG integration demonstration failed: {str(e)}")
        return None

# 目录处理演示，展示如何使用批量处理功能递归处理整个目录中的文件
def demonstrate_directory_processing():
    """Demonstrate processing entire directories"""
    """展示处理整个目录的功能"""
    print("\n" + "=" * 60)
    print("DIRECTORY PROCESSING DEMONSTRATION")
    print("=" * 60)

    # Create a directory structure with nested files
    # 创建一个包含嵌套文件的目录结构
    temp_dir = Path(tempfile.mkdtemp())

    # Create main directory files
    # 在主目录中创建文件
    main_files = {
        "overview.txt": "Main directory overview document",
        "readme.md": "# Project README\n\nThis is the main project documentation.",
    }

    # Create subdirectory
    # 创建子目录
    sub_dir = temp_dir / "subdirectory"
    sub_dir.mkdir() # 创建文件夹(子目录)
    # 子目录中的文件
    sub_files = {
        "details.txt": "Detailed information in subdirectory",
        "notes.md": "# Notes\n\nAdditional notes and information.",
    }

    # Write all files
    # 将所有文件写入磁盘，并收集它们的路径以供批量处理使用
    all_files = []
    for filename, content in main_files.items():
        file_path = temp_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content) # 写入文件内容
        all_files.append(str(file_path)) # 将文件路径添加到列表中

    # 将子目录中的文件写入磁盘，并添加到文件列表中
    for filename, content in sub_files.items():
        file_path = sub_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content) # 写入文件内容
        all_files.append(str(file_path)) # 将文件路径添加到列表中

    try:
        print("Created directory structure:")
        print(f"  Main directory: {temp_dir}")
        print(f"  Files in main: {list(main_files.keys())}")
        print(f"  Subdirectory: {sub_dir}")
        print(f"  Files in sub: {list(sub_files.keys())}")

        # Create batch parser
        # 创建批量解析器实例，配置解析器类型、并行工作线程数、进度显示和跳过安装检查等参数
        batch_parser = BatchParser(
            parser_type="mineru", # 解析器类型
            max_workers=2, # 最大并行文件数
            show_progress=True, # 显示处理进度
            skip_installation_check=True, # 演示时跳过安装校验
        )

        # Process entire directory recursively
        # 使用批量处理功能递归处理整个目录中的文件，并输出结果摘要、处理时间和成功率等统计信息
        print("\nProcessing entire directory recursively...")

        result = batch_parser.process_batch(
            file_paths=[str(temp_dir)],  # 传递目录路径
            output_dir=str(temp_dir / "directory_output"), # 输出目录
            parse_method="auto", # 解析方法为自动
            recursive=True,  # 递归处理子目录
        )

        print("\n" + "-" * 40)
        print("DIRECTORY PROCESSING RESULTS")
        print("-" * 40)
        print(result.summary()) # 打印处理结果摘要
        print(f"Total files found and processed: {result.total_files}") # 打印找到并处理的文件总数
        print(f"Success rate: {result.success_rate:.1f}%") # 打印成功率

        if result.successful_files: # 打印成功处理的文件列表
            print("\nSuccessfully processed:")
            for file_path in result.successful_files:
                relative_path = Path(file_path).relative_to(temp_dir)
                print(f"  ✅ {relative_path}")

        return result

    except Exception as e:
        print(f"❌ Directory processing demonstration failed: {str(e)}")
        return None

# 错误处理演示，展示如何处理批量处理过程中可能出现的各种错误，并实现错误恢复机制
def demonstrate_error_handling():
    """Demonstrate error handling and recovery"""
    print("\n" + "=" * 60)
    print("ERROR HANDLING DEMONSTRATION")
    print("=" * 60)

    # 创建一个临时目录，并在其中创建一些具有不同问题的文件（如空文件、大文件、非存在文件等），以测试批量处理的错误处理能力
    temp_dir = Path(tempfile.mkdtemp())

    # Create files with various issues
    # 创建具有各种问题的文件
    files_with_issues = {
        "valid_file.txt": "This is a valid file that should process successfully.", # 正常的文件
        "empty_file.txt": "",  # 空文件
        "large_file.txt": "x" * 1000000,  # 大文件
    }

    # 遍历字典，把文件放到临时列表中，并写入磁盘
    created_files = []
    for filename, content in files_with_issues.items():
        file_path = temp_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        created_files.append(str(file_path))

    # Add a non-existent file to the list
    # 添加一个不存在的文件到列表中，以测试处理过程中对缺失文件的错误处理能力
    created_files.append(str(temp_dir / "non_existent_file.txt"))

    try:
        print(f"Testing error handling with {len(created_files)} files:")
        for file_path in created_files:
            name = Path(file_path).name # 获取文件名
            exists = Path(file_path).exists() # 检查文件是否存在
            size = Path(file_path).stat().st_size if exists else 0 # 获取文件大小，如果文件存在则获取，否则为0
            print(f"  - {name}: {'exists' if exists else 'missing'}, {size} bytes")

        # Create batch parser with short timeout for demonstration
        # 创建批量解析器实例，配置解析器类型、并行工作线程数、进度显示、短超时时间和跳过安装检查等参数，以测试错误处理能力
        batch_parser = BatchParser(
            parser_type="mineru",
            max_workers=2,
            show_progress=True,
            timeout_per_file=30,  # 设置较短的超时时间以触发处理大文件时的超时错误
            skip_installation_check=True,
        )

        # Process files and handle errors
        # 处理文件并捕获处理过程中可能出现的各种错误（如文件不存在、处理超时、解析错误等），并输出处理结果摘要、成功率以及失败文件的错误详情
        result = batch_parser.process_batch(
            file_paths=created_files,
            output_dir=str(temp_dir / "error_test_output"),
            parse_method="auto", # 解析方法为自动
        )

        print("\n" + "-" * 40)
        print("ERROR HANDLING RESULTS")
        print("-" * 40)
        print(result.summary())
        # 如果有成功处理的文件，打印成功文件列表
        if result.successful_files:
            print("\nSuccessful files:")
            for file_path in result.successful_files:
                print(f"  ✅ {Path(file_path).name}")
        # 如果有失败的文件，打印失败文件列表和对应的错误详情
        if result.failed_files:
            print("\nFailed files with error details:")
            for file_path in result.failed_files:
                error = result.errors.get(file_path, "Unknown error")
                print(f"  ❌ {Path(file_path).name}: {error}")

        # Demonstrate retry logic
        # 如果有失败的文件，展示如何使用批量处理的重试机制来重新处理这些失败的文件，并输出重试结果摘要和成功率等统计信息
        if result.failed_files:
            print(
                f"\nDemonstrating retry logic for {len(result.failed_files)} failed files..."
            )

            # Retry only the failed files
            # 仅重试失败的文件，使用相同的批量处理方法，并输出重试结果摘要和成功率等统计信息
            retry_result = batch_parser.process_batch(
                file_paths=result.failed_files,
                output_dir=str(temp_dir / "retry_output"),
                parse_method="auto",
            )

            print(f"Retry results: {retry_result.summary()}") # 打印重试结果摘要

        return result

    except Exception as e:
        print(f"❌ Error handling demonstration failed: {str(e)}")
        return None


async def main():
    """Main demonstration function"""
    """主函数，运行所有演示"""
    # Configure logging
    # 配置日志记录
    logging.basicConfig(
        level=logging.INFO, # 设置日志级别为INFO，即只输出INFO及以上级别的日志
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        # 日志输出格式模板，包含时间戳、日志记录器名称、日志级别和日志具体内容
    )

    print("RAG-Anything Batch Processing Demonstration") # 打印演示标题
    print("=" * 70)
    print("This example demonstrates various batch processing capabilities:") # 这个示例演示了各种批量处理功能
    print("  - Basic batch processing with progress tracking") # 基本批量处理，带有进度跟踪功能
    print("  - Asynchronous processing for improved performance") # 异步处理以提高性能
    print("  - Integration with RAG-Anything pipeline") # 与RAG-Anything管道的集成
    print("  - Directory processing with recursive file discovery") # 目录处理，带有递归文件发现功能
    print("  - Comprehensive error handling and recovery") # 综合错误处理和恢复机制

    results = {} # 用于存储每个演示的结果，以便在最后进行总结和比较

    # Run demonstrations
    print("\n🚀 Starting demonstrations...")

    # Basic batch processing
    # 运行基本批量处理演示
    results["basic"] = demonstrate_basic_batch_processing()

    # Asynchronous processing
    # 运行异步批量处理演示
    results["async"] = await demonstrate_async_batch_processing()

    # RAG integration
    # 运行RAG集成演示
    results["rag"] = await demonstrate_rag_integration()

    # Directory processing
    # 运行目录处理演示
    results["directory"] = demonstrate_directory_processing()

    # Error handling
    # 运行错误处理演示
    results["error_handling"] = demonstrate_error_handling()

    # Summary
    print("\n" + "=" * 70)
    print("DEMONSTRATION SUMMARY")
    print("=" * 70)

    for demo_name, result in results.items():
        if result:
            if hasattr(result, "success_rate"): # 检查result是否有success_rate属性，如果有则打印成功率，否则只打印完成信息  
                print(
                    f"✅ {demo_name.upper()}: {result.success_rate:.1f}% success rate"
                )
            else:
                print(f"✅ {demo_name.upper()}: Completed successfully")
        else:
            print(f"❌ {demo_name.upper()}: Failed or had limitations")

    print("\n📊 Key Features Demonstrated:")
    print("  - Parallel document processing with configurable worker counts")
    print("  - Real-time progress tracking with tqdm progress bars")
    print("  - Comprehensive error handling and reporting")
    print("  - File filtering based on supported document types")
    print("  - Directory processing with recursive file discovery")
    print("  - Asynchronous processing for improved performance")
    print("  - Integration with RAG-Anything document pipeline")
    print("  - Retry logic for failed documents")
    print("  - Detailed processing statistics and timing")

    print("\n💡 Best Practices Highlighted:")
    print("  - Use appropriate worker counts for your system")
    print("  - Enable progress tracking for long-running operations")
    print("  - Handle errors gracefully with retry mechanisms")
    print("  - Filter files to supported types before processing")
    print("  - Set reasonable timeouts for document processing")
    print("  - Use skip_installation_check for environments with conflicts")

# 如果这个文件是被直接运行的，那么执行main函数
if __name__ == "__main__":
    asyncio.run(main())
