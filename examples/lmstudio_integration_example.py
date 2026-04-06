"""
LM Studio Integration Example with RAG-Anything

This example demonstrates how to integrate LM Studio with RAG-Anything for local
text document processing and querying.

Requirements:
- LM Studio running locally with server enabled
- OpenAI Python package: pip install openai
- RAG-Anything installed: pip install raganything

Environment Setup:
Create a .env file with:
LLM_BINDING=lmstudio
LLM_MODEL=openai/gpt-oss-20b
LLM_BINDING_HOST=http://localhost:1234/v1
LLM_BINDING_API_KEY=lm-studio
EMBEDDING_BINDING=lmstudio
EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5
EMBEDDING_BINDING_HOST=http://localhost:1234/v1
EMBEDDING_BINDING_API_KEY=lm-studio
"""

"""
RAG-Anything 与 LM Studio 集成示例

这个示例演示了如何将 LM Studio 与 RAG-Anything 集成，以实现本地文本文档处理和查询。

要求：
- LM Studio 本地运行并启用服务器
- 安装 OpenAI Python 包：pip install openai
- 安装 RAG-Anything：pip install raganything

环境设置：
创建一个 .env 文件，内容如下：
LLM_BINDING=lmstudio
LLM_MODEL=openai/gpt-oss-20b
LLM_BINDING_HOST=http://localhost:1234/v1
LLM_BINDING_API_KEY=lm-studio
EMBEDDING_BINDING=lmstudio
EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5
EMBEDDING_BINDING_HOST=http://localhost:1234/v1
EMBEDDING_BINDING_API_KEY=lm-studio
"""

import os # 处理环境变量，特别是从 .env 文件加载配置
import uuid # 生成唯一的工作目录和文档 ID，避免不同运行之间的冲突和缓存问题
import asyncio # 处理异步函数，特别是在与 LM Studio 的 API 交互时使用
from typing import List, Dict, Optional # 类型提示，增强代码可读性和维护性
from dotenv import load_dotenv # 从 .env 文件加载环境变量，简化配置管理
from openai import AsyncOpenAI # LM Studio 的 API 客户端，提供异步方法与 LM Studio 服务器通信

# Load environment variables
# 加载环境变量，特别是与 LM Studio 连接相关的配置，如服务器地址、API 密钥和模型名称。这些变量在 .env 文件中定义，使用 load_dotenv() 函数加载到环境中，使其在代码中可访问
load_dotenv()

# RAG-Anything imports
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache

LM_BASE_URL = os.getenv("LLM_BINDING_HOST", "http://localhost:1234/v1")
LM_API_KEY = os.getenv("LLM_BINDING_API_KEY", "lm-studio")
LM_MODEL_NAME = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
LM_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5")

# 为LightRAG框架提供基于本地模型的文本生成能力
async def lmstudio_llm_model_func(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = None,
    **kwargs,
) -> str:
    """Top-level LLM function for LightRAG (pickle-safe)."""
    """LM Studio 的顶级 LLM 函数（pickle-safe）"""
    return await openai_complete_if_cache( # 调用 openai_complete_if_cache 函数，提供基于 LM Studio 的文本生成能力，同时支持缓存机制以提高性能和响应速度
        model=LM_MODEL_NAME,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        base_url=LM_BASE_URL,
        api_key=LM_API_KEY,
        **kwargs,
    )


async def lmstudio_embedding_async(texts: List[str]) -> List[List[float]]:
    """Top-level embedding function for LightRAG (pickle-safe)."""
    """LM Studio 的顶级嵌入函数（pickle-safe）"""
    from lightrag.llm.openai import openai_embed

    # 调用 openai_embed 函数，提供基于 LM Studio 的文本嵌入能力，返回嵌入向量列表
    embeddings = await openai_embed(
        texts=texts,
        model=LM_EMBED_MODEL,
        base_url=LM_BASE_URL,
        api_key=LM_API_KEY,
    )
    return embeddings.tolist() # 将嵌入结果转换为列表格式，确保与 LightRAG 的 EmbeddingFunc 兼容


class LMStudioRAGIntegration:
    """Integration class for LM Studio with RAG-Anything."""
    """LM Studio 与 RAG-Anything 的集成类"""
    
    def __init__(self):
        # LM Studio configuration using standard LLM_BINDING variables
        # 使用标准的 LLM_BINDING 变量配置 LM Studio 的连接参数，包括服务器地址、API 密钥和模型名称。这些参数从环境变量中读取，提供默认值以确保在没有 .env 文件时也能运行
        self.base_url = os.getenv("LLM_BINDING_HOST", "http://localhost:1234/v1")
        self.api_key = os.getenv("LLM_BINDING_API_KEY", "lm-studio")
        self.model_name = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
        self.embedding_model = os.getenv(
            "EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5"
        )

        # RAG-Anything configuration
        # Use a fresh working directory each run to avoid legacy doc_status schema conflicts
        # RAG-Anything 的配置，使用一个新的工作目录来避免与旧的 doc_status 模式冲突。配置包括解析器、解析方法以及是否启用图像、表格和方程处理等功能
        self.config = RAGAnythingConfig(
            working_dir=f"./rag_storage_lmstudio/{uuid.uuid4()}",
            parser="mineru",
            parse_method="auto",
            enable_image_processing=False,
            enable_table_processing=True,
            enable_equation_processing=True,
        )
        print(f"📁 Using working_dir: {self.config.working_dir}")

        self.rag = None

    # 测试 LM Studio 连接是否成功
    async def test_connection(self) -> bool:
        """Test LM Studio connection."""
        try:
            print(f"🔌 Testing LM Studio connection at: {self.base_url}")
            client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key) # 创建一个 AsyncOpenAI 客户端实例，使用配置的服务器地址和 API 密钥连接到 LM Studio 服务器
            models = await client.models.list() # 调用 models.list() 方法获取可用模型列表，以验证连接是否成功，并检查指定的模型是否存在
            print(f"✅ Connected successfully! Found {len(models.data)} models")

            # Show available models
            # 显示可用模型列表
            print("📊 Available models:")
            for i, model in enumerate(models.data[:5]):
                # 通过在当前使用的模型前添加一个标记（如 🎯）来突出显示指定的模型，帮助用户快速识别他们配置的模型是否在服务器上可用
                marker = "🎯" if model.id == self.model_name else "  "
                print(f"{marker} {i+1}. {model.id}")

            # 只显示前 5 个模型，如果模型数量超过 5 个，则提示用户还有更多模型可用
            if len(models.data) > 5:
                print(f"  ... and {len(models.data) - 5} more models")

            return True
        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
            print("\n💡 Troubleshooting tips:")
            print("1. Ensure LM Studio is running")
            print("2. Start the local server in LM Studio")
            print("3. Load a model or enable just-in-time loading")
            print(f"4. Verify server address: {self.base_url}")
            return False
        finally:
            try:
                await client.close()
            except Exception:
                pass

    # 测试 LM Studio 的基本聊天功能，验证文本生成是否正常工作
    async def test_chat_completion(self) -> bool:
        """Test basic chat functionality."""
        try:
            print(f"💬 Testing chat with model: {self.model_name}")
            client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key) # 创建一个 AsyncOpenAI 客户端实例，使用配置的服务器地址和 API 密钥连接到 LM Studio 服务器
            response = await client.chat.completions.create( # 调用 chat.completions.create() 方法发送一个简单的聊天请求，使用指定的模型和一组消息来测试文本生成能力
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {
                        "role": "user",
                        "content": "Hello! Please confirm you're working and tell me your capabilities.",
                    },
                ],
                max_tokens=100,
                temperature=0.7,
            )

            result = response.choices[0].message.content.strip() # 从响应中提取生成的文本结果，并去除前后空白字符
            print("✅ Chat test successful!")
            print(f"Response: {result}")
            return True
        except Exception as e:
            print(f"❌ Chat test failed: {str(e)}")
            return False
        finally: # 确保在测试完成后关闭客户端连接，释放资源
            try:
                await client.close()
            except Exception:
                pass

    # Deprecated factory helpers removed to reduce redundancy
    # 过时的工厂助手已被移除，以减少冗余

    # 创建一个完全可序列化的嵌入函数，返回一个 EmbeddingFunc 实例，配置了嵌入维度、最大 token 大小和实际的嵌入函数（lmstudio_embedding_async）
    def embedding_func_factory(self):
        """Create a completely serializable embedding function."""
        return EmbeddingFunc(
            embedding_dim=768,  # 嵌入维度，基于 nomic-embed-text-v1.5 模型的输出维度
            max_token_size=8192,  # 最大 token 大小，确保能够处理较长的文本输入
            func=lmstudio_embedding_async, # 实际的嵌入函数，使用基于 LM Studio 的异步嵌入函数来生成文本的嵌入向量
        )

    # 初始化 RAG-Anything，配置使用 LM Studio 的文本生成和嵌入功能，并处理与旧版本 LightRAG 兼容性相关的问题
    async def initialize_rag(self):
        """Initialize RAG-Anything with LM Studio functions."""
        print("Initializing RAG-Anything with LM Studio...")

        try:
            self.rag = RAGAnything( # 创建一个 RAGAnything 实例，配置使用 LM Studio 的文本生成函数和嵌入函数，同时传入 RAG-Anything 的配置参数
                config=self.config,
                llm_model_func=lmstudio_llm_model_func,
                embedding_func=self.embedding_func_factory(),
            )

            # Compatibility: avoid writing unknown field 'multimodal_processed' to LightRAG doc_status
            # Older LightRAG versions may not accept this extra field in DocProcessingStatus
            # 兼容性：避免将未知字段 'multimodal_processed' 写入 LightRAG 的 doc_status
            # 旧版本的 LightRAG 可能无法接受 DocProcessingStatus 中的这个额外字段
            async def _noop_mark_multimodal(doc_id: str): # 定义一个空操作的函数来替代 RAG-Anything 中标记多模态处理完成的方法，避免在旧版本的 LightRAG 中因为 doc_status 模式不兼容而导致错误
                return None

            self.rag._mark_multimodal_processing_complete = _noop_mark_multimodal

            print("✅ RAG-Anything initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ RAG initialization failed: {str(e)}")
            return False

    async def process_document_example(self, file_path: str):
        """Example: Process a document with LM Studio backend."""
        """示例：使用 LM Studio 后端处理文档"""
        if not self.rag: # 检查 RAG 是否已初始化，如果没有，则提示用户先调用 initialize_rag() 方法进行初始化
            print("❌ RAG not initialized. Call initialize_rag() first.")
            return

        try:
            print(f"📄 Processing document: {file_path}")
            await self.rag.process_document_complete( # 处理指定路径的文档
                file_path=file_path,
                output_dir="./output_lmstudio",
                parse_method="auto",
                display_stats=True,
            )
            print("✅ Document processing completed!")
        except Exception as e:
            print(f"❌ Document processing failed: {str(e)}")

    # 示例查询，展示如何使用不同的查询模式（hybrid、local、global）来查询处理过的文档内容
    async def query_examples(self):
        """Example queries with different modes."""
        if not self.rag:
            print("❌ RAG not initialized. Call initialize_rag() first.")
            return

        # Example queries
        queries = [
            ("What are the main topics in the processed documents?", "hybrid"),
            ("Summarize any tables or data found in the documents", "local"),
            ("What images or figures are mentioned?", "global"),
        ]

        print("\n🔍 Running example queries...")
        for query, mode in queries:
            try:
                print(f"\nQuery ({mode}): {query}")
                result = await self.rag.aquery(query, mode=mode) # 对不同的示例进行不同模式的查询
                print(f"Answer: {result[:200]}...") # 打印查询结果的前 200 个字符，提供一个简短的预览，避免输出过长的文本
            except Exception as e:
                print(f"❌ Query failed: {str(e)}")

    # 示例基本文本查询，使用一些预定义的内容来测试查询功能，验证 RAG-Anything 是否能够正确处理和查询文本内容
    async def simple_query_example(self):
        """Example basic text query with sample content."""
        if not self.rag:
            print("❌ RAG not initialized")
            return

        try:
            print("\nAdding sample content for testing...")

            # Create content list in the format expected by RAGAnything
            # 创建一个内容列表，格式符合 RAGAnything 的预期
            content_list = [
                {
                    "type": "text",
                    "text": """LM Studio Integration with RAG-Anything

This integration demonstrates how to connect LM Studio's local AI models with RAG-Anything's document processing capabilities. The system uses:

- LM Studio for local LLM inference
- nomic-embed-text-v1.5 for embeddings (768 dimensions)
- RAG-Anything for document processing and retrieval

Key benefits include:
- Privacy: All processing happens locally
- Performance: Direct API access to local models
- Flexibility: Support for various document formats
- Cost-effective: No external API usage""",
                    "page_idx": 0,
                }
            ]

            # Insert the content list using the correct method
            # 使用正确的方法插入内容列表
            await self.rag.insert_content_list(
                content_list=content_list,
                file_path="lmstudio_integration_demo.txt",
                # Use a unique doc_id to avoid collisions and doc_status reuse across runs
                # 使用一个唯一的 doc_id 来避免不同运行之间的冲突和 doc_status 的重用
                doc_id=f"demo-content-{uuid.uuid4()}",
                display_stats=True,
            )
            print("✅ Sample content added to knowledge base")

            print("\nTesting basic text query...")

            # Simple text query example
            result = await self.rag.aquery(
                "What are the key benefits of this LM Studio integration?",
                mode="hybrid",
            )
            print(f"✅ Query result: {result[:300]}...")

        except Exception as e:
            print(f"❌ Query failed: {str(e)}")

# 主函数，执行整个集成示例的流程，包括测试连接、测试聊天功能、初始化 RAG-Anything 以及运行示例查询
async def main():
    """Main example function."""
    print("=" * 70)
    print("LM Studio + RAG-Anything Integration Example")
    print("=" * 70)

    # Initialize integration
    # 初始化集成类，创建一个 LMStudioRAGIntegration 实例
    integration = LMStudioRAGIntegration()

    # Test connection
    # 测试连接
    if not await integration.test_connection():
        return False

    print()
    # 测试聊天功能
    if not await integration.test_chat_completion():
        return False

    # Initialize RAG
    # 初始化 RAG-Anything
    print("\n" + "─" * 50)
    if not await integration.initialize_rag():
        return False

    # Example document processing (uncomment and provide a real file path)
    # await integration.process_document_example("path/to/your/document.pdf")

    # Example queries (uncomment after processing documents)
    # await integration.query_examples()

    # Example basic query
    # 基础查询示例
    await integration.simple_query_example()

    print("\n" + "=" * 70)
    print("Integration example completed successfully!")
    print("=" * 70)

    return True


if __name__ == "__main__":
    print("🚀 Starting LM Studio integration example...")
    success = asyncio.run(main())

    exit(0 if success else 1)
