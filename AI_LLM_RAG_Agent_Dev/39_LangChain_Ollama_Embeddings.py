"""
LangChain 调用 Ollama 本地嵌入模型（Embeddings）示例

本示例演示如何在 LangChain 中使用 Ollama 本地的嵌入模型：
- 使用 `OllamaEmbeddings` 创建嵌入模型实例
- 调用 `embed_query()` 对单条文本生成向量
- 调用 `embed_documents()` 对多条文本批量生成向量

核心概念：
- Embedding（向量化）：将一段文本转换成一个浮点数列表（向量），
  使得「相似的文本」在向量空间中的距离更近，用于相似度搜索、向量数据库、RAG 检索等。
"""
import os
from typing import List

from langchain_ollama import OllamaEmbeddings


def init_embedding_model(model_name: str = "embeddinggemma:latest") -> OllamaEmbeddings:
    """
    初始化 OllamaEmbeddings 嵌入模型实例。

    Args:
        model_name: Ollama 本地模型名称，默认为 embeddinggemma:latest

    Returns:
        OllamaEmbeddings 实例
    """
    embed = OllamaEmbeddings(
        model="embeddinggemma:latest",
        base_url=os.getenv("EMBEDDING_BASE_URL"),  # Ollama 默认地址
    )
    return embed


def demo_embed_query(embed: OllamaEmbeddings) -> None:
    """
    演示 `embed_query`：对单条文本进行向量化。
    """

    print("【示例 1】embed_query：对单条文本生成向量")

    text = "我喜欢你"
    print(f"原始文本：{text}")

    vector: List[float] = embed.embed_query(text)

    print(f"\n向量维度：{len(vector)}")
    # 只展示前几个维度，避免输出过长
    preview_dims = 8
    print(f"前 {preview_dims} 维示例：{vector[:preview_dims]}")

    print()


def demo_embed_documents(embed: OllamaEmbeddings) -> None:
    """
    演示 `embed_documents`：对多条文本批量生成向量。
    """

    print("【示例 2】embed_documents：对多条文本批量生成向量")

    docs = ["我喜欢你", "我稀饭你", "晚上吃啥"]
    print("原始文本列表：")
    for i, d in enumerate(docs, 1):
        print(f"  {i}. {d}")

    vectors: List[List[float]] = embed.embed_documents(docs)

    print(f"\n共生成 {len(vectors)} 个向量，每个向量维度：{len(vectors[0]) if vectors else 0}")
    print("\n每条文本的向量前几维示例：")
    preview_dims = 6

    # 使用 zip(docs, vectors) 将「原始文本」和「对应的向量」一一配对，
    # 再用 enumerate(..., 1) 给每一对 (文本，向量) 编上从 1 开始的序号 i。
    # 这样在 for 循环里就可以同时拿到：序号 i、文本 d，以及该文本的向量 v。
    for i, (d, v) in enumerate(zip(docs, vectors), 1):
        print(f"  {i}. 文本：{d}")
        print(f"     向量前 {preview_dims} 维：{v[:preview_dims]}")

    print()


def intro_summary() -> None:
    """
    简要总结：什么时候使用嵌入模型？
    """

    print("【嵌入模型简介】")

    print()
    print("📌 嵌入模型（Embeddings）的典型应用场景：")
    print("- 相似度搜索：找到与查询文本语义最接近的文档")
    print("- 向量数据库：将文档向量化后存入 Milvus、Faiss、PGVector、Chroma 等")
    print("- RAG 检索：根据用户问题，在知识库中检索相关文档再交给大模型回答")
    print("- 文本聚类 / 降维可视化：基于语义相似性对文本分组")
    print()
    print("一般流程是：文本 -> 嵌入向量 -> 相似度计算 / 向量索引 -> 返回最相似结果。")
    print()


def main() -> None:
    """
    主函数：演示 Ollama 嵌入模型在 LangChain 中的基本用法。
    """

    print("LangChain 调用 Ollama 本地嵌入模型（OllamaEmbeddings）示例")
    print("使用模型：embeddinggemma:latest")

    print()

    embed = init_embedding_model()

    # 示例 1：单条文本向量化
    demo_embed_query(embed)

    # 示例 2：多条文本批量向量化
    demo_embed_documents(embed)

    # 嵌入模型使用说明
    intro_summary()

    print("示例结束")


if __name__ == "__main__":
    main()
