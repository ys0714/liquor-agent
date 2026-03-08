"""
LangChain 消息简写形式示例

本示例演示 LangChain 中消息定义的两种方式：
1. 标准格式：使用 SystemMessage、HumanMessage、AIMessage 对象
2. 简写格式：使用 2 元组 (role, content) 的形式

核心概念：
- 标准格式：显式使用消息类，类型安全，代码清晰
- 简写格式：使用元组，代码更简洁，但需要手动指定角色字符串
- 两种格式在功能上完全等价，LangChain 会自动转换

优缺点对比：
标准格式优点：
- 类型安全，IDE 可以提供更好的代码补全和类型检查
- 代码可读性强，一眼就能看出消息类型
- 支持更多高级功能（如消息元数据、工具调用等）

标准格式缺点：
- 代码相对冗长，需要导入多个类
- 对于简单场景可能显得过于正式

简写格式优点：
- 代码简洁，减少导入和类名
- 适合快速原型开发和简单场景
- 消息列表更紧凑，易于阅读

简写格式缺点：
- 类型安全性较差，字符串拼写错误不会在编译时发现
- IDE 支持较弱，缺少代码补全
- 不支持消息的高级属性（如 name、tool_calls 等）
"""

import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def init_chat_model() -> ChatOllama:
    """
    初始化 ChatOllama 聊天模型实例。

    优先从以下环境变量中读取密钥（依次回退）：
    - DASHSCOPE_API_KEY（阿里云官方推荐）
    - API_KEY（与本项目其他示例保持兼容）

    注意：使用 qwen3-max，这是聊天模型，适合对话场景
    """
    load_dotenv()

    # 兼容两种环境变量命名方式
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise ValueError(
            "未找到 DASHSCOPE_API_KEY 或 API_KEY 环境变量，请先在 .env 或系统环境中配置后再运行。"
        )

    # LangChain 的 ChatOllama 封装会自动从环境变量中读取 key，
    # 这里设置一份到 DASHSCOPE_API_KEY，确保兼容性。
    os.environ["DASHSCOPE_API_KEY"] = api_key

    # 使用 qwen3-max 聊天模型
    chat = ChatOllama(model=os.getenv("MODEL"))
    return chat


def standard_format_demo(chat: ChatOllama) -> None:
    """
    演示标准格式：使用 SystemMessage、HumanMessage、AIMessage 对象。

    这是 LangChain 推荐的标准方式，提供了更好的类型安全和代码可读性。
    """
    
    print("【示例1】标准格式：使用 SystemMessage、HumanMessage、AIMessage 对象")
    

    # 准备消息列表（标准格式）
    messages = [
        SystemMessage(content="你是一个边塞诗人。"),
        HumanMessage(content="写一首唐诗"),
        AIMessage(content="锄禾日当午,汗滴禾下土,谁知盘中餐,粒粒皆辛苦。"),
        HumanMessage(content="按照你上一个回复的格式,再写一首唐诗。"),
    ]

    print("消息列表（标准格式）：")
    for i, msg in enumerate(messages, 1):
        if isinstance(msg, SystemMessage):
            print(f"  {i}. [系统] {msg.content}")
        elif isinstance(msg, HumanMessage):
            print(f"  {i}. [用户] {msg.content}")
        elif isinstance(msg, AIMessage):
            print(f"  {i}. [AI] {msg.content}")

    print("\n模型回复（流式输出）：")
    

    # for 循环迭代打印输出，通过 .content 来获取到内容
    for chunk in chat.stream(input=messages):
        print(chunk.content, end="", flush=True)

    print("\n")
    
    print()


def shorthand_format_demo(chat: ChatOllama) -> None:
    """
    演示简写格式：使用 2 元组 (role, content) 的形式。

    通过 2 元组封装信息：
    - 第一个元素为角色：字符串 "system" / "human" / "ai"
    - 第二个元素为内容：消息的文本内容

    这种方式代码更简洁，但类型安全性较差。
    """
    
    print("【示例2】简写格式：使用 2 元组 (role, content)")
    

    # 准备消息列表（简写格式）
    messages = [
        ("system", "你是一个边塞诗人。"),
        ("human", "写一首唐诗"),
        ("ai", "锄禾日当午,汗滴禾下土,谁知盘中餐,粒粒皆辛苦。"),
        ("human", "按照你上一个回复的格式,再写一首唐诗。"),
    ]

    print("消息列表（简写格式）：")
    for i, (role, content) in enumerate(messages, 1):
        role_map = {"system": "系统", "human": "用户", "ai": "AI"}
        print(f"  {i}. [{role_map.get(role, role)}] {content}")

    print("\n模型回复（流式输出）：")
    

    # for 循环迭代打印输出，通过 .content 来获取到内容
    for chunk in chat.stream(input=messages):
        print(chunk.content, end="", flush=True)

    print("\n")
    
    print()


def comparison_demo(chat: ChatOllama) -> None:
    """
    对比演示：展示两种格式的等价性。

    两种格式在功能上完全等价，LangChain 会自动将元组格式转换为对应的消息对象。
    """
    
    print("【示例3】对比演示：两种格式的等价性")
    

    # 标准格式
    messages_standard = [
        SystemMessage(content="你是一个专业的 Python 编程助手。"),
        HumanMessage(content="请用 Python 写一个简单函数。"),
    ]

    # 简写格式
    messages_shorthand = [
        ("system", "你是一个专业的 Python 编程助手。"),
        ("human", "请用 Python 写一个简单函数。"),
    ]

    print("--- 使用标准格式 ---")
    print("消息列表：")
    for i, msg in enumerate(messages_standard, 1):
        if isinstance(msg, SystemMessage):
            print(f"  {i}. [系统] {msg.content}")
        elif isinstance(msg, HumanMessage):
            print(f"  {i}. [用户] {msg.content}")

    print("\n模型回复：")
    print("-" * 40)
    for chunk in chat.stream(input=messages_standard):
        print(chunk.content, end="", flush=True)
    print("\n")

    print("\n--- 使用简写格式 ---")
    print("消息列表：")
    for i, (role, content) in enumerate(messages_shorthand, 1):
        role_map = {"system": "系统", "human": "用户", "ai": "AI"}
        print(f"  {i}. [{role_map.get(role, role)}] {content}")

    print("\n模型回复：")
    print("-" * 40)
    for chunk in chat.stream(input=messages_shorthand):
        print(chunk.content, end="", flush=True)
    print("\n")

    
    print()


def advantages_and_disadvantages() -> None:
    """
    详细阐述两种格式的优缺点。
    """
    
    print("【优缺点对比】")
    
    print()

    print("📌 标准格式（SystemMessage、HumanMessage、AIMessage）")
    
    print("✅ 优点：")
    print("  1. 类型安全：IDE 可以提供代码补全和类型检查")
    print("  2. 代码可读性强：一眼就能看出消息类型")
    print("  3. 支持高级功能：可以设置消息的 name、tool_calls 等属性")
    print("  4. 错误检测：如果使用了错误的类，会在运行时立即发现")
    print("  5. 更好的 IDE 支持：自动补全、类型提示等")
    print()
    print("❌ 缺点：")
    print("  1. 代码相对冗长：需要导入多个类")
    print("  2. 对于简单场景可能显得过于正式")
    print("  3. 需要记住不同的类名")
    print()

    print("📌 简写格式（元组 (role, content)）")
    
    print("✅ 优点：")
    print("  1. 代码简洁：减少导入和类名，代码更紧凑")
    print("  2. 适合快速原型：快速编写和测试代码")
    print("  3. 消息列表更易读：结构清晰，一目了然")
    print("  4. 减少代码量：对于简单场景，代码更少")
    print()
    print("❌ 缺点：")
    print("  1. 类型安全性差：字符串拼写错误不会在编译时发现")
    print("  2. IDE 支持较弱：缺少代码补全和类型提示")
    print("  3. 不支持高级属性：无法设置消息的 name、tool_calls 等")
    print("  4. 运行时错误：角色字符串错误只能在运行时发现")
    print("  5. 可维护性较差：字符串硬编码，重构困难")
    print()

    print("💡 使用建议：")
    
    print("  • 生产环境：推荐使用标准格式，保证代码质量和可维护性")
    print("  • 快速原型：可以使用简写格式，提高开发效率")
    print("  • 复杂场景：必须使用标准格式，以支持高级功能")
    print("  • 团队协作：建议统一使用标准格式，保持代码风格一致")
    print()

    
    print()


def main() -> None:
    """
    主函数：演示 LangChain 消息的两种定义方式及其优缺点。
    """
    
    print("LangChain 消息简写形式示例")
    
    print()

    chat = init_chat_model()

    # 示例1：标准格式
    standard_format_demo(chat)

    # 示例2：简写格式
    shorthand_format_demo(chat)

    # 示例3：对比演示
    comparison_demo(chat)

    # 优缺点分析
    advantages_and_disadvantages()

    
    print("演示结束")
    


if __name__ == "__main__":
    main()
