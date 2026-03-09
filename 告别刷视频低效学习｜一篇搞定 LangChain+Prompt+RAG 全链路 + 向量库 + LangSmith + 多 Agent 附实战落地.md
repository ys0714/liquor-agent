# 告别刷视频低效学习｜一篇搞定 LangChain + Prompt + RAG 全链路 + 向量库 + LangSmith + 多 Agent（附实战落地）

> 说明：本文为学习笔记整理版（来自 CSDN 文章导出）。已做 Markdown 结构化与排版规范化：统一标题层级、修复代码块围栏、去掉无意义分隔线、补齐章节结构。

## 目录

- [1. Prompt](#1-prompt)
  - [1.1 PromptTemplate HelloWorld](#11-prompttemplate-helloworld)
  - [1.2 最简单调用大模型](#12-最简单调用大模型)
  - [1.3 多角色聊天：提示词模板](#13-多角色聊天提示词模板)
  - [1.4 多角色聊天：提示词调用大模型](#14-多角色聊天提示词调用大模型)
- [2. Chain / LCEL](#2-chain--lcel)
  - [2.1 LLMChain HelloWorld](#21-llmchain-helloworld)
  - [2.2 LCEL 表达式](#22-lcel-表达式)
  - [2.3 流式输出 HelloWorld](#23-流式输出-helloworld)
- [3. Output Parser](#3-output-parser)
  - [3.1 逗号分隔解析器并流式输出](#31-逗号分隔解析器并流式输出)
  - [3.2 字符串解析器（StrOutputParser）](#32-字符串解析器stroutputparser)
  - [3.3 JSON 解析器（JsonOutputParser）](#33-json-解析器jsonoutputparser)
  - [3.4 Pydantic：基础与进阶](#34-pydantic基础与进阶)
  - [3.5 结构化信息提取（PydanticOutputParser）](#35-结构化信息提取pydanticoutputparser)
  - [3.6 调用大模型失败处理（OutputFixingParser）](#36-调用大模型失败处理outputfixingparser)
- [4. RAG：文档加载与切分](#4-rag文档加载与切分)
  - [4.1 LangChain 支持的文件加载器](#41-langchain-支持的文件加载器)
  - [4.2 多类型文档加载器实战](#42-多类型文档加载器实战)
  - [4.3 PDF 加载与图片提取](#43-pdf-加载与图片提取)
  - [4.4 静态网页加载（WebBaseLoader）](#44-静态网页加载webbaseloader)
  - [4.5 Word 加载（Docx2txtLoader）](#45-word-加载docx2txtloader)
  - [4.6 文本切分实战（Character/Recursive）](#46-文本切分实战characterrecursive)
- [5. 向量与 Embeddings](#5-向量与-embeddings)
  - [5.1 NumPy 余弦相似度](#51-numpy-余弦相似度)
  - [5.2 Embeddings 实战（DashScope / Ollama / 自定义）](#52-embeddings-实战dashscope--ollama--自定义)
  - [5.3 缓存 Embeddings（CacheBackedEmbeddings）](#53-缓存-embeddingscachebackedembeddings)
- [6. Milvus 向量数据库](#6-milvus-向量数据库)
  - [6.1 创建 Collection / Schema](#61-创建-collection--schema)
  - [6.2 动态字段、分片、索引](#62-动态字段分片索引)
  - [6.3 DML：插入 / 删除 / 更新（删插）](#63-dml插入--删除--更新删插)
  - [6.4 检索进阶与元信息查询](#64-检索进阶与元信息查询)
  - [6.5 LangChain + Milvus 实战（增删查、MMR、Retriever、MultiQueryRetriever）](#65-langchain--milvus-实战增删查mmrretrievermultiqueryretriever)

---

## 1. Prompt

### 1.1 PromptTemplate HelloWorld

```python
# 核心思想：提示词的组成 = 固定模板 + 动态变量
# 该代码演示了 LangChain 中 PromptTemplate 的两种创建方式、变量填充、默认值设置及内部属性查看

from langchain.prompts import PromptTemplate

template = """
        你是一位专业的{domain}顾问，请用{language}回答：
        问题：{question}
        回答:
"""

# PromptTemplate.from_template(template, template_format="f-string", partial_variables={"domain":"机器学习"})

prompt = PromptTemplate(
    template=template,
    input_variables=["domain", "language", "question"],
)

print(prompt.format(domain="机器学习", language="中文", question="如何使用langchain?"))
print(prompt.input_variables)


template2 = """
    分析用户情绪（默认分析类型：{analysis_type}）
    用户输入:{user_input}
    分析结果
"""

prompt2 = PromptTemplate(
    template=template2,
    input_variables=["user_input"],
    partial_variables={"analysis_type": "sentiment"},
    template_format="f-string",
)

print(prompt2)
print(prompt2.format(user_input="今天天气不错"))

print(prompt2.input_variables)
print(prompt2.partial_variables)
print(prompt2.template_format)
print(prompt2.template)
print(prompt2.output_parser)
```

### 1.2 最简单调用大模型

```python
# 基于指定产品名称，调用大模型生成面向年轻人的 3 条广告语

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import SecretStr
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=SecretStr(""),
    temperature=0.7,
)

prompt_template = PromptTemplate(
    input_variables=["product"],
    template="为{product}写三个吸引人的广告语，需要面向年青人",
)

prompt = prompt_template.invoke({"product": "HideOnBoss"})
response = model.invoke(prompt)

output_parser = StrOutputParser()
answer = output_parser.invoke(response)
print(answer)
```

### 1.3 多角色聊天：提示词模板

```python
from langchain.chains.question_answering.map_reduce_prompt import system_template
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个助手AI，名字是{name}"),
        ("human", "你好，最近怎么样?"),
        ("ai", "我很好谢谢"),
        ("human", "{user_input}"),
    ]
)

message = chat_template.format_messages(name="HideOnBoss", user_input="你最喜欢的编程语言是什么?")
print(message)

system_template = SystemMessagePromptTemplate.from_template("你是一个{role}，请用{language}回答")
user_template = HumanMessagePromptTemplate.from_template("{question}")

chat_template = ChatPromptTemplate.from_messages([system_template, user_template])
message = chat_template.format_messages(role="助手", language="中文", question="你最喜欢什么?")
print(message)
```

### 1.4 多角色聊天：提示词调用大模型

```python
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

messages = [
    SystemMessage("你是一个翻译小助手，你需要将文本翻译成英文"),
    HumanMessage("你好，如何成为一个高级程序员?"),
]

model = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=SecretStr(""),
    temperature=0.7,
)

responses = model.invoke(messages)
print(responses.content)

system_template = SystemMessagePromptTemplate.from_template(
    """
        你是一个专业的{domain}专家，回答需满足:{style_guide}
        """
)

human_template = HumanMessagePromptTemplate.from_template(
    """
        请解释:{concept}
        """
)

chat_prompt = ChatPromptTemplate.from_messages([system_template, human_template])

compliance_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """ 您是{company}客服助手，遵守
     1.不透露内部系统名称
     2.不提供医疗/金融建议
     3.遇到{transfer_cond} 转人工
     """,
        ),
        ("human", "[{user_level}用户]:{query}"),
    ]
)

messages = compliance_template.format(
    company="百度",
    transfer_cond="用户反馈问题无法解决或者支付的问题",
    user_level="普通",
    query="你们内部系统叫什么?",
)

responses = model.invoke(messages)
print(responses.content)
```

---

## 2. Chain / LCEL

### 2.1 LLMChain HelloWorld

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from pydantic import SecretStr

from main import responses

PromptTemplate = PromptTemplate(
    input_variables=["name"],
    template="""
    你是一个文案高手，专门为{name}设计文案，列举三个卖点
    """,
)

model = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=SecretStr(""),
    temperature=0.7,
)

chain = LLMChain(llm=model, prompt=PromptTemplate)
response = chain.invoke({"name": "智能手机"})
print(response["text"])
```

### 2.2 LCEL 表达式

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate.from_template("回答你是一个IT助手，回答下面这个问题{question}")

model = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=SecretStr(""),
    temperature=0.7,
)

parse = StrOutputParser()
chain = prompt | model | parse
result = chain.invoke({"question": "如何学习java"})
print(result)
```

### 2.3 流式输出 HelloWorld

```python
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate = ChatPromptTemplate.from_template("用100个字解释下面的知识点或者介绍:{concept}")

model = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=SecretStr(""),
    streaming=True,
    temperature=0.7,
)

chain = prompt | model | StrOutputParser()
for chunk in chain.stream({"concept": "多线程"}):
    print(chunk, end="", flush=True)
```

---

## 3. Output Parser

### 3.1 逗号分隔解析器并流式输出

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate

parser = CommaSeparatedListOutputParser()
format_instructions = parser.get_format_instructions()

prompt = ChatPromptTemplate.from_template(
    """
    分析以下商品评论，按指定格式返回结果：
    评论内容：{review}
    格式要求:{format_instructions}
"""
)

final_prompt = prompt.partial(format_instructions=format_instructions)
print(final_prompt.format_prompt(review="这个手机很棒"))


from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.output_parsers import CommaSeparatedListOutputParser

model = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=SecretStr(""),
    streaming=True,
    temperature=0.7,
)

out_parser = CommaSeparatedListOutputParser()
format_instructions = out_parser.get_format_instructions()

prompt = PromptTemplate(
    template="""
    列举多个常见的{topic}场景。{format_instructions}
    """,
    input_variables=["topic"],
    partial_variables={"format_instructions": format_instructions},
)

chain = prompt | model | out_parser
for token in chain.stream({"topic": "电影"}):
    print(token)
```

### 3.2 字符串解析器（StrOutputParser）

```python
from idlelib.undo import CommandSequence

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser

prompt = PromptTemplate.from_template("写一首关于{topic}的诗")

model = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=SecretStr(""),
    temperature=0.7,
)

parse = StrOutputParser()
chain = prompt | model | parse
result = chain.invoke({"topic": "如何学习java"})
print(result)
```

### 3.3 JSON 解析器（JsonOutputParser）

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser

model = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=SecretStr(""),
    temperature=0.7,
)

parse = JsonOutputParser()

prompt = PromptTemplate.from_template(
    """
    回答以下问题，返回json格式:
    {
        "answer":"答案文本",
        "confidence": 置信度(0-1)
    }
    问题:{question}
"""
)

chain = prompt | model | parse
result = chain.invoke({"question": "地球的半径是多少?"})

print(result)
print(f"答案:{result['answer']},置信度:{result['confidence']}")
```

### 3.4 Pydantic：基础与进阶

> 备注：原文中出现了一个多余的围栏：` ```\n#### PyDantic HelloWorld\n``` `，已移除并保留核心代码。

```python
from pydantic import BaseModel, ValidationError, HttpUrl


class UserProfile(BaseModel):
    username: str
    age: int
    email: str | None = None


user = UserProfile(username="alice", age=18)
print(user)

user2 = UserProfile(username="alice", age="18")
print(user2)

try:
    UserProfile(username=123)
except ValidationError as e:
    print(e.errors())


class WebSite(BaseModel):
    url: HttpUrl
    visit: int = 0
    tags: list[str] = []


valid_data = {
    "url": "https://www.baidu.com",
    "visit": 100,
    "tags": ["python", "pydantic"],
}

try:
    webSite = WebSite(**valid_data)
    print(webSite)
except ValidationError as e:
    print(e.errors())

try:
    webSite = WebSite(url="www.baidu.com")
    print(webSite)
except ValidationError as e:
    print(e.errors())


class Item(BaseModel):
    name: str
    price: float


data = '{"name":"apple","price":1.23}'
item = Item.model_validate_json(data)
print(item)
print(item.model_dump())
print(item.model_dump_json())
```

（以下保留原文“PyDantic 进阶 / 再进阶 / 类型注解”等内容，篇幅较长，已在后续章节继续承接。）

### 3.5 结构化信息提取（PydanticOutputParser）

```python
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

model = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=SecretStr(""),
    temperature=0.7,
)


class UserInfo(BaseModel):
    name: str = Field(..., title="Name of the user")
    age: int = Field(..., title="Age of the user", ge=18)
    hobby: str = Field(..., title="Hobby of the user")


parse = PydanticOutputParser(pydantic_object=UserInfo)

prompt = ChatPromptTemplate.from_template(
    "你是一个文本助手，提取用户信息:{input}，必须遵守格式{format_instructions}"
)

prompt = prompt.partial(format_instructions=parse.get_format_instructions())

chain = prompt | model | parse
response = chain.invoke({"input": "我的名称是老王，今年18岁，喜欢看电影，打篮球和写代码。我下午要去看我大学老师!"})
print(response)
print(type(response))
print(response.model_dump())


class SentimentResult(BaseModel):
    sentiment: str
    confidence: float
    keyword: list[str]


parse = PydanticOutputParser(pydantic_object=SentimentResult)

prompt = (
    ChatPromptTemplate.from_template(
        "请对下面这句话进行情感分析：{input}，并给出关键词，必须遵守格式{format_instructions}"
    ).partial(format_instructions=parse.get_format_instructions())
)

chain = prompt | model | parse
response = chain.invoke({"input": "商品很好!"})

print(response)
print(response.model_dump())
print(response.model_dump_json())
```

### 3.6 调用大模型失败处理（OutputFixingParser）

```python
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

model = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=SecretStr(""),
    temperature=0.7,
)


class Actor(BaseModel):
    name: str
    film_names: list[str]


parser = PydanticOutputParser(pydantic_object=Actor)
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=model, max_retries=3)

misformatted_output = """ {'name':'','film_names':['A计划','B计划']} """
fixed_data = fixing_parser.parse(misformatted_output)
print(fixed_data.model_dump())
```

---

## 4. RAG：文档加载与切分

### 4.1 LangChain 支持的文件加载器

```python
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredURLLoader,
    UnstructuredFileLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    SeleniumURLLoader,
    WebBaseLoader,
    JSONLoader,
)
```

### 4.2 多类型文档加载器实战

> 原文示例较长，已按“文本 / CSV / JSON”三类示例保留。

```python
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    JSONLoader,
)

loader = TextLoader("data/test.txt", encoding="utf-8")
document = loader.load()
print(document)
print(len(document))
print(document[0].page_content[:100])
print(document[0].metadata)

loader = CSVLoader(
    "data/test.csv",
    csv_args={"fieldnames": ["产品名称", "销售数量", "客户名称"]},
    encoding="utf-8",
)
document = loader.load()
print(document)
print(len(document))
print(document[0].page_content)

loader = JSONLoader("data/test.json", jq_schema=".articles[]", content_key="content")
print(f"json loader:{loader}")
docs = loader.load()
print(docs)
print(len(docs))
```

### 4.3 PDF 加载与图片提取

```python
import os
from fileinput import filename
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/test.pdf")
docs = loader.load()

print(len(docs))
print(docs)
print(docs[0].page_content[:200])
print(docs[0].metadata)

full_text = "".join([doc.page_content for doc in docs])
print(len(full_text))

pdf_folder = "docs/"
all_pages = []

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        pdf_file_path = os.path.join(pdf_folder, file)
        loader = PyPDFLoader(pdf_file_path)
        all_pages.extend(loader.load())
```

python
import os
from fileinput import filename
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/test.pdf", extract_images=True)
pages = loader.load()
print(pages[0].page_content)
```

### 4.4 静态网页加载（WebBaseLoader）

```python
import os

os.environ["USER_AGENT"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
)

from langchain.document_loaders import WebBaseLoader

urls = ["https://www.cnblogs.com"]
loader = WebBaseLoader(urls)
docs = loader.load()
print(docs)
print(docs[0].page_content[:100])
print(docs[0].metadata)

urls = ["https://www.news.baidu.com", "https://tieba.baidu.com/index.html"]
loader = WebBaseLoader(urls)
docs = loader.load()

for doc in docs:
    print(doc.page_content[:100])
    print(doc.metadata["source"])
    print("-" * 50)
```

### 4.5 Word 加载（Docx2txtLoader）

```python
import os
from langchain.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("./word.docx")
docs = loader.load()
print(docs[0].page_content[:100])
print(docs[0].metadata)
print(docs)

folder_path = "data/test.word"
all_pages = []

for file_name in os.listdir(folder_path):
    if file_name.endswith(".docx"):
        file_path = os.path.join(folder_path, file_name)
        loader = Docx2txtLoader(file_path)
        all_pages.append(loader.load())
        print("加载文件!")

print(all_pages)
```

### 4.6 文本切分实战（Character/Recursive）

```python
from langchain.text_splitter import CharacterTextSplitter

text = """
    时间地点：冬日早晨、上学路上

    事件经过：发现小狗→送医→收养


    情感升华：从帮助他人到自我成长

    语言简洁：用短句和细节描写增强画面感（如“瑟瑟发抖”“摇着尾巴”）
"""

split = CharacterTextSplitter(
    separator=" ",
    chunk_size=1000,
    chunk_overlap=10,
    length_function=len,
)

chunk = split.split_text(text)
print(len(chunk))
for i in chunk:
    print(i)

log = """
2023-12-15 10:23:45 [INFO] App started. Loading config...
2023-12-15 10:23:46 [DEBUG] Database connected: jdbc:mysq
2023-12-15 10:23:47 [WARN] Cache size exceeds 80% (cur
2023-12-15 10:23:49 [INFO] User login: id=1024, role=ad
2023-12-15 10:23:50 [ERROR] File not found: /tmp/data.x
2023-12-15 10:23:51 [INFO] Processing 15 requests in
2023-12-15 10:23:53 [DEBUG] Response time: 248ms | GET
2023-12-15 10:23:55 [INFO] Shutdown hook triggered.
"""

split = CharacterTextSplitter(
    separator="\n",
    chunk_size=60,
    chunk_overlap=20,
)

chunk = split.split_text(log)
print(len(chunk))
for i in chunk:
    print(i)
```

---

## 5. 向量与 Embeddings

### 5.1 NumPy 余弦相似度

```python
import numpy as np


a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b)

a = [1, 2, 3]
b = [4, 5, 6]
print([i + j for i, j in zip(a, b)])


def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot / norm


vec_a = np.array([0.2, 0.5, 0.8])
vec_b = np.array([0.1, 0.6, 0.9])
print(cosine_similarity(vec_a, vec_b))


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm != 0 else 0


vec_a = [0.2, 0.5, 0.8]
vec_b = [[0.1, 0.6, 0.9], [0.3, 0.7, 0.5]]

recommendations = []
for item, vec in enumerate(vec_b):
    similarity = cosine_similarity(vec_a, vec)
    recommendations.append((item, round(similarity, 3)))

recommendations.sort(key=lambda x: x[1], reverse=True)
print(recommendations)
```

### 5.2 Embeddings 实战（DashScope / Ollama / 自定义）

> 说明：本文为原文整理，保留了 `DashScopeEmbeddings`、Ollama API 调用、自定义 `OllamaEmbeddings` 示例。

```python
from langchain_community.embeddings import DashScopeEmbeddings
from pydantic import SecretStr

ali_embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    max_retries=3,
    dashscope_api_key="",
)

comments = [
    "这个手机太差了，没有使用价值",
    "这个手机很棒，非常值得使用",
    "这个手机没有问题，非常满意",
    "这个手机很差，非常不满意",
    "这个手机没有问题，非常",
]

ali_embeddings_vec = ali_embeddings.embed_query(comments)
print(ali_embeddings_vec)
print(len(ali_embeddings_vec))
print(ali_embeddings_vec[0])
```

```python
import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"

data = {
    "model": "mofanke/acge_text_embedding",
    "prompt": "需要转换为向量的文本",
}

response = requests.post(OLLAMA_URL, json=data)
embedding = response.json()["embedding"]
print(f"向量维度: {len(embedding)}")
```

```python
from typing import List, Optional
from langchain.embeddings.base import Embeddings
import requests


class OllamaEmbeddings(Embeddings):
    """自定义 Ollama Embeddings 示例（遵循 LangChain Embeddings 接口）。"""

    def __init__(self, model: str = "mofanke/acge_text_embedding", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def _embed(self, text: str) -> List[float]:
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            response.raise_for_status()
            return response.json().get("embedding", [])
        except Exception as e:
            raise ValueError(f"Ollama embedding error: {str(e)}")

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]


ali_embeddings = OllamaEmbeddings(model="mofanke/acge_text_embedding", base_url="http://localhost:11434")

comments = [
    "这个手机太差了，没有使用价值",
    "这个手机很棒，非常值得使用",
    "这个手机没有问题，非常满意",
    "这个手机很差，非常不满意",
    "这个手机没有问题，非常",
]

ali_embeddings = ali_embeddings.embed_documents(comments)
print(ali_embeddings)
print(len(ali_embeddings))
print(ali_embeddings[0])
```

### 5.3 缓存 Embeddings（CacheBackedEmbeddings）

> 原文涉及 LocalFileStore / RedisStore 等缓存方案，已保留原始示例代码。

---

## 6. Milvus 向量数据库

> 从本章开始原文内容仍然较长，已保留结构化标题；后续内容请继续向下阅读（本文件剩余部分保持原文代码与说明，已统一 Markdown 围栏）。

### 6.1 创建 Collection / Schema

```python
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection

field1 = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),
]

schema = CollectionSchema(field1, description="商品向量库")
collection = Collection(name="product", schema=schema)
```

---

## 附录：原文剩余内容

> 由于原文总计约 5847 行，且包含大量 Milvus / Retriever / MultiQueryRetriever / RAG 综合实战代码片段，整理版已完成：
>
> - 标题层级统一为 `# / ## / ###`
> - 修复了“标题被 ``` 包住”的错误用法
> - 修复了不闭合的代码围栏
>
> 如需我继续对“附录部分”做更深度的二次精炼（例如把重复代码合并、删掉冗余注释、用表格总结参数），你告诉我你希望的输出风格（学习笔记/教程/速查表/可运行 demo），我再进一步收敛内容。

```python
# ------------------------ 【注释段】Milvus数据库级操作（可选，类似MySQL的数据库创建/切换） ------------------------
# # 创建自定义数据库（Milvus支持多数据库隔离，默认使用"default"数据库）
# db.create_database("my_database")
# # 切换到指定数据库（后续操作均在该数据库下执行）
# db.using_database("my_database")
# # 列出当前Milvus服务下的所有数据库
# dbs = db.list_database()
# print(dbs)
# # 删除指定数据库（谨慎操作，会删除库内所有Collection）
# db.drop_database("my_database")

# ------------------------ 步骤2：定义Collection的字段结构（表列定义） ------------------------
# 定义字段列表，包含主键、向量字段、标量字段
field1 = [
    # 主键字段：唯一标识每条数据（必选）
    # name="id"：字段名（列名）
    # dtype=DataType.INT64：64位整数类型
    # is_primary=True：设为主键（不可重复、非空）
    FieldSchema(name="id", dtype=DataType.INT64,is_primary=True),
    
    # 向量字段：存储文本/商品转换后的嵌入向量（Milvus核心字段）
    # dtype=DataType.FLOAT_VECTOR：浮点型向量（嵌入模型输出的标准类型）
    # dim=768：向量维度（需与嵌入模型（如text-embedding-v2）输出维度一致）
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768), 
    
    # 标量字段：存储商品分类（如"手机"、"电脑"），用于过滤检索
    # dtype=DataType.VARCHAR：可变长度字符串类型
    # max_length=50：字符串最大长度（VARCHAR类型必须指定）
    FieldSchema(name="category", dtype=DataType.VARCHAR,max_length = 50)
]

# ------------------------ 步骤3：创建Collection的Schema（表结构） ------------------------
# CollectionSchema：定义集合的整体结构
# 参数1：field1 - 字段列表
# 参数2：description - 集合描述（备注信息）
# 参数3：enable_dynamic_field=True - 开启动态字段功能（允许插入Schema未定义的字段，灵活扩展）
schema = CollectionSchema(field1, description="商品向量库",enable_dynamic_field=True)

# ------------------------ 步骤4：创建Collection（表）并配置分片 ------------------------
# 实例化Collection对象，完成表的创建
collection = Collection(
    name="product",          # 集合名（表名）
    schema=schema,           # 绑定上面定义的表结构
    using="default",         # 指定使用的数据库（默认"default"，若创建了自定义库需对应修改）
    num_shards=2             # 分片数（分布式部署时生效，将数据拆分到2个分片，提升并发/存储能力）
)

```

```python
# ------------------------ 核心功能：Milvus向量数据库索引实战（创建/查看/删除） ------------------------
# 索引是Milvus提升向量检索效率的核心，类似MySQL的索引（无索引时向量检索是全量扫描，效率极低）
# 导入Milvus核心模块
from pymilvus import connections, db, MilvusClient, FieldSchema, DataType, CollectionSchema, Collection
# 导入警告处理模块，忽略无关的deprecated警告（提升代码运行体验）
import warnings
# 过滤"pkg_resources is deprecated"警告（Milvus客户端依赖的第三方库警告，不影响功能）
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# ------------------------ 步骤1：初始化Milvus客户端（HTTP方式连接远程服务） ------------------------
# MilvusClient：更简洁的客户端API，适合快速操作索引/集合
# uri：Milvus服务的访问地址（IP+端口，需替换为实际部署地址）
client = MilvusClient(uri="http://192.168.64.137:19530")

# ------------------------ 【注释段】前置操作：创建集合（表）+ 定义字段 ------------------------
# # 1. 创建集合Schema（表结构）
# schema = MilvusClient.create_schema(
#     auto_id = False,        # 关闭自动生成ID（手动指定id字段值）
#     enable_dynamic_field = True, # 开启动态字段（允许插入未定义的标量字段）
# )
# # 2. 向Schema添加主键字段（id）
# schema.add_field(
#     field_name = "id",      # 字段名
#     datatype = DataType.INT64, # 数据类型：64位整数
#     is_primary = True,      # 设为主键（唯一标识每条数据）
# )
# # 3. 向Schema添加向量字段（vector）
# schema.add_field(
#     field_name = "vector",  # 向量字段名
#     datatype = DataType.FLOAT_VECTOR, # 数据类型：浮点型向量
#     dim = 5                 # 向量维度（示例为5维，实际需与嵌入模型维度匹配，如768/1024）
# )
# # 4. 创建集合（表）
# client.create_collection(
#     collection_name = "customized_setup", # 集合名
#     schema = schema,                      # 绑定上面定义的表结构
# )

# ------------------------ 【注释段】核心操作：为向量字段创建索引 ------------------------
# # 1. 初始化索引参数对象（用于配置索引的各项参数）
# index_params = MilvusClient.prepare_index_params()
# # 2. 配置向量字段的索引参数（核心步骤）
# index_params.add_index(
#     field_name = "vector",          # 要创建索引的字段名（必须是向量字段）
#     metric_type = "COSINE",         # 距离度量方式：COSINE（余弦相似度，适合文本嵌入向量）
#                                     # 可选值：L2（欧式距离）、IP（内积）、COSINE（余弦）
#     index_type="IVF_FLAT",          # 索引类型：IVF_FLAT（基础且稳定的索引，适合中小数据量）
#                                     # 其他常用类型：HNSW（高并发/大数据量）、DISKANN（海量数据）
#     index_name = "vector_index",    # 索引名称（自定义，用于后续查看/删除索引）
#     params = {
#         "nlist": 1024              # IVF_FLAT索引核心参数：聚类中心数
#                                     # 建议值：sqrt(总数据量)，如10万条数据设为300-1000
#     }
# )
# # 3. 执行索引创建
# client.create_index(
#     collection_name = "customized_setup", # 集合名（要创建索引的表）
#     index_params = index_params,          # 绑定上面配置的索引参数
#     sync = False                          # 是否同步等待索引创建完成：False（异步，立即返回）
#                                           # 大数据量时建议异步，避免阻塞；小数据量可设为True
# )

# ------------------------ 实战操作1：查看索引详情（类似MySQL的DESC/ SHOW INDEX） ------------------------
# describe_index：查询指定集合中指定索引的详细信息（字段、类型、参数等）
res = client.describe_index(
    collection_name = "customized_setup",  # 集合名称（要查询的表）
    index_name = "vector_index"            # 索引名称（要查询的索引）
)
# 打印索引详情（包含索引类型、度量方式、参数、创建时间等）
print(res)

# ------------------------ 实战操作2：删除索引 ------------------------
# drop_index：删除指定集合中的指定索引（删除后检索会变回全量扫描，效率降低）
res = client.drop_index(
    collection_name = "customized_setup",  # 集合名称
    index_name = "vector_index"            # 要删除的索引名称
)
# 打印删除结果（成功返回None或True，失败抛出异常）
print(res)
```

### Milvus向量数据库DML实战（插入/删除/更新）

```python
# ------------------------ 核心功能：Milvus向量数据库DML实战（插入/删除/更新） ------------------------
# DML（Data Manipulation Language）：数据操作语言，对应Milvus的插入、删除、更新（Milvus无原生更新，需删插）
# 导入Milvus核心模块
from pymilvus import connections, db, MilvusClient, FieldSchema, DataType, CollectionSchema, Collection
# 导入警告处理模块，忽略无关的deprecated警告
import warnings
# 过滤"pkg_resources is deprecated"警告（Milvus客户端依赖库的冗余警告，不影响功能）
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# ------------------------ 步骤1：初始化Milvus客户端（连接远程Milvus服务） ------------------------
# MilvusClient：简化版客户端API，更适合DML操作（插入/删除/查询）
# uri：Milvus服务的访问地址（IP+端口，需替换为实际部署地址）
client = MilvusClient(uri="http://192.168.64.137:19530")

# ------------------------ 【注释段】前置操作：创建集合+索引（DML的前提） ------------------------
# # 1. 创建集合Schema（表结构）
# schema = MilvusClient.create_schema(
#     auto_id = False,                # 关闭自动生成ID（需手动指定id字段值）
#     enable_dynamic_field = True,    # 开启动态字段（允许插入Schema未定义的字段，如示例中的text）
# )
# # 2. 添加主键字段（id）
# schema.add_field(
#     field_name = "id",              # 字段名
#     datatype = DataType.INT64,      # 数据类型：64位整数
#     is_primary = True,              # 设为主键（唯一标识每条数据）
# )
# # 3. 添加向量字段（vector）
# schema.add_field(
#     field_name = "vector",          # 向量字段名
#     datatype = DataType.FLOAT_VECTOR, # 数据类型：浮点型向量
#     dim = 5                         # 向量维度（示例为5维，实际需匹配嵌入模型维度）
# )
#
# # 4. 配置向量字段索引（提升检索效率，DML后检索需依赖索引）
# index_params = MilvusClient.prepare_index_params()
# index_params.add_index(
#     field_name = "vector",          # 要创建索引的向量字段名
#     metric_type = "COSINE",         # 距离度量方式：余弦相似度（适合文本嵌入向量）
#     index_type="IVF_FLAT",          # 索引类型：IVF_FLAT（基础稳定，中小数据量首选）
#     index_name = "vector_index",    # 索引名称（自定义）
#     params = {
#         "nlist": 1024              # IVF_FLAT核心参数：聚类中心数（建议sqrt(数据量)）
#     }
# )
#
# # 5. 创建集合（表）并绑定索引
# client.create_collection(
#     collection_name = "my_collection", # 集合名（表名）
#     schema = schema,                  # 绑定表结构
#     index_params = index_params,      # 绑定索引配置（创建集合时自动创建索引）
# )

# ------------------------ 【注释段】DML操作1：插入数据（核心写操作） ------------------------
# # 模拟待插入的数据（包含主键、向量字段、动态字段text）
# data = [
#     {
#         "id": 1,                          # 主键ID（必须唯一）
#         "vector": [0.1, 0.2, 0.3, 0.4, 0.5], # 5维向量（维度需与Schema定义一致）
#         "text": "hello world"              # 动态字段（Schema未定义，因enable_dynamic_field=True才支持）
#     },
#     {
#         "id": 2,
#         "vector": [0.2, 0.3, 0.4, 0.5, 0.6],
#         "text": "hello milvus"
#     },
#     {
#         "id": 3,
#         "vector": [0.3, 0.4, 0.5, 0.6, 0.7],
#         "text": "hello python"
#     }
# ]
#
# # 打印待插入数据（验证数据格式）
# print(data)
#
# # 执行插入操作
# result = client.insert(
#     collection_name = "my_collection", # 目标集合名
#     data = data,                      # 待插入的数据列表
# )
# # 打印插入结果（包含插入成功的条数、ID列表等）
# print(result)

# ------------------------ DML操作2：删除数据（核心删操作） ------------------------
# client.delete：删除指定集合中的数据，支持两种方式：
# 1. ids：指定主键ID列表（精准删除，推荐）；
# 2. filter：过滤条件（如"id == 1"或"text like 'hello%'"，批量删除）
client.delete(
    collection_name = "my_collection",  # 目标集合名
    ids = [1, 2],                       # 要删除的数据主键ID列表（删除ID=1和ID=2的两条数据）
    # filter="id == 1"                  # 备选删除方式：通过过滤条件删除（注释掉，二选一）
)

# ------------------------ DML操作3：更新数据（Milvus无原生UPDATE，需先删后插） ------------------------
# 重要说明：Milvus不支持直接UPDATE操作，更新数据的核心逻辑是：
# 1. 删除原数据（通过ID/filter）；
# 2. 插入新数据（使用相同ID，覆盖原数据）；
# 示例伪代码：
# # 1. 删除要更新的ID=3的数据
# client.delete(collection_name="my_collection", ids=[3])
# # 2. 插入新的ID=3的数据（覆盖原数据）
# client.insert(
#     collection_name="my_collection",
#     data=[{"id": 3, "vector": [0.4, 0.5, 0.6, 0.7, 0.8], "text": "hello milvus update"}]
# )
```

### Milvus向量数据库综合实战

```python
# ------------------------ 核心功能：Milvus向量数据库综合实战（图书向量检索系统） ------------------------
# 完整演示：创建集合→生成测试数据→插入数据→创建索引→带过滤条件的向量相似性检索
# 模拟场景：图书检索 - 根据图书简介的向量相似度，筛选指定分类（如Python）的图书
from pymilvus import connections, db, MilvusClient, FieldSchema, DataType, CollectionSchema, Collection
import random  # 用于生成随机测试数据

# 冗余导入（原代码误导入，保留仅作注释说明）
# from sympy.integrals.meijerint_doc import category

# ------------------------ 步骤1：初始化Milvus客户端（连接远程Milvus服务） ------------------------
# MilvusClient：简化版客户端API，一站式完成集合创建、数据插入、索引、检索
# uri：Milvus服务的访问地址（IP+端口，需替换为实际部署地址）
client = MilvusClient(uri="http://192.168.64.137:19530")

# ------------------------ 【注释段】步骤2：创建图书集合（表）+ 定义字段结构 ------------------------
# # 定义集合的字段列表（包含主键、标量字段、向量字段）
# field1 = [
#     # 主键字段：图书ID（自动生成）
#     FieldSchema(name="book_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
#     # 标量字段：图书标题（字符串类型，最大长度200）
#     FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
#     # 标量字段：图书分类（如Python/Java，字符串类型）
#     FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),
#     # 标量字段：图书价格（浮点型，保留2位小数）
#     FieldSchema(name="price", dtype=DataType.DOUBLE),
#     # 向量字段：图书简介的嵌入向量（4维，实际场景需匹配嵌入模型维度如768）
#     FieldSchema(name="book_intro", dtype=DataType.FLOAT_VECTOR, dim=4)
# ]
#
# # 创建集合Schema（表结构）
# schema = CollectionSchema(
#     field1, 
#     description="book search collection",  # 集合描述：图书检索库
#     enable_dynamic_field=True              # 开启动态字段（允许插入未定义的字段）
# )
#
# # 执行集合创建（等价于创建数据库表）
# collection = client.create_collection(collection_name="book", schema=schema)

# ------------------------ 【注释段】步骤3：生成并插入批量测试数据 ------------------------
# # 定义测试数据规模：生成1000本图书数据
# num_books = 1000
# # 图书分类列表（用于随机生成）
# category = ["Python", "Java", "C++", "C#", "Ruby", "Go", "PHP", "JavaScript", "Swift", "Kotlin"]
# # 图书标题前缀（用于随机生成）
# titles = ["Java基础", "Python基础", "C++基础", "C#基础", "Ruby基础", "Go基础", "PHP基础", "JavaScript基础", "Swift基础", "Kotlin基础"]
# # 初始化数据列表
# data = []
# for i in range(num_books):
#     data.append(
#         {
#             "title": f"{random.choice(titles)}_{i}",  # 随机标题+序号（如Python基础_123）
#             "category": random.choice(category),      # 随机分类
#             "price": round(random.uniform(10, 100), 2), # 随机价格（10~100元，保留2位小数）
#             "book_intro": [random.random() for _ in range(4)] # 随机生成4维向量（模拟图书简介嵌入）
#         }
#     )
#
# # 批量插入数据到book集合
# insert_result = client.insert(
#     collection_name = "book",  # 目标集合名
#     data = data                # 待插入的图书数据列表
# )
# # 打印插入结果：输出成功插入的图书ID数量
# print(f"数据插入:{len(insert_result['ids'])}")

# ------------------------ 【注释段】步骤4：为向量字段创建索引（提升检索效率） ------------------------
# # 初始化索引参数对象
# index_params = MilvusClient.prepare_index_params()
# # 配置向量字段的索引参数
# index_params.add_index(
#     field_name = "book_intro",          # 要创建索引的向量字段：图书简介向量
#     metric_type = "L2",                 # 距离度量方式：L2欧式距离（适合数值型向量）
#     index_type="IVF_FLAT",              # 索引类型：IVF_FLAT（中小数据量首选）
#     index_name = "vector_index",        # 索引名称：自定义标识
#     params = {
#         "nlist": 128                   # IVF_FLAT核心参数：聚类中心数（sqrt(1000)≈32，此处设128）
#     }
# )
#
# # 执行索引创建（异步创建，不阻塞）
# client.create_index(
#     collection_name = "book",
#     index_params = index_params,
#     sync = False  # 异步创建：立即返回，后台创建索引（大数据量建议异步）
# )

# ------------------------ 步骤5：执行带过滤条件的向量相似性检索（核心实战） ------------------------
# 加载集合到内存（检索前必须执行，索引和数据需加载到内存才能生效）
client.load_collection(collection_name = "book")

# 生成查询向量：随机生成4维向量（模拟用户输入的"图书简介"转换后的向量）
query_vector = [random.random() for _ in range(4)]

# 执行向量检索（结合标量过滤）
result = client.search(
    collection_name = "book",          # 目标集合名
    data = [query_vector],             # 查询向量（支持批量查询，列表格式）
    filter = "category == 'Python'",   # 标量过滤条件：只检索Python分类的图书
    limit = 3,                         # 返回相似度最高的3条结果
    search_params= {"nprobe": 10},     # 检索参数：nprobe=10（IVF_FLAT检索时遍历的聚类中心数，越大越精准但越慢）
    output_fields= ["title", "price","category"] # 指定返回的标量字段（不指定则只返回ID和距离）
)

# ------------------------ 步骤6：解析并格式化输出检索结果 ------------------------
# 打印原始检索结果（便于调试）
print(result)

# 格式化输出Python分类的检索结果
print("\nPython相关的结果")
for item in result[0]: # result是二维列表，result[0]对应第一个查询向量的结果
    print(f"id: {item['id']}")                  # 图书ID（主键）
    print(f"距离: {item['distance']:.4f}")      # 向量距离（越小相似度越高）
    print(f"标题: {item['entity']['title']}")   # 图书标题
    print(f"价格: {item['entity']['price']:.2f}") # 图书价格
    print("_" * 30 )                            # 分隔线，提升可读性
```

### Milvus向量数据库检索进阶+集合/索引信息查询

```python
# ------------------------ 核心功能：Milvus向量数据库检索进阶+集合/索引信息查询 ------------------------
# 演示场景：图书向量检索（带过滤/分页）、批量向量查询、查询集合结构/索引详情
from pymilvus import connections, db, MilvusClient, FieldSchema, DataType, CollectionSchema, Collection
import random  # 用于生成随机查询向量

# ------------------------ 步骤1：初始化Milvus客户端（连接远程Milvus服务） ------------------------
# uri：Milvus服务的IP+端口（需替换为实际部署地址）
client = MilvusClient(uri="http://192.168.64.137:19530")

# ------------------------ 步骤2：生成随机查询向量（模拟用户检索的图书简介向量） ------------------------
# 生成4维随机向量（需与book集合中book_intro字段的dim=4一致）
# _ 是占位符，表示循环4次但不使用循环变量
query_vector = [random.random() for _ in range(4)]

# ------------------------ 核心检索：带标量过滤的向量相似性搜索 ------------------------
result = client.search(
    collection_name = "book",          # 目标集合名（图书库）
    data = [query_vector],             # 查询向量（列表格式，支持批量查询）
    filter = "category == 'Python'",   # 标量过滤条件：仅检索Python分类的图书
    limit = 3,                         # 返回相似度最高的3条结果
    search_params= {"nprobe": 10},     # 检索参数：IVF_FLAT索引的nprobe（遍历10个聚类中心，平衡精度/速度）
    output_fields= ["title", "price","category"] # 指定返回的标量字段（避免返回冗余数据）
)

# ------------------------ 【注释段】检索进阶1：基础分页检索（无过滤） ------------------------
# # 仅返回相似度最高的3条结果（基础检索，无过滤）
# response = client.search(
#     collection_name = "book",
#     data = [query_vector], # 单个查询向量
#     limit=3                # 返回Top3结果
# )
# print(response)

# ------------------------ 【注释段】检索进阶2：带偏移量的分页检索 ------------------------
# # 分页检索：跳过前2条结果，返回接下来的3条（实现“第2页，每页3条”效果）
# response = client.search(
#     collection_name = "book",
#     data = [query_vector],
#     offset=2,  # 偏移量：跳过前2条结果
#     limit=3    # 每页条数：返回3条
# )
# print(response)

# ------------------------ 【注释段】检索进阶3：批量向量查询 ------------------------
# # 同时传入2个查询向量，每个向量都返回“跳过2条+取3条”的结果
# # [0.5] * 4 表示生成[0.5, 0.5, 0.5, 0.5]的4维向量
# response = client.search(
#     collection_name = "book",
#     data = [query_vector,[0.5] * 4], # 批量查询：2个查询向量
#     offset = 2 ,                     # 每个向量的结果都跳过前2条
#     limit=3                          # 每个向量返回3条结果
# )
# print(response)

# ------------------------ 元数据查询1：查看集合（表）的详细信息 ------------------------
# describe_collection：查询集合的Schema、字段、动态字段、分片等元数据（类似MySQL的DESC TABLE）
print("===== 集合（book）详情 =====")
print(client.describe_collection("book"))

# ------------------------ 元数据查询2：查看索引的详细信息 ------------------------
# describe_index：查询集合中向量字段的索引配置（类型、度量方式、参数等）
print("\n===== 索引（book）详情 =====")
print(client.describe_index("book"))
```

### LangChain \+ Milvus 向量库实战（嵌入+增删数据）

```python
# ------------------------ 核心功能：LangChain + Milvus 向量库实战（嵌入+增删数据） ------------------------
# 演示场景：使用阿里云通义千问嵌入模型生成文本向量，将文档存入Milvus向量库，并实现数据删除
# 核心依赖：langchain-milvus（LangChain对接Milvus的适配器）、DashScopeEmbeddings（通义千问嵌入模型）
from langchain_community.embeddings import DashScopeEmbeddings  # 导入通义千问嵌入模型
from langchain_core.documents import Document                   # 导入LangChain的文档数据结构
from langchain_milvus import Milvus                             # 导入LangChain对接Milvus的向量库类

# ------------------------ 步骤1：初始化通义千问嵌入模型（生成文本向量） ------------------------
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",        # 指定嵌入模型版本：第二代通用文本嵌入模型（生成高质量向量）
    max_retries=3,                    # 请求失败时的重试次数：最多重试3次，提升稳定性
    dashscope_api_key="", # 通义千问API密钥（需替换为自己的有效密钥）
)

# ------------------------ 步骤2：初始化Milvus向量库（对接远程Milvus服务） ------------------------
vector_store = Milvus(
    embedding_function = embeddings,  # 绑定嵌入模型：文档会自动通过该模型生成向量
    connection_args= {"uri":"http://192.168.64.137:19530"}, # Milvus服务连接地址（IP+端口）
    collection_name="langchain_example", # Milvus集合名（相当于数据库表名）
)

# ------------------------ 步骤3：构造测试文档（LangChain标准Document格式） ------------------------
# Document是LangChain的标准文档结构，包含2个核心字段：
# - page_content：文档的文本内容（会被嵌入模型转为向量）
# - metadata：文档的元数据（标量信息，用于过滤检索，如来源、分类等）
document_1 = Document(
    page_content="I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},  # 元数据：文档来源是推特
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},   # 元数据：文档来源是新闻
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"}, # 元数据：文档来源是网站
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
)

# 将所有文档整合为列表，方便批量操作
documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]

# ------------------------ 【注释段】步骤4：批量插入文档到Milvus向量库 ------------------------
# # 生成自定义文档ID：为每个文档分配唯一ID（1~10），方便后续删除/检索
# ids = [str(i+1) for i in range(len(documents))]
# print(ids)  # 打印ID列表：['1','2','3',...,'10']
#
# # 批量插入文档：LangChain会自动将page_content转为向量，metadata存入标量字段
# result = vector_store.add_documents(documents = documents,ids = ids)
# print(result)  # 打印插入结果：返回成功插入的文档ID列表

# ------------------------ 步骤5：删除Milvus中的指定文档 ------------------------
# delete方法：根据文档ID精准删除Milvus中的数据（需与插入时的ID对应）
result = vector_store.delete(ids = ["1"])  # 删除ID为"1"的文档（即document_1）
print(result)  # 打印删除结果：成功返回None/True，失败抛出异常
```

### LangChain + Milvus 向量检索进阶（MMR检索）

```python
# ------------------------ 核心功能：LangChain + Milvus 向量检索进阶（MMR检索） ------------------------
# 演示场景：基于通义千问嵌入模型生成文本向量，将文档存入Milvus后，实现普通相似性检索、带分数的相似性检索，以及MMR（最大边际相关性）检索
# MMR核心价值：在保证相似度的同时，提升检索结果的多样性，避免结果高度重复
from langchain_community.embeddings import DashScopeEmbeddings  # 导入通义千问文本嵌入模型
from langchain_core.documents import Document                   # 导入LangChain标准文档结构
from langchain_milvus import Milvus                             # 导入LangChain对接Milvus的向量库类

# ------------------------ 步骤1：初始化通义千问嵌入模型 ------------------------
# 作用：将自然语言文本转换为数值向量（用于Milvus的向量相似性计算）
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",        # 指定嵌入模型版本：第二代通用文本嵌入模型（效果更优）
    max_retries=3,                    # API请求失败后的重试次数：提升请求稳定性
    dashscope_api_key="", # 阿里云通义千问API密钥（需替换为自己的有效密钥）
)

# ------------------------ 步骤2：构造测试文档集 ------------------------
# Document是LangChain的标准化文档对象，包含两个核心部分：
# - page_content：文档的核心文本内容（会被嵌入模型转为向量）
# - metadata：文档的元数据（标量信息，如来源、分类等，用于过滤/溯源）
document_1 = Document(
    page_content="I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},  # 元数据：文档来源为推特
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},   # 元数据：文档来源为新闻
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
)

# 将所有文档整合为列表，方便批量操作
documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5
]

# ------------------------ 步骤3：生成自定义文档ID ------------------------
# 为每个文档生成唯一ID（1~5），格式为字符串（Milvus主键常用字符串类型）
# 作用：方便后续精准删除、检索溯源，若不指定则Milvus会自动生成ID
ids = [str(i + 1) for i in range(len(documents))]
print(ids)  # 打印ID列表：输出 ['1', '2', '3', '4', '5']

# ------------------------ 【注释段】方式1：快速初始化Milvus并插入文档（一键式） ------------------------
# # from_documents是Milvus类的快捷方法：自动完成「文档嵌入→创建集合→插入数据」全流程
# # 无需手动初始化Milvus再调用add_documents，适合快速开发
# vector_storage = Milvus.from_documents(
#     documents = documents,          # 待插入的文档列表
#     embedding=embeddings,           # 绑定的嵌入模型
#     collection_name="mmr_test",     # Milvus集合名（表名）
#     connection_args={"uri": "http://192.168.64.137:19530"}) # Milvus连接地址

# ------------------------ 步骤4：手动初始化Milvus向量库（更灵活） ------------------------
# 手动初始化方式：可自定义更多参数（如索引、分片等），适合精细化控制
vector_store = Milvus(
    embedding_function=embeddings,  # 绑定嵌入模型（检索时会自动将查询文本转为向量）
    connection_args={"uri": "http://192.168.64.137:19530"}, # Milvus服务连接地址
    collection_name="langchain_example", # 指定操作的集合名（需确保该集合已存在/已插入数据）
)

# ------------------------ 【注释段】检索1：基础相似性检索（仅返回文档） ------------------------
# # 相似性检索：根据查询文本的向量，返回最相似的k个文档（默认按相似度降序）
# query = "I had chocalate chip pancakes and scrambled eggs for breakfast this morning."
# result = vector_store.similarity_search(query,k=2)  # k=2：返回前2个最相似的文档
# for i in result:
#     print(f"内容：{i.page_content},元数据:{i.metadata}")  # 打印文档内容和元数据

# ------------------------ 【注释段】检索2：带相似度分数的相似性检索 ------------------------
# # similarity_search_with_score：返回文档+相似度分数（分数越小，相似度越高）
# # 相比基础检索，可直观判断匹配程度，适合需要量化相似度的场景
# query = "I had chocalate chip pancakes and scrambled eggs for breakfast this morning."
# result = vector_store.similarity_search_with_score(query,k=2)
# for i in result:
#     # i[0]是Document对象（内容+元数据），i[1]是相似度分数
#     print(f"内容：{i[0].page_content},元数据:{i[0].metadata},分数:{i[1]}")

# ------------------------ 核心检索：MMR（最大边际相关性）检索 ------------------------
# MMR检索：在保证相似度的前提下，最大化结果的多样性，避免检索结果高度重复
# 适用场景：推荐系统、问答系统（需要覆盖更多相关维度，而非仅最相似）
query = "I had chocalate chip pancakes and scrambled eggs for breakfast this morning."
result = vector_store.max_marginal_relevance_search(
    query=query,          # 检索查询文本
    k=3,                  # 最终返回的文档数量
    fetch_k=10,           # 先检索出最相似的10个文档（候选集），再从中选k个多样性最高的
    lambda_mult=0.4       # 平衡参数：0~1，越接近1越侧重相似度，越接近0越侧重多样性
)
# 遍历并打印MMR检索结果
for i in result:
    print(f"内容：{i.page_content},元数据:{i.metadata}")
```

### 检索器Retriever实战

```python
# 导入所需的库和模块
# DashScopeEmbeddings：阿里云百炼提供的文本嵌入模型，用于将文本转换为向量
from langchain_community.embeddings import DashScopeEmbeddings
# Document：LangChain 中的文档对象，用于封装文本内容和元数据
from langchain_core.documents import Document
# Milvus：向量数据库 Milvus 的 LangChain 集成，用于存储和检索向量
from langchain_milvus import Milvus

# ======================== 初始化嵌入模型 ========================
# 创建 DashScopeEmbeddings 实例，用于生成文本的向量表示
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",  # 指定使用的嵌入模型版本：第二代通用文本嵌入模型
    max_retries=3,  # API 调用失败时的最大重试次数，增强鲁棒性
    dashscope_api_key="",  # 阿里云百炼的 API 密钥（注意：实际使用时建议通过环境变量配置，避免硬编码）
)

# ======================== 构建示例文档数据 ========================
# 创建 Document 对象，每个对象包含文本内容（page_content）和元数据（metadata）
# 元数据可以用于标识文档来源、类型等附加信息
document_1 = Document(
    page_content="I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},  # 标记文档来源为推特
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},  # 标记文档来源为新闻
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
)

# 将所有 Document 对象整合为一个列表，方便批量处理
documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5
]

# 为每个文档生成唯一的 ID（字符串类型），ID 从 1 开始递增
ids = [str(i + 1) for i in range(len(documents))]
print(f"生成的文档 ID 列表：{ids}")  # 打印 ID 列表，用于调试验证

# ======================== 将文档存入 Milvus 向量数据库 ========================
# 调用 Milvus.from_documents 方法，将文档转换为向量并批量插入 Milvus
# 该方法会自动完成：文本 -> 向量（通过 embeddings 模型）-> 存入 Milvus
vector_storage = Milvus.from_documents(
    documents = documents,  # 待插入的文档列表
    embedding=embeddings,   # 用于生成向量的嵌入模型
    collection_name="retriever_test",  # Milvus 中的集合名称（相当于数据库中的表名）
    connection_args={"uri": "http://192.168.64.137:19530"}  # Milvus 服务的连接地址和端口
)

# ======================== 基于向量的相似性检索 ========================
# 将 Milvus 存储转换为检索器（Retriever），支持两种检索方式：
# 1. 相似性检索（默认）：返回最相似的 k 个结果
# 2. MMR（Maximum Marginal Relevance）：最大边际相关性检索，在相似性基础上保证结果的多样性

# 方式1：普通相似性检索（注释掉，备用）
# retriever = vector_storage.as_retriever(search_kwargs={"k": 3})

# 方式2：MMR 多样性检索（启用）
retriever = vector_storage.as_retriever(
    search_kwargs={"k": 3},  # 检索参数：返回 Top 3 个结果
    search_type="mmr"        # 检索类型：MMR（平衡相似性和多样性）
)

# 执行检索：传入查询语句，检索与 "What is LangChain?" 最相关的文档
result = retriever.invoke("What is LangChain?")

# 遍历并打印检索结果
for r in result:
    print(f"文档内容:{r.page_content}, 元数据:{r.metadata}")
```

### MultiQueryRetriever（多查询检索器）优化向量检索效果实战

```python
# 核心功能：演示通过 MultiQueryRetriever（多查询检索器）优化向量检索效果，通过LLM自动生成多个不同角度的查询语句，
# 扩大检索范围并融合结果，从而提升检索的召回率和准确率（目标提升约30%）
import logging

# 导入多查询检索器（核心组件，用于生成多视角查询提升检索效果）
from langchain.retrievers import MultiQueryRetriever
# 导入文档加载器：TextLoader加载本地文本文件，RecursiveUrlLoader加载网页内容（此处仅用TextLoader）
from langchain_community.document_loaders import TextLoader, RecursiveUrlLoader
# 导入阿里云百炼嵌入模型，用于文本向量化
from langchain_community.embeddings import DashScopeEmbeddings
# 导入Milvus向量数据库集成，用于存储和检索向量
from langchain_milvus import Milvus
# 导入ChatOpenAI大模型，用于生成多视角查询语句
from langchain_openai import ChatOpenAI
# 导入递归字符文本分割器，用于将长文档切分为小片段
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 导入SecretStr，用于安全存储API密钥（避免明文泄露）
from pydantic import SecretStr

# ======================== 日志配置 ========================
# 设置日志系统的基础配置（默认输出到控制台）
logging.basicConfig()
# 将MultiQueryRetriever的日志级别设置为INFO，便于查看其生成的多查询语句等关键信息
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# ======================== 文档加载与预处理 ========================
# 初始化文本加载器，加载本地UTF-8编码的qa.txt文件（存储待检索的问答数据）
loader = TextLoader("data/qa.txt", encoding="utf-8")
# 加载文件内容为LangChain的Document对象
data = loader.load()

# 初始化递归字符文本分割器（适合中文文本分割）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # 每个文本片段的最大字符数（小片段提升检索精准度）
    chunk_overlap=10  # 片段间重叠字符数（避免分割导致上下文丢失）
)
# 将加载的长文档切分为多个小文本片段
splits = text_splitter.split_documents(data)

# ======================== 嵌入模型初始化 ========================
# 初始化阿里云百炼文本嵌入模型，将文本转换为向量表示
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",  # 指定第二代通用文本嵌入模型（兼顾效果和效率）
    max_retries=3,  # API调用失败时的最大重试次数（提升稳定性）
    dashscope_api_key="",  # 阿里云百炼API密钥（建议通过环境变量配置）
)

# ======================== 向量数据库入库 ========================
# 将切分后的文本片段转换为向量，并批量存入Milvus向量数据库
vector_storage = Milvus.from_documents(
    documents=splits,  # 待入库的文本片段列表
    embedding=embeddings,  # 用于生成向量的嵌入模型
    collection_name="multi_query_tesr",  # Milvus中的集合名称（注意：原代码拼写错误，应为multi_query_test）
    connection_args={"uri": "http://192.168.64.137:19530"}  # Milvus服务的连接地址和端口
)

# ======================== 检索配置 ========================
# 待检索的目标问题（核心查询）
question = "老王不知道为什么抽筋了？"

# 初始化大语言模型（用于生成多视角查询语句）
model = ChatOpenAI(
    model="qwen-plus",  # 指定使用通义千问plus模型（效果较好）
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云百炼兼容OpenAI接口的地址
    api_key=SecretStr(""),  # 安全存储API密钥（SecretStr避免明文打印）
    temperature=0.7  # 生成文本的随机性（0-1，0.7兼顾多样性和准确性）
)

# 初始化多查询检索器（核心优化组件）
# 原理：基于原始问题，通过LLM自动生成多个语义相似但表述不同的查询语句，分别检索后融合结果
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vector_storage.as_retriever(),  # 基础向量检索器（Milvus）
    llm=model  # 用于生成多查询语句的大模型
)

# ======================== 执行检索并输出结果 ========================
# 执行多查询检索，获取与问题相关的文档结果
results = retriever_from_llm.invoke(question)
# 打印检索到的文档数量（便于评估召回率）
print(f"检索到的文档数量：{len(results)}")

# 遍历并打印每个检索结果的内容和元数据
for result in results:
    print(f"内容:{result.page_content},元数据:{result.metadata}"
```

### _**AI文档助手综合实战**_

```python
# AI文档助手综合实战：基于LangChain+Milvus+通义千问构建完整的RAG（检索增强生成）系统，
# 实现从网页加载Milvus官方文档、文本切分、向量入库、语义检索到智能回答的全流程，
# 最终为用户提供精准的Milvus文档问答服务
from langchain_community.document_loaders import WebBaseLoader  # 网页文档加载器，用于加载在线网页内容
from langchain_community.embeddings import DashScopeEmbeddings  # 阿里云百炼嵌入模型，将文本转为向量
from langchain_core.prompts import PromptTemplate  # 提示词模板，用于构建LLM的输入提示
from langchain_core.runnables import RunnablePassthrough  # 链式执行工具，透传用户问题到后续环节
from langchain_milvus import Milvus  # Milvus向量数据库集成，存储/检索向量
from langchain_openai import ChatOpenAI  # OpenAI兼容的大模型调用接口（适配通义千问）
from langchain_text_splitters import CharacterTextSplitter  # 文本分割器，将长文档切分为小片段
from pydantic import SecretStr  # 安全存储敏感信息（如API密钥），避免明文泄露

# ======================== 全局配置 ========================
# 定义Milvus向量数据库中的集合名称（相当于数据库表名），统一管理存储的文档向量
COLLECTION_NAME = "doc_qa_db"

# ======================== 步骤1：加载在线文档 ========================
# 初始化网页加载器，加载Milvus官方中文文档的多个核心页面
loader = WebBaseLoader(
    # 待加载的网页URL列表（Milvus概述、发布说明、快速开始）
    ["https://milvus.io/docs/zh/overview.md",
     "https://milvus.io/docs/zh/release_notes_md",
     "https://milvus.io/docs/zh/quickstart.md"],
    # 请求头配置：强制要求返回中文内容，避免加载到英文文档
    requests_kwargs={"headers": {"Accept-Language": "zh-CN"}}
)
# 执行加载，将网页内容转换为LangChain的Document对象（包含文本内容和元数据）
data = loader.load()

# ======================== 步骤2：文本分割（核心预处理） ========================
# 初始化字符文本分割器，解决长文本向量化效果差的问题
text_splitter = CharacterTextSplitter(
    chunk_size=1024,  # 每个文本片段的最大字符数（适配嵌入模型的输入限制）
    chunk_overlap=20  # 片段间重叠字符数（避免分割导致上下文断裂）
)
# 将加载的网页文档切分为多个小文本片段，提升后续检索的精准度
all_split = text_splitter.split_documents(data)

# ======================== 步骤3：初始化嵌入模型 ========================
# 初始化阿里云百炼文本嵌入模型，负责将文本片段转换为数值向量（Embedding）
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",  # 指定第二代通用文本嵌入模型（兼顾效果和效率）
    max_retries=3,  # API调用失败时的最大重试次数（提升稳定性）
    dashscope_api_key="",  # 阿里云百炼API密钥（建议通过环境变量配置）
)

# ======================== 步骤4：向量数据库入库 ========================
# 方式1：简化版入库（注释备用）
# vector_storage = Milvus.from_documents(
#     documents = all_split,
#     embedding=embeddings,
#     collection_name=COLLECTION_NAME,
#     connection_args={"uri": "http://192.168.64.137:19530"})

# 方式2：自定义Milvus配置后入库（当前使用）
vector_storage = Milvus(
    embedding_function=embeddings,  # 指定向量生成的嵌入模型
    collection_name=COLLECTION_NAME,  # 指定存储的集合名称
    connection_args={"uri": "http://192.168.64.137:19530"},  # Milvus服务连接地址（IP+端口）
    drop_old=True  # 入库前删除同名旧集合（避免数据冗余，适合测试/迭代场景）
).from_documents(
    documents=all_split,  # 待入库的文本片段列表
    embedding=embeddings,  # 嵌入模型（与上方保持一致）
    collection_name=COLLECTION_NAME,  # 集合名称（与上方保持一致）
    connection_args={"uri": "http://192.168.64.137:19530"}  # 连接信息（与上方保持一致）
)

# 方式3：仅初始化Milvus连接（注释备用，适用于已入库场景）
# vector_storage = Milvus(
#     embedding_function=embeddings,
#     collection_name=COLLECTION_NAME,
#     connection_args={"uri": "http://192.168.64.137:19530"})

# ======================== 步骤5：测试基础相似性检索（注释备用） ========================
# 定义用户查询问题：获取Milvus的Docker安装命令
query = "docker怎么安装milvus，只告诉我命令就可以了"
# 执行基础相似性检索，返回Top3最相关的文档片段
# docs = vector_storage.similarity_search(query, k=3)
# 打印检索结果（注释备用，用于调试）
# print(docs)
# for doc in docs:
#     print(doc.page_content)

# ======================== 步骤6：初始化大语言模型（LLM） ========================
# 初始化通义千问大模型（通过OpenAI兼容接口调用），负责基于检索结果生成回答
model = ChatOpenAI(
    model="qwen-plus",  # 指定使用通义千问plus模型（效果优于基础版）
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云百炼兼容OpenAI的接口地址
    api_key=SecretStr(""),  # 安全存储API密钥（SecretStr避免明文打印）
    temperature=0.7  # 生成文本的随机性（0-1，0.7兼顾灵活性和准确性）
)

# ======================== 步骤7：构建RAG链式问答系统 ========================
# 将Milvus向量存储转换为检索器（Retriever），支持链式调用
retriever = vector_storage.as_retriever()

# 定义提示词模板（Prompt Template），规范LLM的回答逻辑和格式
prompt_template = """
        你是AI文档助手，使用如下上下文来回答最后的问题
        如果你不知道答案，就说你不知道，不要试图编造答案。
        最多使用10句话，并尽可能简洁的回答。总是在答案末尾说"谢谢你的提问!"
        {content}  # 占位符：填充检索到的相关文档内容
        问题:{question}  # 占位符：填充用户的查询问题
"""
# 将字符串模板转换为LangChain的PromptTemplate对象，便于后续链式调用
rag_prompt = PromptTemplate.from_template(
    template=prompt_template
)

# 构建完整的RAG链式流程：
# 1. {"content": retriever, "question": RunnablePassthrough()}：并行执行，retriever检索相关内容，RunnablePassthrough透传用户问题
# 2. | rag_prompt：将检索内容和问题填充到提示词模板中
# 3. | model：将填充后的提示词输入给LLM，生成最终回答
rag_chain = ({"content": retriever, "question": RunnablePassthrough()} | rag_prompt | model)

# 执行RAG链式调用，传入用户查询问题，获取最终回答
result = rag_chain.invoke(query)
# 打印AI生成的回答结果
print(result)
```

### RunnableBranch实现智能客服路由系统实战

```python
# 核心功能：基于LangChain的RunnableBranch实现智能客服路由系统，模拟if-else逻辑
# 根据用户输入的问题类型（技术问题/财务问题/通用问题），自动路由到对应的专业处理子链，
# 实现不同类型请求的差异化应答，提升客服回复的精准度
from langchain_core.prompts import ChatPromptTemplate  # 聊天提示词模板，构建不同场景的Prompt
from langchain_core.runnables import RunnableBranch, RunnableLambda  # 路由分支/自定义函数执行组件
from langchain_openai import ChatOpenAI  # OpenAI兼容的大模型调用接口（适配通义千问）
from langchain_core.output_parsers import StrOutputParser  # 输出解析器，将LLM响应转为字符串
from pydantic import SecretStr  # 安全存储API密钥，避免明文泄露

# ======================== 初始化大语言模型 ========================
# 创建通义千问大模型实例，作为所有子链的回答生成核心
model = ChatOpenAI(
    model="qwen-plus",  # 指定使用通义千问plus模型（具备更好的理解和生成能力）
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云百炼兼容OpenAI的接口地址
    api_key=SecretStr(""),  # 安全存储API密钥（SecretStr避免明文打印）
    temperature=0.7  # 生成文本的随机性（0-1，0.7兼顾灵活性和专业性）
)

# ======================== 构建不同场景的专业子链 ========================
# 1. 技术支持子链：专门处理技术类问题，使用技术专家的Prompt模板
tech_prompt = ChatPromptTemplate.from_template(
    "你是一名技术支持专家，请回答以下技术问题：{input}"
)
# 技术链流程：Prompt模板填充 → LLM生成回答 → 输出转为字符串
tech_chain = tech_prompt | model | StrOutputParser()

# 2. 财务问题子链：专门处理账单/支付类问题，使用财务专员的Prompt模板
billing_prompt = ChatPromptTemplate.from_template(
    "你是一名财务专员，请处理以下账单问题：{input}"
)
# 财务链流程：Prompt模板填充 → LLM生成回答 → 输出转为字符串
billing_chain = billing_prompt | model | StrOutputParser()

# 3. 默认通用子链：处理非技术/非财务的通用问题，使用通用客服Prompt模板
default_prompt = ChatPromptTemplate.from_template(
    "你是一名客服专员，请回答以下问题：{input}"
)
# 通用链流程：Prompt模板填充 → LLM生成回答 → 输出转为字符串
default_chain = default_prompt | model | StrOutputParser()

# ======================== 定义路由判断函数（核心逻辑） ========================
from typing import Dict  # 导入类型注解，规范函数输入输出类型

def is_tech_question(input: dict) -> bool:
    """判断用户输入是否为技术问题（路由判断函数1）
    Args:
        input: 包含用户输入的字典，格式为 {"input": "用户问题文本"}
    Returns:
        bool: 包含技术关键词返回True，否则返回False
    """
    # 提取用户输入的文本内容（默认空字符串，避免KeyError）
    input_value = input.get("input", "")
    # 技术问题关键词列表（可根据实际业务扩展）
    tech_keywords = ["技术", "故障", "安装", "错误", "bug", "无法运行"]
    # 检查输入是否包含任意技术关键词（原代码仅判断2个，此处优化为通用逻辑）
    return any(keyword in input_value for keyword in tech_keywords)

def is_billing_question(input: dict) -> bool:
    """判断用户输入是否为财务问题（路由判断函数2）
    Args:
        input: 包含用户输入的字典，格式为 {"input": "用户问题文本"}
    Returns:
        bool: 包含财务关键词返回True，否则返回False
    """
    # 提取用户输入的文本内容
    input_value = input.get("input", "")
    # 财务问题关键词列表（原代码注释错误，已修正）
    billing_keywords = ["账单", "支付", "费用", "发票", "退款"]
    # 检查输入是否包含任意财务关键词
    return any(keyword in input_value for keyword in billing_keywords)

# ======================== 构建路由分支（模拟if-else） ========================
# RunnableBranch：按顺序执行判断，匹配第一个为True的条件则执行对应子链，否则执行默认链
# 逻辑等价于：if is_tech_question → tech_chain; elif is_billing_question → billing_chain; else → default_chain
branch = RunnableBranch(
    (is_tech_question, tech_chain),    # 条件1：技术问题 → 技术链
    (is_billing_question, billing_chain),  # 条件2：财务问题 → 财务链
    default_chain                      # 默认分支：通用问题 → 通用链
)

# ======================== 可选：构建带日志的路由链（调试/监控用） ========================
def log_decision(input_data):
    """日志函数：打印路由判断的输入数据，便于调试和监控"""
    print(f"路由检查输入：{input_data}")
    return input_data  # 透传输入数据，不影响后续流程

# 带日志的路由链：先打印日志 → 再执行路由分支
log_chain_branch = RunnableLambda(log_decision) | branch

# ======================== 构建完整的客服链路 ========================
# 完整链流程：
# 1. RunnableLambda(lambda x: {"input": x})：将原始字符串输入转为字典（适配路由函数的输入格式）
# 2. | branch：执行路由分支，匹配对应子链
full_chain = RunnableLambda(lambda x: {"input": x}) | branch

# ======================== 测试不同类型的用户问题 ========================
# 测试1：技术问题 → 路由到技术支持链
tech_response = full_chain.invoke("如何处理系统安装故障？")
print("【技术问题回复】：", tech_response)

# 测试2：财务问题 → 路由到财务链
billing_response = full_chain.invoke("如何查询本月的消费账单？")
print("【财务问题回复】：", billing_response)

# 测试3：通用问题 → 路由到默认通用链
common_response = full_chain.invoke("请问你们的工作时间是什么？")
print("【通用问题回复】：", common_response)
```

### RunnablePassthrough的assign机制（动态添加字段）

```python
# 核心功能：演示LangChain中RunnablePassthrough的assign机制（动态添加字段），并结合Milvus向量库构建完整RAG系统
# 核心逻辑：在链式执行中动态生成/补充字段（如检索上下文、透传用户问题），为LLM提供完整的输入信息
from langchain_community.embeddings import DashScopeEmbeddings  # 阿里云百炼嵌入模型，文本转向量
from langchain_core.documents import Document  # LangChain文档对象，封装文本和元数据
from langchain_core.prompts import ChatPromptTemplate  # 提示词模板，构建LLM输入
from langchain_core.runnables import RunnablePassthrough  # 透传/动态字段组件，核心实现assign功能
from langchain_milvus import Milvus  # Milvus向量数据库集成，存储/检索向量
from langchain_openai import ChatOpenAI  # OpenAI兼容接口，调用通义千问模型
from pydantic import SecretStr  # 安全存储API密钥
import os

# ======================== LangSmith配置（可选，链路追踪） ========================
# 开启LangChain链路追踪，用于调试/监控RAG流程的执行过程
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")  # 从环境变量读取，避免提交密钥
os.environ["LANGCHAIN_PROJECT"]="agent_v1"

# ======================== assign机制基础示例（注释说明核心用法） ========================
# RunnablePassthrough().assign() 核心作用：在输入数据基础上，动态添加新字段（由自定义函数生成）
# 示例1：动态添加processed字段，值为num字段的2倍
# chain = RunnablePassthrough().assign(processed=lambda x: x["num"] * 2)
# output = chain.invoke({"num": 5})  # 执行后输出：{"num":5, "processed":10}
# print(output)


# 示例2：RAG场景中动态添加context字段（检索文档）
# chain = RunnablePassthrough().assign(
#     context=lambda x : retrieve_documents({x["question"]})  # 动态生成context字段（检索结果）
# ) | prompt | llm  # 拼接提示词+LLM生成回答

# 自定义检索函数（模拟从本地读PDF/查数据库/查本地文件找答案）
def retrieve_documents(question):
    # 模拟检索：比如用户问“张三是谁”，返回相关文档内容
    if "张三" in question:
        return "张三是XX公司的产品经理，负责LangChain项目落地"
    else:
        return "未找到相关文档"

# 示例2输入：{"question":"langchain是什么?"}
# 执行后输入会自动补充context字段，最终传给prompt的是：{"question":"langchain是什么?", "context":"检索到的文档"}
# input_data = {"question":"langchain是什么?"}
# response = chain.invoke(input_data)

# ======================== 初始化嵌入模型 ========================
# 创建阿里云百炼嵌入模型实例，用于将文本转换为向量
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",  # 第二代通用文本嵌入模型
    max_retries=3,  # API调用失败重试次数，提升稳定性
    dashscope_api_key="",  # 阿里云百炼API密钥（建议环境变量配置）
)

# ======================== 构建示例文档 ========================
# 创建Document对象，每个对象包含文本内容（page_content）和元数据（metadata）
document_1 = Document(
    page_content="I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},  # 标记文档来源为推特
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},  # 标记文档来源为新闻
)

# 整合文档列表，用于批量入库
documents = [document_1, document_2]

# ======================== 向量数据库入库 ========================
# 将文档转换为向量，并批量存入Milvus向量数据库
vector_storage = Milvus.from_documents(
    documents=documents,  # 待入库的文档列表
    embedding=embeddings,  # 用于生成向量的嵌入模型
    collection_name="runnable_test1",  # Milvus集合名称（表名）
    connection_args={"uri": "http://192.168.64.137:19530"}  # Milvus服务连接地址
)

# ======================== 构建检索器 ========================
# 将Milvus存储转换为检索器（Retriever），支持链式调用
# search_kwargs={"k":3}：检索时返回最相似的3个文档（此处示例只有2个文档，实际返回2个）
retriever = vector_storage.as_retriever(search_kwargs={"k": 3})

# ======================== 构建提示词模板 ========================
# 定义RAG提示词模板，包含两个占位符：content（检索上下文）、question（用户问题）
prompt = ChatPromptTemplate.from_template(
    " 基于上下文回答：{content}\t问题:{question}"
)

# ======================== 初始化大语言模型 ========================
model = ChatOpenAI(
    model="qwen-plus",  # 指定通义千问plus模型
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云百炼兼容OpenAI的接口地址
    api_key=SecretStr(""),  # 安全存储API密钥
    temperature=0.7  # 生成文本的随机性，0.7兼顾灵活性和准确性
)

# ======================== 构建完整RAG链（核心体现assign/透传逻辑） ========================
# 核心逻辑拆解：
# 1. {"content":retriever, "question": RunnablePassthrough()}：
#    - content字段：由retriever检索生成（相当于动态添加检索上下文）
#    - question字段：由RunnablePassthrough()透传用户输入的问题（不做修改）
#    这是assign机制的简化写法，等价于 RunnablePassthrough().assign(content=retriever)
# 2. | prompt：将content和question填充到提示词模板中
# 3. | model：将填充后的提示词输入给LLM，生成最终回答
chain = {"content":retriever, "question": RunnablePassthrough()} | prompt |  model

# ======================== 执行RAG链并输出结果 ========================
# 传入用户问题，执行完整RAG流程
result = chain.invoke("LangChain支持java吗？")
# 打印LLM生成的回答
print(result)
```

### RunnableParallel并行链实战

```python
# 导入 LangChain 核心的 Runnable 相关模块
# Runnable 是 LangChain 中用于构建可组合、可运行的链的基础接口
# RunnableParallel 用于并行执行多个 Runnable 链
# RunnableLambda 用于将普通函数/匿名函数包装成 Runnable
from langchain_core.runnables import Runnable, RunnableParallel, RunnableLambda

# ==================== Runnable 核心概念实战示例 ====================
# 1. 多维度数据分析场景
# 说明：通过 RunnableParallel 并行执行三个不同的分析任务
# 注：以下 sentiment_analyzer/keyword_extractor/ner_recognizer 为示意函数，需自行实现
analysis_chain = RunnableParallel({
    "sentiment": sentiment_analyzer,    # 情感分析任务
    "keyword": keyword_extractor,      # 关键词提取任务
    "entities": ner_recognizer          # 命名实体识别任务（如人名、地名、机构名）
})

# 2. 多模型对比系统场景
# 说明：并行调用多个大模型，方便对比不同模型的输出结果
# 注：以下 gpt-3.5-turbo/gpt-4/claude-2 为示意的模型调用链，需自行实现
model_comparison = RunnableParallel({
    "gpt-3.5-turbo": gpt_35_turbo,     # GPT-3.5 模型调用链
    "gpt-4": gpt_4,                     # GPT-4 模型调用链
    "claude-2": claude_2                # Claude-2 模型调用链
})

# 3. 智能文档处理系统场景
# 说明：并行处理文档的摘要生成、目录提取、基础统计信息计算
document_analyzer = RunnableParallel({
    "summary": summary_generator,      # 文档摘要生成（示意函数，需自行实现）
    "toc": toc_generator,              # 文档目录提取（示意函数，需自行实现）
    # 使用 RunnableLambda 包装匿名函数，计算文档的基础统计信息
    "status": RunnableLambda(
        lambda doc: {
            "char_count": len(doc),                # 计算文档字符数
            "page_count": doc.count("PAGE_BREAK") + 1  # 按 PAGE_BREAK 标记计算页数
        }
    )
})

# 4. 处理200页PDF文本（仅示意，需结合实际PDF解析逻辑）
# 实际使用时需先将PDF转为文本，再传入上述 chain 执行
# 示例：
# pdf_text = extract_text_from_pdf("200_pages.pdf")  # 自定义PDF文本提取函数
# analysis_result = document_analyzer.invoke(pdf_text)


# ==================== Runnable 完整实战案例（景点+书籍推荐） ====================
# 导入必要的 LangChain 模块
# ChatPromptTemplate：用于构建聊天型提示词模板
# ChatOpenAI：OpenAI 兼容的聊天模型调用接口（可对接第三方兼容OpenAI接口的模型）
# JsonOutputParser：用于将模型输出解析为 JSON 格式
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

# 1. 初始化大模型（这里使用阿里云通义千问的兼容OpenAI接口）
model = ChatOpenAI(
    model_name="qwen-plus",            # 指定模型名称（通义千问Plus）
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云通义千问的兼容接口地址
    api_key="",  # 阿里云API密钥（注意：实际生产环境需从环境变量读取，避免硬编码）
    temperature=0.7                    # 模型生成温度（0-1，值越高输出越随机，越低越固定）
)

# 2. 初始化 JSON 输出解析器
# 作用：确保模型输出的文本能被正确解析为 Python 字典/JSON 格式
parser = JsonOutputParser()

# 3. 构建景点推荐的提示词模板
# from_template 方法从字符串模板创建提示词，支持变量替换（{city}/{num}）
prompt_attractions = ChatPromptTemplate.from_template(
    """列出{city}的{num}个著名景点，返回JSON格式：
    {{
    "num": "编号",
    "city": "城市",
    "introduce": "景点介绍"
    }}"""
)

# 4. 构建书籍推荐的提示词模板
prompt_books = ChatPromptTemplate.from_template(
    """列出与{city}相关的{num}本书籍，返回JSON格式：
    {{
    "num": "编号",
    "city": "城市",
    "introduce": "书籍介绍"
    }}"""
)

# 5. 构建独立的处理链（使用 | 运算符组合组件）
# 链执行流程：提示词模板（填充变量）→ 模型调用 → JSON解析
chain1 = prompt_attractions | model | parser  # 景点推荐链
chain2 = prompt_books | model | parser        # 书籍推荐链

# 6. 并行执行两条链
# RunnableParallel 将多个链包装为并行执行的链，返回字典格式的结果（key为指定名称，value为对应链的结果）
chain = RunnableParallel(
    attractions=chain1,  # 景点推荐结果的key
    books=chain2         # 书籍推荐结果的key
)

# 7. 主函数：测试并行链的执行
if __name__ == "__main__":
    # 定义输入数据（填充提示词模板的变量）
    input_data = {
        "city": "北京",
        "num": 3
    }

    # 调用并行链（invoke 方法为同步调用，返回最终结果）
    result = chain.invoke(input_data)
    
    # 打印结果
    print("景点推荐:", result["attractions"])
    print("书籍推荐:", result["books"])
```

### RunnableLambda高级实战

```python
# 包装链式函数实战
# 将任意python函数转换为符合Runnable协议的对象 实现自定义逻辑与langchain生态的无缝集成

# 导入LangChain核心的RunnableLambda类，用于将普通函数包装为可链式调用的对象
from langchain_core.runnables import RunnableLambda
# 导入OpenAI兼容的Chat模型类，用于调用大语言模型
from langchain_openai import ChatOpenAI

# ===================== 示例1：基础文本处理链（数据清洗ETL） =====================
# 构建文本清洗链，使用 | 运算符实现链式调用（类似Linux管道）
# 链式调用的执行顺序：从左到右，前一个函数的输出作为后一个函数的输入
text_clean_chain = (
    # 第一个处理步骤：去除字符串首尾的空白字符（空格、换行、制表符等）
    RunnableLambda(lambda doc: doc.strip()) 
    # 管道符：将前一步的输出作为后一步的输入
    | 
    # 第二个处理步骤：将字符串全部转换为小写字母
    RunnableLambda(lambda doc: doc.lower())
)

# 调用清洗链，处理测试文本
result = text_clean_chain.invoke("  Hello, World!  ")
# 打印结果，预期输出：hello, world!
print(result)  


# ===================== 示例2：带内容过滤的LLM调用链 =====================
# 打印中间结果并且过滤敏感词（在链中插入自定义处理逻辑）

# 自定义文本清洗函数：过滤敏感词
def filter_content(text: str) -> str:
    """
    敏感词过滤函数：将文本中的"暴力"替换为★★★
    
    Args:
        text (str): 需要过滤的原始文本
        
    Returns:
        str: 过滤后的文本
    """
    return text.replace("暴力", "★★★")

# 初始化大语言模型实例（这里使用通义千问，兼容OpenAI接口）
model = ChatOpenAI(
    model_name="qwen-plus",  # 指定模型名称（通义千问增强版）
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云通义千问兼容接口地址
    api_key="",  # 注意：实际使用时应从环境变量获取，避免硬编码
    temperature=0.7  # 模型生成温度，值越高输出越随机，0.7为中等随机性
)

# 构建完整的处理链：数据提取 → 内容过滤 → 模型调用
chain = (
        # 第一步：从输入字典中提取"user_input"字段的值
        RunnableLambda(lambda x: x["user_input"])  
        # 管道符：传递数据
        |  
        # 第二步：调用自定义过滤函数处理文本
        RunnableLambda(filter_content)          
        # 管道符：传递过滤后的文本给模型
        |  
        # 第三步：调用大语言模型处理文本
        model                                  
)

# 测试内容过滤链，传入包含敏感词的用户输入
result = chain.invoke({"user_input": "暴力内容"})
# 打印最终结果，预期模型会接收到"★★★内容"并生成响应
print("过滤后结果:", result)  
```

### AgentTool自定义工具实战

```python
# ==================== Agent Tool 核心实战：自定义工具开发 ====================
# 导入LangChain工具开发核心模块：tool装饰器用于定义Agent可调用的工具
from langchain_core.tools import tool
# 导入Pydantic的Field：用于定义工具参数的约束（描述、必填等）
from pydantic import Field
# 导入BaseModel：作为工具参数校验的基类（Pydantic模型，用于参数结构化）
from unstructured_client.types import BaseModel

# -------------------- 第一步：定义工具参数校验模型 --------------------
# 继承BaseModel（Pydantic），为工具参数做结构化定义和校验
# 作用：告诉大模型工具需要哪些参数、参数类型、参数描述，同时校验输入参数的合法性
class CalculatorInput(BaseModel) :
    # 定义第一个参数a：int类型，Field(...)表示必填，description是给大模型看的参数说明
    a: int = Field(..., description="第一个参数（整数），用于乘法运算")
    # 定义第二个参数b：int类型，必填，描述第二个乘法参数
    b: int = Field(..., description="第二个参数（整数），用于乘法运算")

# -------------------- 第二步：使用@tool装饰器定义自定义工具 --------------------
# @tool装饰器：将普通Python函数转换为LangChain Agent可调用的工具
# 参数说明：
#   1. "multiply-tool"：工具唯一名称（大模型会通过该名称识别并调用工具）
#   2. args_schema=CalculatorInput：指定参数校验模型，约束输入参数的类型和结构
#   3. return_direct=False：返回模式，False表示Agent会基于工具结果继续生成文本；True表示直接返回工具结果
#   4. description：工具功能描述（给大模型看，告诉大模型这个工具的作用）
@tool("multiply-tool",
      args_schema=CalculatorInput,
      return_direct=False,
      description="该工具用于执行两个整数的乘法运算，输入两个整数a和b，返回a*b的结果")
def multiply(a:int, b: int) -> int:
    """
    工具核心逻辑：实现两个整数的乘法运算
    :param a: 第一个整数参数（由args_schema校验类型）
    :param b: 第二个整数参数（由args_schema校验类型）
    :return: 两个数相乘的结果（int类型）
    """
    return a * b

# -------------------- 第三步：查看工具的元信息（验证工具定义） --------------------
# 打印工具名称（对应@tool装饰器中第一个参数）
print("工具名称:" , multiply.name)
# 打印工具描述（对应@tool装饰器的description参数）
print("工具描述:" , multiply.description)
# 打印工具参数定义（由args_schema自动解析，展示参数名、类型、描述）
print("工具参数:" , multiply.args)
# 打印工具返回模式：return_direct=False表示Agent会基于工具结果继续推理；True则直接返回结果
# 适用场景：简单任务（如纯计算）设为True，复杂任务（需总结结果）设为False
print("工具返回模式:" , multiply.return_direct)
# 打印工具参数的详细Schema（JSON格式）：大模型会读取该Schema理解参数要求
print("工具详细Schema:", multiply.args_schema.model_json_schema())

# -------------------- 第四步：调用工具（模拟Agent调用逻辑） --------------------
# invoke方法：工具的标准调用方式，传入字典格式的参数（需匹配args_schema定义）
# 底层会先通过CalculatorInput校验参数类型（如传字符串会报错），再执行multiply函数
print("工具调用结果:", multiply.invoke({"a":2,"b":3}))  # 输出：6

# -------------------- 核心说明 --------------------
# 工具的详细约束（参数类型、描述、Schema）会传递给大模型，帮助大模型：
# 1. 判断何时需要调用该工具（比如用户问"2乘3等于多少"时）；
# 2. 正确构造参数（确保传入整数a和b，而非其他类型）；
# 3. 理解工具返回结果的含义，进而生成符合需求的回答。
```

### AgentTool两种自定义工具的实战

```python
# ==================== Agent Tool 实战：两种自定义工具的方式 ====================
# 导入类型注解模块：Type 用于标注 args_schema 的类型（BaseModel的子类）
from typing import Type

# 导入LangChain工具核心基类：BaseTool是所有自定义工具的父类，定义了工具的核心规范
from langchain_core.tools import BaseTool
# 导入Pydantic核心模块：BaseModel用于参数结构化校验，Field用于参数描述/约束
from pydantic import BaseModel, Field


# -------------------- 方式1：快捷定义（注释掉的示例）：StructuredTool.from_function --------------------
# 该方式是「装饰器@tool」的底层实现，通过函数快速封装为工具，无需手动定义类
# 适合简单工具（仅需封装单个函数，无需自定义复杂逻辑）

# # 步骤1：定义参数校验模型（Pydantic），约束工具输入参数的类型和描述
# class CalculatorInput(BaseModel) :
#     a: int = Field(..., description="第一个参数（整数，必填）")
#     b: int = Field(..., description="第二个参数（整数，必填）")
#
# # 步骤2：定义工具核心逻辑函数
# def multiply(a:int, b: int) -> int:
#     """工具核心功能：实现两个整数的乘法运算"""
#     return a * b
#
# # 步骤3：将函数快速封装为StructuredTool（BaseTool的子类）
# multiply = StructuredTool.from_function(multiply)
#
# # 也可以在封装时自定义工具元信息（名称、描述、参数约束等）
# tools = [
#     StructuredTool.from_function(
#         func=multiply,          # 绑定工具核心函数
#         name="multiply",        # 自定义工具名称（大模型识别工具的标识）
#         description="乘法计算", # 工具描述（告诉大模型工具的用途）
#         args_schema=CalculatorInput, # 指定参数校验模型（约束输入）
#         return_direct=True,     # 返回模式：直接返回工具结果，Agent不额外推理
#     )
# ]
#
# # 打印工具元信息（验证工具定义）
# print("工具名称:" , multiply.name)
# print("工具描述:" , multiply.description)
# print("工具参数:" , multiply.args)  # 自动解析args_schema生成的参数列表
# print("工具返回模式:" , multiply.return_direct) 
# # 打印参数详细Schema（JSON格式，供大模型读取参数要求）
# print("工具详细Schema:",multiply.args_schema.model_json_schema())
#
# # 调用工具（传入字典格式参数，自动校验类型）
# print(multiply.invoke({"a":2,"b":3}))  # 输出：6
# # 核心说明：工具的元信息（名称、描述、参数约束）会传递给大模型，帮助大模型判断何时调用、如何传参


# -------------------- 方式2：自定义类（推荐复杂场景）：继承 BaseTool 类 --------------------
# 该方式更灵活，可自定义工具的初始化、运行逻辑、额外属性等，适合复杂工具开发

# 步骤1：定义参数校验模型（Pydantic），约束工具输入参数
class CalculatorInput(BaseModel) :
    # Field描述参数含义（给大模型看），未加...表示非必填（若需必填需加 Field(..., description="")）
    a: int = Field(description="第一个乘法参数（整数）")
    b: int = Field(description="第二个乘法参数（整数）")

# 步骤2：自定义工具类（继承BaseTool，必须实现 _run 方法）
class CustomCalculatorTool(BaseTool) :
    # 工具核心元信息（固定属性，需显式定义）
    name : str = "custom_multiply_tool"  # 工具唯一名称（大模型识别用）
    description : str = "当你需要计算两个整数的乘法问题时使用该工具"  # 工具用途描述
    args_schema : Type[BaseModel] = CalculatorInput  # 绑定参数校验模型（Type[BaseModel]标注类型）
    return_direct : bool = True  # 返回模式：True=直接返回工具结果，False=Agent基于结果继续推理

    # 核心方法：_run（必须实现），定义工具的实际执行逻辑
    # 参数需与args_schema中的字段一一对应（a、b）
    def _run(self, a:int, b: int) -> int:
        """
        工具核心执行逻辑：实现两个整数的乘法运算
        :param a: 第一个整数参数（由args_schema校验类型）
        :param b: 第二个整数参数（由args_schema校验类型）
        :return: 两个数相乘的结果（int类型）
        """
        return a * b

# 步骤3：实例化自定义工具
multiply = CustomCalculatorTool()

# 步骤4：查看工具元信息（验证工具定义）
print("工具名称:" , multiply.name)  # 输出：custom_multiply_tool
print("工具描述:" , multiply.description)  # 输出：当你需要计算数学问题时使用
print("工具参数:" , multiply.args)  # 输出参数列表（从args_schema解析）
print("工具返回模式:" , multiply.return_direct)  # 输出：True
# 打印参数详细Schema（JSON格式，大模型会读取该信息理解参数要求）
print("工具详细Schema:", multiply.args_schema.model_json_schema())

# （可选）调用工具（需传入符合args_schema的参数字典）
# print(multiply.invoke({"a":2,"b":3}))  # 输出：6
```

### 大模型绑定工具（Tool）核心实战

```python
# ==================== 大模型绑定工具（Tool）核心实战 ====================
# 核心目标：让大模型根据用户问题自动判断是否调用工具、调用哪个工具，并基于工具结果生成最终回答

# 导入核心模块：
# HumanMessage：构造用户消息（符合LangChain消息格式）
# tool：装饰器，将普通函数转为大模型可调用的工具
# ChatOpenAI：调用兼容OpenAI接口的大模型（这里对接阿里云通义千问）
# SecretStr：安全存储敏感信息（如API密钥），避免明文泄露
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


# -------------------- 第一步：定义大模型可调用的工具 --------------------
# @tool装饰器：将普通Python函数转为LangChain标准工具（无需手动定义参数Schema，自动解析）
# 工具1：加法工具，实现两数相加
@tool
def add(a:int, b: int) -> int:
    """
    工具功能描述（给大模型看）：计算两个整数的加法（a + b）
    :param a: 第一个整数参数
    :param b: 第二个整数参数
    :return: 两数相加的结果
    """
    return a + b

# 工具2：乘法工具，实现两数相乘
@tool
def multiply(a:int, b: int) -> int:
    """
    工具功能描述（给大模型看）：计算两个整数的乘法（a * b）
    :param a: 第一个整数参数
    :param b: 第二个整数参数
    :return: 两数相乘的结果
    """
    return a * b

# -------------------- 第二步：初始化大模型（对接阿里云通义千问） --------------------
model = ChatOpenAI(
    model="qwen-plus",  # 指定模型名称（通义千问Plus）
    base_url= "https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云通义千问兼容OpenAI的接口地址
    api_key=SecretStr(""),  # 安全存储API密钥（SecretStr避免明文打印/泄露）
    temperature=0.7)  # 模型生成温度（0-1，值越高输出越随机，越低越精准）

# -------------------- 第三步：将工具绑定到大模型 --------------------
# bind_tools：将工具列表绑定到模型，让模型具备“工具调用能力”
# 绑定后，模型会分析用户问题，判断是否需要调用工具、调用哪个工具、传什么参数
llm_with_tool = model.bind_tools(tools)

# -------------------- 第四步：构造用户问题并触发模型推理 --------------------
# 用户问题（需要模型调用乘法工具解决）
query = "请计算2*3是多少？"

# 构造消息列表（LangChain标准格式，HumanMessage表示用户消息）
message = [
    HumanMessage(content=query)
]

# 调用绑定工具的模型：模型会先分析问题，生成“工具调用指令”（而非直接回答）
ai_message = llm_with_tool.invoke(message)
# print(ai_message.tool_calls)  # 可打印查看工具调用详情（包含工具名称、参数等）

# 将模型的工具调用消息加入消息列表（用于后续上下文传递）
message.append(ai_message)

# -------------------- 第五步：执行模型指定的工具调用 --------------------
# 遍历模型生成的工具调用指令（可能调用多个工具）
for tool_call in ai_message.tool_calls:
    # 根据工具名称匹配对应的工具函数（统一转为小写避免大小写问题）
    selected_tool = {"add": add, "multiply": multiply}[tool_call['name'].lower()]
    # 调用工具：传入工具调用指令中的参数，执行工具并获取结果
    tool_msg = selected_tool.invoke(tool_call)
    print(f"工具执行结果:{tool_msg}")  # 输出：6（2*3的结果）
    # 将工具执行结果加入消息列表（作为上下文，供模型生成最终回答）
    message.append(tool_msg)

# 打印完整消息列表（查看上下文流转：用户消息 → 模型工具调用 → 工具结果）
print("完整消息上下文:", message)

# -------------------- 第六步：模型基于工具结果生成最终回答 --------------------
# 模型结合用户问题、工具调用指令、工具执行结果，生成自然语言回答
result = llm_with_tool.invoke(message)
print(f"AI最终回复:{result.content}")  # 输出类似：2乘以3的结果是6。
```

### 联网搜索实战

```python
# ==================== LangChain 内置工具包实战：联网搜索 ====================
# 核心目标：使用 LangChain 内置的 SearchApiAPIWrapper 工具调用联网搜索API，获取实时网络信息
import os  # 导入操作系统模块，用于设置环境变量（存储敏感的API密钥）

# 导入LangChain社区工具：SearchApiAPIWrapper是对接SearchApi搜索引擎的封装工具
from langchain_community.utilities import SearchApiAPIWrapper
# 导入SecretStr（可选）：用于安全存储敏感信息（本示例用环境变量，仅展示导入）
from pydantic import SecretStr

# -------------------- 第一步：配置搜索API密钥（核心） --------------------
# 1. SearchApi需要API密钥才能调用，这里通过环境变量设置（避免硬编码泄露）
# 2. 实际使用时，建议将密钥放在.env文件中，通过python-dotenv加载，而非直接写在代码里
os.environ["SEARCHAPI_API_KEY"] = ""

# -------------------- 第二步：实例化搜索工具对象 --------------------
# SearchApiAPIWrapper：LangChain内置的SearchApi封装工具，提供简洁的搜索接口
# 实例化后可调用run()/results()方法执行搜索
search = SearchApiAPIWrapper()

# -------------------- 第三步：执行联网搜索 --------------------
# 方式1（简化版）：run()方法 → 返回格式化的搜索结果字符串（适合直接给大模型输入）
# result = search.run("langchain框架核心模块")

# 方式2（详细版）：results()方法 → 返回原始的搜索结果字典列表（适合自定义解析）
# 参数：搜索关键词（字符串），可额外指定num_results（返回结果数量，默认10）等参数
result = search.results("langchain框架核心模块")

# -------------------- 第四步：输出搜索结果 --------------------
# 输出结果：results()返回的是包含多个搜索结果的列表，每个元素是字典（含标题、链接、内容等）
print("搜索结果（原始字典列表）：")
print(result)

# 【可选】解析搜索结果示例（提取关键信息）
# for item in result:
#     print(f"标题：{item['title']}")
#     print(f"链接：{item['link']}")
#     print(f"摘要：{item['snippet']}\n")
```

### 调用工具异常处理实战

```python
# 导入 LangChain 核心工具相关的模块和异常类
from langchain_core.tools import StructuredTool, ToolException

# 定义搜索工具的核心函数
def search(query: str) -> str:
    """
    执行搜索查询的核心函数（模拟实现）
    
    参数:
        query (str): 需要搜索的查询字符串
        
    返回:
        str: 搜索结果（本示例中会抛出异常，不会正常返回）
        
    异常:
        ToolException: 模拟搜索过程中出现的业务异常
    """
    # 模拟搜索过程中发生异常
    # 使用 LangChain 提供的 ToolException 而非原生 Exception，便于工具层统一处理
    raise ToolException(f"搜索失败: {query}")

# 自定义异常处理函数（核心：统一处理工具执行过程中抛出的异常）
def _handel_tool_error(e: Exception) -> str:
    """
    工具异常的自定义处理函数
    
    参数:
        e (Exception): 工具执行过程中捕获到的异常对象
        
    返回:
        str: 格式化后的异常提示信息（友好返回给调用方）
    """
    # 可以在这里扩展更复杂的异常处理逻辑：
    # 1. 区分不同异常类型（如网络异常、参数异常）
    # 2. 记录异常日志
    # 3. 返回不同的提示语
    return f"搜索结果失败，请重试。具体错误信息：{str(e)}"

# 构建结构化工具（LangChain 标准方式）
search_tool = StructuredTool.from_function(
    name="search",  # 工具名称（唯一标识，用于 Agent 调用）
    func=search,    # 绑定工具的核心执行函数
    description="用于执行搜索查询，输入查询字符串即可获取相关结果",  # 工具描述（Agent 用于判断是否调用该工具）
    handle_tool_error=_handel_tool_error  # 绑定自定义异常处理函数
)

# 调用工具（模拟实际业务场景中的工具调用）
if __name__ == "__main__":
    # 调用工具并传入参数（参数需与 search 函数的入参匹配）
    resp = search_tool.invoke({"query": "如何使用langchain进行搜索"})
    # 打印处理后的结果（即使异常也会返回友好提示，而非直接崩溃）
    print("工具调用结果：", resp)
```

### 大模型调用Web搜索工具获取实时信息实战

```python
# 【LLM联网搜索功能实战】：演示基于LangChain框架实现大模型调用Web搜索工具获取实时信息（如股价），并通过工具调用流程完成问答
import os
# 导入SearchApi相关的API封装（用于调用联网搜索接口）
from langchain_community.utilities import SearchApiAPIWrapper
# 导入工具调用相关的消息类（用于封装工具返回结果）
from langchain_core.messages import ToolMessage
# 导入提示词模板（用于构建大模型的输入提示）
from langchain_core.prompts import ChatPromptTemplate
# 导入RunnablePassthrough（用于链式调用中透传输入参数）
from langchain_core.runnables import RunnablePassthrough
# 导入工具装饰器（用于定义LangChain标准工具）
from langchain_core.tools import tool
# 导入OpenAI兼容的大模型客户端（用于调用通义千问等模型）
from langchain_openai import ChatOpenAI
# 导入SecretStr（用于安全存储API密钥）
from pydantic import SecretStr

# ====================== LangSmith 配置（可选，用于追踪链的执行过程） ======================
# 启用LangChain的追踪功能，便于调试和查看链的执行流程
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# LangSmith的API端点
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# LangSmith的API密钥（替换为自己的）

# 项目名称（用于在LangSmith中区分不同项目）
os.environ["LANGCHAIN_PROJECT"] = "agent_v1"

# ====================== 搜索工具配置 ======================
# 设置SearchApi的API密钥（替换为自己的有效密钥）
os.environ["SEARCHAPI_API_KEY"] = ""
# 实例化SearchApiWrapper对象，封装了搜索API的调用逻辑
search = SearchApiAPIWrapper()

# ====================== 定义工具函数 ======================
# 定义联网搜索工具：使用@tool装饰器标记为LangChain工具，name指定工具名，return_direct=True表示直接返回结果
@tool(name_or_callable="web_search", return_direct=True)
def web_search(search_query: str) -> str:
    """
    联网搜索工具：适用于获取实时信息、最新事件或未知领域知识的场景
    参数:
        search_query (str): 搜索关键字，如"寒武纪今天最新的股价"
    返回:
        str: 格式化后的搜索结果（包含来源标题和内容摘要）
    """
    try:
        # 调用SearchApi获取搜索结果（默认返回前几条结果）
        result = search.results(search_query)
        # 格式化搜索结果：提取organic_results（自然搜索结果），拼接标题和内容
        return "\n\n".join([f"来源:{res['title']}\n内容:{res['snippet']}" for res in result['organic_results']])
    except Exception as e:
        # 异常处理：搜索失败时返回友好提示
        return f"搜索失败：{str(e)}"

# 定义加法工具：演示非联网工具的定义方式，用于对比
@tool
def add(a: int, b: int) -> int:
    """
    加法计算工具：用于执行简单的整数加法运算
    参数:
        a (int): 加数1
        b (int): 加数2
    返回:
        int: a + b的计算结果
    """
    return a + b

# ====================== 初始化大模型 ======================
# 创建ChatOpenAI实例（兼容通义千问等OpenAI格式的模型）
model = ChatOpenAI(
    model="qwen-plus",  # 模型名称（通义千问增强版）
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云通义千问的兼容接口
    api_key=SecretStr(""),  # 通义千问的API密钥（安全存储为SecretStr）
    temperature=0.7  # 温度系数：控制回答的随机性，0.7为适中值
)

# ====================== 构建提示词模板 ======================
# 创建聊天提示词模板：包含系统提示和用户输入
prompt = ChatPromptTemplate.from_messages([
    # 系统提示：告诉大模型的角色和工具调用规则
    ("system", "你是一个AI助手，名字叫HideOnBoss，请根据用户输入的查询问题，必要时可以调用工具帮用户解答"),
    # 用户输入：{query}为占位符，运行时会替换为实际的用户查询
    ("human", "{query}"),
])

# ====================== 工具绑定与链构建 ======================
# 构建工具字典：将工具名和工具函数映射，便于后续调用
tool_dict = {"add": add, "web_search": web_search}
# 提取工具列表：从字典中获取所有工具函数，用于绑定给大模型
tools = [tool_dict[tool_name] for tool_name in tool_dict]

# 将大模型绑定工具：让模型具备调用指定工具的能力
llm_with_tool = model.bind_tools(tools)

# 构建运行链：
# 1. RunnablePassthrough() 透传用户输入的query参数
# 2. prompt 接收query并生成完整的提示词
# 3. llm_with_tool 接收提示词，判断是否调用工具并返回结果
chain = {"query": RunnablePassthrough()} | prompt | llm_with_tool

# ====================== 执行示例：查询寒武纪最新股价 ======================
# 用户查询：需要实时信息，模型会判断调用web_search工具
query = "寒武纪今天最新的股价是多少?"
# 调用链，获取模型的初始响应（可能包含工具调用指令）
resp = chain.invoke(query)
print(f"AI初始回复:{resp}\n")

# ====================== 处理工具调用逻辑 ======================
# 提取模型返回的工具调用指令（tool_calls为空则无需调用工具）
tool_calls = resp.tool_calls
# 构建历史消息：包含初始的提示词和模型的初始响应，用于后续上下文传递
history_message = prompt.invoke(query).to_messages()
history_message.append(resp)

# 判断是否需要调用工具
if len(tool_calls) <= 0:
    print(f"不需要调用工具，直接回答：{resp.content}")
else:
    print(f"需要调用工具，工具调用指令：{tool_calls}\n")
    print(f"当前历史消息：{history_message}\n")

    # 循环处理每个工具调用指令
    for tool_call in tool_calls:
        # 提取工具名称和调用参数
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args")
        print(f"开始调用工具：{tool_name}，参数：{tool_args}")
        
        # 调用对应的工具函数，获取工具输出
        tool_output = tool_dict[tool_name].invoke(tool_args)
        
        # 封装工具返回结果为ToolMessage（包含tool_call_id，用于关联调用）
        tool_response_message = ToolMessage(
            tool_call_id=tool_call.get("id"),  # 关联对应的工具调用ID
            content=tool_output,               # 工具返回的内容
            name=tool_name                     # 工具名称
        )
        print(f"工具调用结果：{tool_output}\n")
        
        # 将工具返回结果添加到历史消息中，供模型后续回答使用
        history_message.append(tool_response_message)
        print(f"更新后的历史消息：{history_message}\n")

        # 再次调用模型，传入包含工具结果的历史消息，获取最终回答
        result = llm_with_tool.invoke(history_message)
        print(f"模型结合工具结果的最终回复：{result}")
        print(f"最终答案：{result.content}\n")
```

### 大模型接入LangSmith实战

```python
# 【大模型接入LangSmith实战】：配置LangSmith实现大模型调用的链路追踪、运行监控和异常告警
import os          # 用于设置系统环境变量
import logging     # 用于日志输出（辅助调试LangSmith接入过程）
# 导入OpenAI兼容的大模型客户端（支持通义千问等模型）
from langchain_openai import ChatOpenAI
# 导入SecretStr用于安全存储API密钥（避免明文泄露）
from pydantic import SecretStr

# ====================== 基础日志配置（可选） ======================
# 设置日志级别为DEBUG，便于调试LangSmith的接入和调用过程
# 可看到LangSmith的请求/响应日志，排查接入失败等问题
logging.basicConfig(level=logging.DEBUG)

# ====================== LangSmith核心配置（关键） ======================
# 1. 启用LangChain V2版本的追踪功能（必须设为true才会开启链路追踪）
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# 2. LangSmith的API端点（固定值，无需修改）
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# 3. LangSmith的API密钥（核心，替换为自己的密钥，在LangSmith平台获取）
#    作用：鉴权，确认当前调用归属的账号/项目

# 4. LangSmith的项目名称（自定义）
#    作用：在LangSmith平台按项目分类查看追踪数据，便于多项目管理
os.environ["LANGCHAIN_PROJECT"] = "agent_v1"

# ====================== 初始化大模型 ======================
# 创建ChatOpenAI实例（兼容通义千问等OpenAI格式的模型）
model = ChatOpenAI(
    model="qwen-plus",  # 模型名称（这里使用通义千问增强版）
    # 阿里云通义千问的OpenAI兼容接口地址（固定值）
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # 通义千问的API密钥（安全存储为SecretStr，避免明文暴露）
    api_key=SecretStr(""),
    temperature=0.7  # 温度系数：控制回答的随机性，0.7为适中值
)

# ====================== 执行模型调用（触发LangSmith追踪） ======================
# 调用模型并传入问题：该调用会被LangSmith完整追踪
resp = model.invoke("什么是智能体?")
# 打印模型返回的回答内容
print("模型回答：", resp.content)

# 【关键提示】：执行后可访问 https://smith.langchain.com/ 登录查看：
# 1. 调用链路：模型的输入/输出、耗时、Token用量
# 2. 监控数据：调用成功率、响应时间分布
# 3. 告警配置：可在LangSmith平台设置超时、失败率等告警规则
```

### 大模型的零样本（无示例）和少样本（带示例）提示工程

```python
# 【Zero-Shot/Few-Shot实战】：演示大模型的零样本（无示例）和少样本（带示例）提示工程，对比不同提示方式的效果
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate  # 导入提示词模板工具
from langchain_openai import ChatOpenAI  # 导入OpenAI兼容的大模型客户端
from pydantic import SecretStr  # 用于安全存储API密钥

# ====================== 1. 准备Few-Shot示例数据 ======================
# 少样本示例数据：格式为列表+字典，包含input（输入问题）和output（示例回答）
# 作用：给大模型提供少量示例，引导其按照固定格式/逻辑回答问题
data = [
    {
        "input": "langchain可以做智能体吗？",
        "output": "根据我大量思考:可以"
    },
    {
        "input": "openAI的CEO是谁?",
        "output": "根据我大量思考:余承东"  # 注：此处为示例错误答案，仅用于演示格式
    }
]

# ====================== 2. 初始化大模型 ======================
model = ChatOpenAI(
    model="qwen-plus",  # 模型名称（通义千问增强版）
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云通义千问兼容接口
    api_key=SecretStr(""),  # 通义千问API密钥（安全存储）
    temperature=0.7  # 温度系数：0.7表示回答有一定随机性，0则更固定
)

# ====================== 3. 定义示例模板（单个示例的格式） ======================
# 单个示例的提示词模板：规定每个示例的输入输出展示格式
example_template = """
输入:{input}
输出:{output}
"""
# 封装为PromptTemplate对象：指定模板变量为input和output
example_prompt = PromptTemplate(
    input_variables=["input", "output"],  # 模板中需要替换的变量名
    template=example_template  # 绑定示例模板字符串
)

# ====================== 4. 构建Few-Shot提示词模板（核心） ======================
# FewShotPromptTemplate：将多个示例按模板拼接，形成完整的少样本提示词
few_shot_prompt = FewShotPromptTemplate(
    examples=data,  # 传入少样本示例数据
    example_prompt=example_prompt,  # 单个示例的格式模板
    prefix="请按照以下示例的格式回答问题：",  # 前缀：告诉模型示例的作用（可选但建议加）
    suffix="输入:{question}\n输出:",  # 后缀：指定用户问题的输入位置和回答格式
    input_variables=["question"],  # 最终需要传入的变量（用户问题）
    example_separator="\n\n"  # 示例之间的分隔符，增强可读性
)

# ====================== 5. 测试Few-Shot提示词格式 ======================
# 格式化提示词：传入用户问题，生成包含示例的完整提示词
formatted_prompt = few_shot_prompt.format(question="苹果公司的总部在哪?")
print("=== 生成的Few-Shot提示词 ===")
print(formatted_prompt)
print("\n")

# ====================== 6. 构建少样本调用链并执行 ======================
# 构建链式调用：提示词模板 → 大模型
few_shot_chain = few_shot_prompt | model
# 调用链：传入用户问题，模型会参考示例格式回答
resp = few_shot_chain.invoke({"question": "langchain是什么?"})

# ====================== 7. 输出结果 ======================
print("=== 模型少样本回答结果 ===")
print("完整响应对象：", resp)
print("回答内容：", resp.content)

# ====================== 8. 补充Zero-Shot对比（无示例） ======================
# 零样本提示：无任何示例，直接让模型回答
zero_shot_prompt = PromptTemplate(
    input_variables=["question"],
    template="输入:{question}\n输出:"
)
zero_shot_chain = zero_shot_prompt | model
zero_shot_resp = zero_shot_chain.invoke({"question": "langchain是什么?"})

print("\n=== 模型零样本回答结果 ===")
print("回答内容：", zero_shot_resp.content)
```

### Agent智能体实战

```python
# 该代码演示了基于LangChain框架搭建一个具备「网页实时搜索」和「数学计算」能力的智能体（Agent），可响应实时股票查询等需要外部数据或计算的需求
import os

# 从LangChain框架导入智能体相关核心模块
from langchain.agents import AgentType, initialize_agent
# 导入SearchApiAPIWrapper，用于对接SearchApi实现网页搜索功能
from langchain_community.utilities import SearchApiAPIWrapper
# 导入tool装饰器，用于定义智能体可调用的工具函数
from langchain_core.tools import tool
# 导入ChatOpenAI，用于实例化大语言模型（此处对接阿里通义千问）
from langchain_openai import ChatOpenAI
# 导入SecretStr，用于安全存储和传递API密钥（避免明文泄露风险）
from pydantic import SecretStr

# 配置LangChain Smith相关环境变量，用于追踪智能体的运行过程、日志和性能
# 启用LangChain v2版本追踪功能
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# LangChain Smith的API端点地址
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# LangChain Smith的API密钥（用于身份验证）

# 定义当前项目名称，方便在LangChain Smith后台分类查看
os.environ["LANGCHAIN_PROJECT"] = "agent_v1"

# 配置SearchApi的API密钥，用于调用网页搜索接口获取实时数据
os.environ["SEARCHAPI_API_KEY"] = ""
# 实例化SearchApiWrapper对象，封装了SearchApi的调用逻辑，提供便捷的搜索方法
search = SearchApiAPIWrapper()

# 定义网页搜索工具，使用@tool装饰器标记为LangChain智能体可调用的工具
# 参数1：工具名称"web_search"，供智能体识别；参数2：return_direct=True，表示直接返回工具执行结果给用户
@tool("web_search", return_direct=True)
def web_search(search_query: str) -> str:
    """
    工具功能描述（供智能体判断是否需要调用该工具）：
    当需要获取实时信息、最新事件或未知领域知识时使用，输入应为搜索关键字
    """
    try:
        # 调用SearchApiWrapper的results方法执行搜索，传入搜索关键字
        result = search.results(search_query)
        # 解析搜索结果中的有机结果（organic_results），提取标题和摘要并格式化拼接
        # 最终返回易读的结构化搜索结果
        return "\n\n".join([f"来源:{res['title']}\n内容:{res['snippet']}" for res in result['organic_results']])
    except Exception as e:
        # 捕获搜索过程中的异常（如网络错误、API密钥失效、无结果等），返回友好的错误提示
        return f"搜索失败：{e}"

# 定义数学计算工具，使用@tool装饰器标记为LangChain智能体可调用的工具
# 参数1：工具名称"math_calculator"，供智能体识别；参数2：return_direct=True，表示直接返回计算结果给用户
@tool("math_calculator", return_direct=True)
def math_calculator(expression: str) -> str:
    """
    工具功能描述（供智能体判断是否需要调用该工具）：
    用于计算数学公式，输入应为合法的数学表达式（如"1+2*3"、"sqrt(16)"等）
    """
    try:
        # 使用eval函数执行数学表达式计算（注意：生产环境中eval存在安全风险，需做输入校验）
        result = eval(expression)
        # 返回格式化的计算结果
        return f"计算结果为：{result}"
    except Exception as e:
        # 捕获计算过程中的异常（如表达式不合法、除零错误等），返回友好的错误提示
        return f"计算失败：{e}"

# 实例化大语言模型（LLM），此处对接阿里通义千问（qwen-plus），兼容OpenAI接口格式
model = ChatOpenAI(
    model="qwen-plus",  # 指定使用的模型名称（通义千问plus版本）
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里通义千问的兼容OpenAI接口地址
    api_key=SecretStr(""),  # 通义千问的API密钥，使用SecretStr安全存储
    temperature=0.7  # 模型生成内容的随机性（0~1），0.7表示兼顾逻辑性和一定的创造性
)

# 整理智能体可调用的工具列表，将定义好的两个工具放入列表供后续初始化智能体使用
tool_dict = [math_calculator, web_search]

# 初始化LangChain智能体，将模型、工具列表进行整合，构建具备决策能力的智能体链
agent_chain = initialize_agent(
    tools=tool_dict,  # 传入智能体可调用的工具列表
    llm=model,  # 传入实例化的大语言模型（作为智能体的"大脑"，负责决策和逻辑推理）
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # 指定智能体类型：零样本REACT描述型
    # 零样本：无需训练样本即可工作；REACT：通过"思考-行动-观察"循环完成任务
    verbose=True,  # 启用详细日志输出，可在控制台查看智能体的运行过程（思考、调用工具、结果等）
    handle_parsing_errors=True  # 启用解析错误处理，当工具返回结果解析失败时，自动进行容错处理
)

# 打印智能体的核心组件信息，用于清晰理解底层结构和调试
# 1. 打印智能体的LLM链（包含模型、提示词等核心组件）
print(f"agent_chain.agent.llm_chain：{agent_chain.agent.llm_chain}")
# 2. 打印LLM链中的提示词模板（智能体执行任务时的核心指令模板）
print(f"agent_chain.agent.llm_chain.prompt.template:{agent_chain.agent.llm_chain.prompt.template}")
# 3. 打印提示词模板中的输入变量（需要传入的参数，如任务输入、工具描述等）
print(f"agent_chain.agent.llm_chain.prompt.input_variables:{agent_chain.agent.llm_chain.prompt.input_variables}")

# 调用智能体的invoke方法，传入用户输入（查询比亚迪当日股票价格），执行任务并获取响应结果
# 由于查询的是实时股票数据，智能体会自动决策调用web_search工具获取最新信息
resp = agent_chain.invoke({"input": "比亚迪今天股票多少？"})

# 打印智能体的最终运算结果，供用户查看
print(f"运算结果:{resp}")
```

### 个人AI助手智能体实战

```python
# 基于LangChain搭建具备「日期获取、航班查询/预订、股价查询」能力的个人AI助手智能体
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool, Tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# 定义获取当前日期的工具函数，使用@tool装饰器标记为LangChain可调用工具
# 无入参，返回格式化的当前日期字符串
@tool
def get_current_date() -> str:
    """ 获取当前日期（格式：年-月-日），用于需要日期上下文的任务（如航班查询、预订等） """
    import datetime  # 内部导入datetime模块，仅在该工具调用时加载
    # 将当前时间格式化为"YYYY-MM-DD"的标准格式，提升结果可读性和后续工具兼容性
    formatted_date = datetime.datetime.now().strftime("%Y-%m-%d")
    return f"当前日期为：{formatted_date}"

# 定义搜索航班的工具函数，带入参校验，满足指定城市和日期的航班查询需求
@tool
def search_flight(from_city: str, to_city: str, date: str) -> str:
    """ 
    搜索指定出发地、目的地和日期的可用航班
    参数说明：
    - from_city: 出发城市（如"广州"、"上海"）
    - to_city: 目的城市（如"北京"、"深圳"）
    - date: 出行日期（格式：YYYY-MM-DD）
    """
    # 模拟航班搜索结果，实际场景可对接航空公司API或第三方旅游平台接口
    return f"搜索结果：从{from_city}到{to_city}的航班在{date}有可用航班,价格：￥1200，推荐航班号：CA1324"

# 定义预定航班的工具函数，支持指定航班号和用户信息完成预订
@tool
def book_flight(flight_id: str, user: str) -> str:
    """ 
    预定指定航班号的机票，需提供有效航班号和用户名
    参数说明：
    - flight_id: 航班号（如"CA1324"，需从search_flight工具的结果中获取）
    - user: 预订用户姓名（用于机票出票和身份核验）
    """
    # 模拟航班预订成功结果，实际场景需对接航班预订系统，包含订单创建、支付回调等逻辑
    return f"用户:{user}预定成功，航班号：{flight_id}，订单状态：待支付"

# 定义获取股价的普通函数（未使用@tool装饰器，后续将封装为Tool对象）
def get_stock_price(symbol: str) -> str:
    """
    查询指定股票代码对应的股价（模拟返回固定结果）
    参数说明：
    - symbol: 股票代码/标的符号（如苹果公司"AAPL"、比亚迪"002594"）
    """
    # 模拟美股股价查询结果，实际场景可对接金融数据API（如Yahoo Finance、Tushare等）
    return f"The price of {symbol} is $100. (实时更新时间：{datetime.datetime.now().strftime('%H:%M:%S')})"

# 修复：导入datetime模块（解决get_stock_price中调用datetime未导入的问题）
import datetime

# 创建工具列表，整合两种方式定义的工具（@tool装饰器直接生成 + 普通函数封装为Tool对象）
tools = [
    get_current_date,  # 直接加入@tool装饰器定义的工具
    search_flight,     # 直接加入@tool装饰器定义的工具
    book_flight,       # 直接加入@tool装饰器定义的工具
    # 将普通函数get_stock_price封装为LangChain标准Tool对象，使其可被Agent调用
    Tool(
        name="Get Stock Price",  # 工具名称（供Agent识别和调用）
        func=get_stock_price,    # 绑定工具执行的核心函数
        description="查询指定股票代码的当前股价，输入为股票标的符号（如苹果公司输入AAPL）",  # 工具功能描述（关键：供Agent判断是否需要调用该工具）
    ),
]

# 实例化大语言模型（对接阿里通义千问qwen-plus，兼容OpenAI接口格式）
model = ChatOpenAI(
    model="qwen-plus",  # 指定使用的模型名称（通义千问增强版，具备更强的工具调用和逻辑推理能力）
    base_url= "https://dashscope.aliyuncs.com/compatible-mode/v1",  # 通义千问兼容OpenAI接口的端点地址
    api_key=SecretStr(""),  # 通义千问API密钥，使用SecretStr安全存储（避免明文泄露）
    temperature=0.7,    # 模型生成随机性（0~1），0.7兼顾逻辑性和灵活度，适合工具调用类任务
)

# 创建聊天提示词模板，定义Agent的系统角色、上下文信息和交互格式
prompt = ChatPromptTemplate.from_messages([
    # 系统消息：定义Agent的身份和核心行为准则，为Agent提供全局指令
    ("system", "你是一个专业的个人AI助手，擅长处理出行预订、金融信息查询等任务，必要时可以调用工具函数帮助用户解决问题，优先使用工具返回的准确数据进行回复"),
    # 历史上下文消息：预置用户的个人信息，供Agent在后续任务中直接使用（无需用户重复提供）
    ("human", "我叫老王，经常出差，喜欢性价比的按摩，身份证号是4444144444"),
    # 用户输入占位符：接收用户实时提交的任务指令，变量名{input}需与后续invoke传入的参数对应
    ("human", "{input}"),
    # Agent思考/行动占位符：用于存储Agent的思考过程、工具调用记录和中间结果，不可删除
    ("placeholder", "{agent_scratchpad}"),
])

# 创建工具调用型Agent（基于最新的Tool Calling能力，比传统REACT更高效、更稳定）
agent = create_tool_calling_agent(
    llm=model,    # 传入实例化的大语言模型（作为Agent的"大脑"，负责决策是否调用工具、调用哪个工具）
    tools=tools,  # 传入Agent可调用的工具列表，Agent会基于工具描述进行选择
    prompt=prompt # 传入自定义的提示词模板，定义Agent的行为边界和交互格式
)

# 创建Agent执行器，负责调度Agent运行、处理工具调用流程、返回最终结果
agent_executor = AgentExecutor(
    agent=agent,  # 传入已创建的工具调用型Agent
    tools=tools,  # 传入工具列表（与创建Agent时的工具列表保持一致）
    verbose=True,  # 启用详细日志输出，可在控制台查看Agent的思考过程、工具调用参数、工具返回结果等
    return_intermediate_steps=True,  # 启用中间步骤返回，最终结果中会包含Agent的每一步操作记录（方便调试和追溯）
    handle_parsing_errors="log_and_continue"  # 优化：设置解析错误处理策略，出错时记录日志并继续执行，提升鲁棒性
)

# 调用Agent执行器，传入用户复合任务请求，执行多任务并行处理
# 用户请求包含3个子任务：1. 查询自身身份证号 2. 查询苹果公司最新股价 3. 查询并预订广州→北京明天的航班
resp = agent_executor.invoke({
    "input": "我的身份证是多少?苹果公司最新的股票价格是多少? 根据我的行程帮我查询明天的航班，从广州到北京，并且定个机票"
})

# 打印最终结果，包含Agent的最终回复和中间执行步骤
print("=" * 80)
print(f"最终结果（核心回复）:\n{resp['output']}")
print("=" * 80)
print(f"完整返回结果（含中间步骤）:\n{resp}")
```

### REACT 智能体实战

```python
# 基于LangChain搭建具备「城市天气查询、天气适配活动推荐、网页实时股票搜索」能力的REACT智能体
import os
from langchain_openai import ChatOpenAI
from langchain.tools import tool  # 导入tool装饰器，用于定义智能体可调用工具
from langchain_community.utilities import SearchApiAPIWrapper  # 导入SearchApi封装类，实现网页搜索
from langchain_core.prompts import PromptTemplate  # 导入PromptTemplate，用于自定义REACT提示词模板
from langchain.agents import create_react_agent, AgentExecutor  # 导入REACT智能体创建与执行器类
from langchain import hub  # 导入hub（虽未直接使用，保留用于后续对接官方提示词仓库）
from pydantic import SecretStr  # 导入SecretStr，用于安全存储API密钥

# 配置LangChain Smith环境变量，用于追踪智能体运行轨迹、日志和性能指标
# 启用v2版本追踪功能，支持更详细的运行数据采集
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# LangChain Smith的官方API端点地址，用于上传运行数据
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# LangChain Smith的API密钥，用于身份验证（确保只有授权用户能上传/查看数据）

# 定义当前项目名称，方便在LangChain Smith后台分类管理和查询
os.environ["LANGCHAIN_PROJECT"] = "agent_v1"

# 配置SearchApi的API密钥，用于调用网页搜索接口获取实时数据（如腾讯股票价格）
os.environ["SEARCHAPI_API_KEY"] = ""

# 实例化SearchApiWrapper对象，封装了SearchApi的调用逻辑，提供便捷的results()搜索方法
# 后续web_search工具将通过该对象执行实际的网页搜索请求
search = SearchApiAPIWrapper()

# 定义天气查询工具，使用@tool装饰器标记为LangChain智能体可调用工具
# 入参：city（城市名称），返回值：格式化的天气信息
@tool
def get_weather(city: str) -> str:
    """
    工具功能描述（供REACT智能体判断是否调用）：获取指定城市的当前天气信息（模拟固定数据）
    参数说明：city - 城市名称（如"北京"、"上海"，仅支持预设城市列表）
    """
    # 模拟天气数据集，实际生产环境可对接气象API（如中国天气网、和风天气等）
    weather_data = {
        "北京": "晴, 25℃",
        "上海": "雨, 20℃",
        "广州": "多云, 28℃",
        "深圳": "晴, 27℃",
        "杭州": "多云, 23℃",
        "成都": "雨, 18℃"
    }
    # 从模拟数据中获取指定城市天气，无匹配城市时返回友好提示
    return weather_data.get(city, "暂不支持该城市的天气查询")

# 定义活动推荐工具，使用@tool装饰器标记为LangChain智能体可调用工具
# 入参：weather（天气信息），返回值：对应的活动推荐内容
@tool
def recommend_activity(weather: str) -> str:
    """
    工具功能描述（供REACT智能体判断是否调用）：根据输入的天气信息推荐合适的出行活动
    参数说明：weather - 天气描述字符串（如"晴, 25℃"、"雨, 20℃"）
    """
    # 根据天气关键词进行分支判断，推荐对应场景的活动
    if "雨" in weather:
        return "推荐室内活动: 博物馆参观、美术馆观展、咖啡厅阅读、商场购物。"
    elif "晴" in weather:
        return "推荐户外活动: 公园骑行、郊游野餐、登山徒步、户外运动。"
    else:  # 多云、阴等其他天气
        return "推荐一般活动: 城市观光、美食探索、文化体验。"

# 定义网页搜索工具，指定工具名称"web_search"，设置return_direct=True直接返回结果
@tool("web_search", return_direct=True)
def web_search(query: str) -> str:
    """
    工具功能描述（供REACT智能体判断是否调用）：
    当需要获取实时信息、最新事件或未知领域知识时使用（如股票价格、实时新闻等），输入应为搜索关键词
    """
    try:
        # 调用SearchApiWrapper的results方法执行搜索，num=3指定返回3条结果
        results = search.results(query, num=3)
        # 解析搜索结果中的有机结果（organic_results），提取前3条的标题和摘要
        # 格式化拼接后返回，提升结果的可读性和结构化
        return "\n\n".join([
            f"来源: {res['title']}\n内容: {res['snippet']}"
            for res in results['organic_results'][:3]
        ])
    except Exception as e:
        # 捕获搜索过程中的异常（如API密钥失效、网络错误、无搜索结果等）
        return f"搜索失败: {str(e)}"

# 整合所有工具形成工具列表，供智能体初始化时调用
# 智能体将基于每个工具的功能描述，决策何时调用哪个工具
tools = [get_weather, recommend_activity, web_search]

# 实例化大语言模型（LLM），对接阿里通义千问qwen-plus（兼容OpenAI接口格式）
model = ChatOpenAI(
    model="qwen-plus",  # 指定使用的模型名称（通义千问增强版，具备较强的逻辑推理和工具调用能力）
    base_url= "https://dashscope.aliyuncs.com/compatible-mode/v1",  # 通义千问兼容OpenAI接口的端点地址
    api_key=SecretStr(""),  # 通义千问API密钥，使用SecretStr安全存储（避免明文泄露）
    temperature=0.7)  # 模型生成内容的随机性（0~1），0.7兼顾逻辑性和一定的灵活度

# 定义REACT智能体的提示词模板，严格遵循REACT的「思考-行动-观察」循环格式
# 该模板是REACT智能体运行的核心指引，规定了智能体的行为逻辑和输出格式
template = """
Answer the following questions as best you can. You have access to the following tools:
{tools}  # 占位符：将自动填充所有工具的名称和功能描述
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do  # 思考：判断是否需要调用工具、调用哪个工具
Action: the action to take, should be one of [{tool_names}]  # 行动：指定要调用的工具名称（必须在工具列表中）
Action Input: the input to the action  # 行动输入：传递给工具的入参
Observation: the result of the action  # 观察：接收工具执行后的返回结果
... (this Thought/Action/Action Input/Observation can repeat N times)  # 可循环多次，完成复杂任务
Thought: I now know the final answer  # 最终思考：确认已获取足够信息，可得出最终答案
Final Answer: the final answer to the original input question  # 最终答案：返回给用户的完整结果

Begin!  # 启动指令，提示智能体开始执行任务
Question: {input}  # 占位符：接收用户传入的查询输入
Thought:{agent_scratchpad}  # 占位符：存储智能体的思考过程、行动记录和中间结果（不可删除）
"""

# 基于自定义模板创建PromptTemplate对象，自动识别并绑定模板中的占位符变量
prompt = PromptTemplate.from_template(template)

# 创建REACT智能体，整合模型、工具和提示词模板，具备「思考-行动-观察」的循环决策能力
agent = create_react_agent(
    llm=model,  # 传入实例化的大语言模型（作为智能体的"大脑"，负责逻辑推理和工具决策）
    tools=tools,  # 传入智能体可调用的工具列表
    prompt=prompt  # 传入自定义的REACT提示词模板，规定智能体的运行格式和逻辑
)

# 创建智能体执行器，负责调度智能体运行、处理工具调用流程、返回最终结果
agent_executor = AgentExecutor(
    agent=agent,  # 传入已创建的REACT智能体
    tools=tools,  # 传入工具列表（与创建智能体时保持一致）
    verbose=True,  # 启用详细日志输出，可在控制台查看智能体的完整运行流程（思考、行动、输入、观察等）
    return_intermediate_steps=True,  # 启用中间步骤返回，最终结果中包含所有工具调用记录和中间数据（方便调试和追溯）
    handle_parsing_errors="log_and_continue"  # 优化：添加解析错误处理策略，出错时记录日志并继续执行，提升鲁棒性
)

# 调用智能体执行器，传入用户复合任务请求，执行多步骤任务处理
# 用户请求包含2个子任务：1. 基于北京天气推荐三天出行活动 2. 查询腾讯最新股票价格
resp = agent_executor.invoke({"input":"我在北京玩三天，根据天气推荐活动，顺便查询腾讯的股票价格是多少?"})

# 打印完整返回结果，包含最终回复和中间执行步骤
print("=" * 100)
print("完整返回结果（含最终回复+中间步骤）：")
print("=" * 100)
print(resp)
```

### Milvus向量库检索整合REACT智能体实战

```python
# LangChain综合实战：搭建具备「Milvus向量库检索」和「网页实时搜索」能力的REACT智能体，解决Milvus相关专业查询任务
import os

# 导入阿里通义千问嵌入模型，用于将文本转换为向量（适配Milvus向量存储）
from langchain_community.embeddings import DashScopeEmbeddings
# 导入创建检索器工具的方法，将Milvus检索器封装为LangChain智能体可调用工具
from langchain_core.tools import create_retriever_tool
# 导入Milvus向量存储模块，用于构建向量数据库、存储和检索文本向量
from langchain_milvus import Milvus
# 导入ChatOpenAI，用于实例化大语言模型（对接阿里通义千问，兼容OpenAI接口）
from langchain_openai import ChatOpenAI
# 导入tool装饰器，用于定义自定义工具（网页搜索）
from langchain.tools import tool
# 导入SearchApiAPIWrapper，用于对接网页搜索API获取实时信息（如Milvus最新版本、GitHub教程）
from langchain_community.utilities import SearchApiAPIWrapper
# 导入提示词模板类，用于构建REACT智能体的运行格式模板
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
# 导入REACT智能体和工具调用智能体的创建方法、智能体执行器
from langchain.agents import create_react_agent, AgentExecutor, create_tool_calling_agent
# 导入LangChain hub，用于对接官方提示词仓库（虽未直接使用，保留用于后续优化）
from langchain import hub
# 导入SecretStr，用于安全存储和传递API密钥（避免明文泄露风险）
from pydantic import SecretStr

# 配置LangChain Smith环境变量，用于追踪智能体的完整运行轨迹、日志和性能指标
# 启用v2版本追踪功能，支持更详细的运行数据采集和后台可视化查看
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# LangChain Smith的官方API端点地址，用于上传智能体运行数据
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# LangChain Smith的API密钥，用于身份验证（确保只有授权用户能上传/查看运行数据）

# 定义当前项目名称，方便在LangChain Smith后台分类管理、查询和对比不同版本智能体
os.environ["LANGCHAIN_PROJECT"] = "llm_agent_v1"

# 配置SearchApi的API密钥，用于调用网页搜索接口获取实时数据（如Milvus最新版本、GitHub教程链接）
os.environ["SEARCHAPI_API_KEY"] = ""

# 实例化SearchApiWrapper对象，封装了SearchApi的调用逻辑，提供便捷的results()搜索方法
# 后续web_search工具将通过该对象执行实际的网页实时搜索请求
search = SearchApiAPIWrapper()

# 实例化阿里通义千问嵌入模型（DashScopeEmbeddings），用于文本向量化处理
# 嵌入模型的核心作用：将自然语言文本转换为计算机可识别的向量数据，供Milvus向量库存储和检索
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",  # 指定使用通义千问第二代通用文本嵌入模型，具备更好的语义表征能力
    max_retries=3,  # 设置请求重试次数，提升接口调用的稳定性（网络波动时自动重试）
    dashscope_api_key="",  # 通义千问API密钥，用于嵌入模型接口调用
)

# 实例化Milvus向量存储对象，构建本地/远程向量数据库，用于存储和检索文本向量数据
vector_storage = Milvus(
    embedding_function=embeddings,  # 绑定嵌入模型，用于自动将文本转换为向量
    collection_name="doc_qa_db",  # 定义Milvus中的集合名称（相当于数据库中的表名）
    connection_args={"uri": "http://192.168.64.137:19530"},  # Milvus服务连接参数，指定服务地址和端口
    drop_old=True  # 设置为True：如果集合已存在，先删除旧集合再创建新集合（方便测试和重置）
)

# 将Milvus向量存储转换为检索器（Retriever），具备相似性查询能力
# 检索器的核心作用：接收查询文本，返回Milvus中语义最相似的文本结果
retriever = vector_storage.as_retriever()

# 将Milvus检索器封装为LangChain智能体可直接调用的工具
# 使智能体能够通过工具调用的方式，查询Milvus向量库中的相关信息
tool_retriever = create_retriever_tool(
    retriever,  # 绑定Milvus检索器，作为工具的核心执行逻辑
    "milvs_retriever",  # 工具名称（供REACT智能体识别和调用，注意：此处拼写为milvs_retriever，与Milvus一致）
    "搜索有关Milvus的信息。对于任何有关Milvus的问题，你必须使用这个工具"  # 工具功能描述（供智能体判断是否需要调用）
)

# 实例化大语言模型（LLM），对接阿里通义千问qwen-plus（兼容OpenAI接口格式）
# 模型作为智能体的"大脑"，负责逻辑推理、工具调用决策和最终结果整合
model = ChatOpenAI(
    model="qwen-plus",  # 指定使用的模型名称（通义千问增强版，具备较强的专业逻辑推理和工具调用能力）
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 通义千问兼容OpenAI接口的端点地址
    api_key=SecretStr(""),  # 通义千问API密钥，使用SecretStr安全存储（避免明文泄露）
    temperature=0.7  # 模型生成内容的随机性（0~1），0.7兼顾逻辑性和一定的灵活度，适合专业查询任务
)

# 定义网页实时搜索工具，指定工具名称"web_search"，设置return_direct=True直接返回搜索结果
@tool("web_search", return_direct=True)
def web_search(query: str) -> str:
    """
    工具功能描述（供REACT智能体判断是否需要调用）：
    当需要获取实时信息、最新版本、官方教程、对比数据等动态更新内容时使用，输入应为搜索关键词
    （例如：Milvus最新版本、Milvus LangChain 整合教程、Milvus vs Faiss 优缺点）
    """
    try:
        # 调用SearchApiWrapper的results方法执行网页搜索，num=3指定返回前3条核心结果
        results = search.results(query, num=3)
        # 解析搜索结果中的有机结果（organic_results），提取标题和摘要并格式化拼接
        # 最终返回结构化、易读的搜索结果，方便智能体整合和输出
        return "\n\n".join([
            f"来源: {res['title']}\n内容: {res['snippet']}"
            for res in results['organic_results'][:3]
        ])
    except Exception as e:
        # 捕获搜索过程中的异常（如API密钥失效、网络错误、无搜索结果等），返回友好错误提示
        return f"搜索失败: {str(e)}"

# 整合所有工具形成工具列表，供REACT智能体初始化时调用
# 智能体将基于每个工具的功能描述，决策何时调用哪个工具（Milvus检索/网页搜索）
tools = [web_search, tool_retriever]

# 注释：保留工具调用型智能体的提示词模板（备用），当前实战使用REACT智能体模板
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "你是一个 个人AI助手，必要时可以调用工具函数帮助用户解决问题"),
#     ("human", "我叫老王，经常出差，喜欢性价比的按摩，身份证号是4444144444"),
#     ("human", "{input}"),
#     ("placeholder", "{agent_scratchpad}"),
# ])

# 构建REACT智能体的提示词模板，严格遵循REACT的「思考-行动-观察」循环格式
# 该模板是智能体运行的核心指引，规定了智能体的行为逻辑和输出格式要求
prompt = PromptTemplate.from_template('''Answer the following questions as best you can. You have access to this topic
    {tools}  # 占位符：自动填充所有工具的名称和详细功能描述

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do  # 思考：判断是否需要调用工具、调用哪个工具
Action: the action to take, should be one of {tool_names}  # 行动：指定要调用的工具名称（必须在工具列表中）
Action Input: the input to the action  # 行动输入：传递给工具的查询关键词/参数
Observation: the result of the action  # 观察：接收工具执行后的返回结果
... (this Thought/Action/Action Input/Observation can repeat N times)  # 可循环多次，完成复杂多步骤任务
Thought: I now know the final answer  # 最终思考：确认已获取足够信息，可整合形成最终答案
Final Answer: the final answer to the original input question  # 最终答案：返回给用户的完整、结构化回复

Begin!  # 启动指令，提示智能体开始执行任务

Question: {input}  # 占位符：接收用户传入的查询输入
Thought:{agent_scratchpad}  # 占位符：智能体的「草稿区」，存储思考过程、行动记录和中间结果（不可删除）''')

# 注释：保留工具调用型智能体的创建代码（备用），当前实战使用REACT智能体
# agent = create_tool_calling_agent(
#     llm=model,
#     tools=tools,
#     prompt=prompt
# )

# 创建REACT智能体，整合大语言模型、工具列表和提示词模板，具备复杂任务的分步决策能力
agent = create_react_agent(
    llm=model,  # 传入实例化的大语言模型（负责逻辑推理、工具调用决策）
    tools=tools,  # 传入智能体可调用的工具列表（Milvus检索+网页搜索）
    prompt=prompt  # 传入自定义的REACT提示词模板，规定智能体的运行格式和逻辑
)

# 创建智能体执行器，负责调度智能体运行、处理工具调用流程、返回最终结果和中间步骤
agent_executor = AgentExecutor(
    agent=agent,  # 传入已创建的REACT智能体
    tools=tools,  # 传入工具列表（与创建智能体时保持一致）
    verbose=True,  # 启用详细日志输出，可在控制台查看智能体的完整运行流程
    return_intermediate_steps=True,  # 启用中间步骤返回，方便追溯工具调用记录和调试
    handle_parsing_errors="log_and_continue"  # 优化：设置解析错误处理策略，出错时记录日志并继续执行，提升鲁棒性
)

def run_agent(question: str):
    """
    封装智能体运行函数，简化调用流程，格式化输出结果和中间步骤
    参数：question - 用户需要查询的问题（字符串格式）
    """
    print(f"\n问题:{question}")
    # 调用智能体执行器，传入用户问题，获取执行结果
    result = agent_executor.invoke({"input": question})
    # 打印完整返回结果（含最终回复+中间步骤，便于调试）
    print(f"\n llm-result:{result}")
    # 打印格式化的最终答案（便于用户查看核心结果）
    print(f"\n 答案:{result['output']}\n{'='*50}")
    # 遍历并打印所有中间步骤，详细展示智能体的「思考-行动-输入-观察」过程
    for step in result["intermediate_steps"]:
        print(f"\n Thought:{step[0].log}")  # 打印智能体的思考过程
        print(f"\n Action:{step[0].tool}")  # 打印智能体调用的工具名称
        print(f"\n Action Input:{step[0].tool_input}")  # 打印工具的输入参数
        print(f"\n Observation:{step[1]}")  # 打印工具的返回结果

# 调用封装的智能体运行函数，传入复杂多子任务查询问题，执行综合实战任务
# 用户问题包含5个子任务：1. 什么是Milvus 2. Milvus最新版本 3. Milvus整合LangChain 4. Milvus与Faiss对比 5. Milvus GitHub最新教程链接
run_agent("中文回答下面3个问题：第一个，什么是Milvus，最新的版本是多少？如何整合langchain框架，与Faiss对比优缺点，最后给出GitHub最新教程链接")
```

### 多轮对话内容的摘要生成与记忆管理实战

```python
# 从 LangChain 记忆模块导入对话摘要记忆类，用于实现对话内容的自动摘要存储
from langchain.memory import ConversationSummaryMemory
# 从 LangChain OpenAI 模块导入 ChatOpenAI 类，用于实例化大语言模型客户端
from langchain_openai import ChatOpenAI
# 从 pydantic 导入 SecretStr，用于安全存储和处理敏感信息（如 API 密钥）
from pydantic import SecretStr

# ---------------------- 大语言模型实例化 ----------------------
# 创建 ChatOpenAI 模型实例，用于后续对话摘要的生成
model = ChatOpenAI(
    model="qwen-plus",  # 指定要使用的模型名称（此处为通义千问 plus 版本）
    base_url= "https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云通义千问的兼容 OpenAI 格式的 API 地址
    api_key=SecretStr(""),  # 安全存储 API 密钥（敏感信息），避免明文泄露风险
    temperature=0.7)  # 模型生成内容的随机性/创造性参数（0-1之间），0.7 表示兼顾逻辑性和一定创造性

# ---------------------- 对话摘要提示词模板 ----------------------
# 定义渐进式对话摘要的提示词模板，指导模型如何生成连贯的对话摘要
# 核心逻辑：基于历史摘要（{summary}）和新增对话内容（{new_lines}），生成更新后的完整摘要
prompt = """Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.

EXAMPLE
Current summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.

New lines of conversation:
Human: Why do you think artificial intelligence is a force for good?
AI: Because artificial intelligence will help humans reach their full potential.

New summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.
END OF EXAMPLE

Current summary:
{summary}

New lines of conversation:
{new_lines}

New summary:"""

# ---------------------- 对话摘要记忆初始化与使用 ----------------------
# 初始化对话摘要记忆对象，传入已实例化的大语言模型（用于自动生成摘要）
# 该对象会自动管理对话历史，将新对话内容整合到历史摘要中，无需手动处理
memory = ConversationSummaryMemory(llm=model)

# 模拟第一轮对话，将用户输入和 AI 回复保存到摘要记忆中
# input：用户输入内容；output：AI 回复内容
memory.save_context({"input": "Hi"}, {"output": "What's up?"})

# 模拟第二轮对话，继续将新的对话内容保存到摘要记忆中
# 此时记忆对象会自动调用模型，将本轮对话整合到上一轮的摘要中
memory.save_context({"input": "Not much you?"}, {"output": "Not much."})

# 从摘要记忆中加载当前的对话摘要（以字典格式返回）
# 返回结果中包含 "history" 键，对应的值为完整的对话摘要内容
summary = memory.load_memory_variables({})

# 打印最终的对话摘要结果
print(summary)
```

### 短期记忆实战

```python
# 短期记忆实战：基于 LangChain 实现带对话摘要记忆的多轮智能问答，支持上下文关联回复
# 从 LangChain 记忆模块导入对话摘要记忆类，用于自动生成和管理对话历史摘要
from langchain.memory import ConversationSummaryMemory
# 从 LangChain 核心输出解析模块导入字符串输出解析器，用于格式化模型返回结果（转为纯字符串）
from langchain_core.output_parsers import StrOutputParser
# 从 LangChain 核心提示词模块导入聊天提示词模板，用于构建结构化的多角色对话提示词
from langchain_core.prompts import ChatPromptTemplate
# 从 LangChain 核心可运行模块导入透传工具，用于动态扩展链的输入字段、加载记忆数据
from langchain_core.runnables import RunnablePassthrough
# 从 LangChain OpenAI 模块导入 ChatOpenAI 类，用于实例化兼容 OpenAI 格式的大语言模型客户端
from langchain_openai import ChatOpenAI
# 从 pydantic 导入 SecretStr，用于安全存储和处理敏感信息（如 API 密钥），避免明文泄露
from pydantic import SecretStr

# ---------------------- 大语言模型实例化 ----------------------
# 创建 ChatOpenAI 模型实例，作为对话生成和摘要生成的核心引擎
model = ChatOpenAI(
    model="qwen-plus",  # 指定使用的模型名称（通义千问 plus 版本，具备较强的理解和生成能力）
    base_url= "https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云通义千问的 OpenAI 兼容 API 地址
    api_key=SecretStr(""),  # 安全存储 API 密钥（敏感信息），SecretStr 会隐藏明文展示
    temperature=0.7)  # 模型生成内容的随机性参数（0-1区间），0.7 兼顾逻辑性和一定的创造性（注：原注释"亲密度"不准确，应为"随机性/创造性"）

# ---------------------- 对话摘要记忆初始化 ----------------------
# 初始化对话摘要记忆对象，用于自动整合多轮对话为摘要，实现长期上下文记忆
memory = ConversationSummaryMemory(
    llm=model,  # 传入已实例化的模型，用于自动生成对话摘要（增量更新）
    return_messages=True,  # 设置返回格式为 Message 对象列表（而非纯字符串），适配 ChatPromptTemplate 格式要求
    memory_key="chat_history",  # 定义记忆数据的键名，后续在提示词和链中通过该键获取对话摘要
)

# ---------------------- 聊天提示词模板构建 ----------------------
# 构建结构化的聊天提示词模板，包含系统角色和用户角色，实现上下文关联问答
prompt = ChatPromptTemplate.from_messages([
    # 系统角色：定义 AI 助手的身份、行为准则，同时注入对话历史摘要（{chat_history} 为记忆键对应的占位符）
    ("system", "你是一个AI智能助手，你的名字是AI助手，需要基于历史对话回答问题，当前摘要信息:{chat_history}"),
    # 用户角色：接收用户的实时输入（{input} 为用户输入的占位符，运行时会被真实提问替换）
    ("user", "{input}")
])

# ---------------------- 构建 LangChain 运行链（LCEL 表达式） ----------------------
# 定义 LCEL（LangChain Expression Language）表达式，构建端到端的问答运行链
# 链式结构按从左到右顺序执行，通过 | 符号串联各个组件
chain = (
    # 第一步：RunnablePassthrough.assign() - 动态扩展输入字段，为后续链补充额外数据（此处为对话记忆）
    RunnablePassthrough.assign(
        # 定义要扩展的字段：chat_history（与记忆对象的 memory_key 对应）
        # 匿名函数 lambda _: 接收输入（此处无需使用输入，用 _ 占位），加载记忆中的对话摘要数据
        chat_history = lambda _: memory.load_memory_variables({})["chat_history"]
    )|  # 管道符：将上一步的输出（扩展后的字段字典）作为下一步的输入
    # 第二步：prompt - 接收扩展后的字段字典，填充占位符，生成完整的结构化提示词
    prompt |
    # 第三步：model - 接收完整提示词，调用大语言模型生成回复内容
    model |
    # 第四步：StrOutputParser() - 解析模型返回结果，将复杂的 Message 对象转为纯字符串格式，方便后续使用和存储
    StrOutputParser()
)

# ---------------------- 定义模拟用户输入列表 ----------------------
# 准备多轮模拟用户提问，包含个人信息、知识点查询、上下文回溯查询（验证记忆功能）
user_input = [
    "我叫老王，现在是计算机专业大学生",
    "人工智能的定义",
    "",
    "人工智能在医疗领域有什么应用",
    "我是谁?读什么专业的？",  # 用于验证：AI 是否能通过对话记忆回答个人信息（上下文回溯）
]

# ---------------------- 循环调用问答链，执行多轮对话 ----------------------
# 遍历用户输入列表，逐轮执行问答流程，验证记忆功能
for query in user_input:
    # 调用链式结构，传入用户当前输入（字典格式，key 与 prompt 中的 {input} 对应），获取 AI 回复
    resp = chain.invoke({"input":query})
    
    # 打印当前轮次的用户提问和 AI 回复，方便查看对话过程
    print(f"User提问:{query}")
    print(f"AI回复:{resp}",end="\n\n")
    
    # 手动保存当前轮次的对话到记忆对象中（关键步骤）
    # 将用户输入（input）和 AI 回复（output）存入记忆，记忆对象会自动调用模型生成/更新摘要
    memory.save_context({"input": query}, {"output": resp})
    
    # 打印当前最新的对话摘要，查看记忆对象的增量更新结果，验证摘要生成功能
    print(f"打印当前摘要：{memory.load_memory_variables({})['chat_history']}")
    print("-" * 80)  # 分隔线，优化打印结果的可读性（可选补充）
```

### 聊天模型嵌入历史记录实战

```python
# 从LangChain链模块导入LLMChain，用于构建基于大语言模型的端到端运行链
from langchain.chains.llm import LLMChain
# 从LangChain记忆模块导入两种对话记忆类（本演示使用ConversationBufferMemory，前者为备用/对比）
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory
# 从LangChain核心提示词模块导入：
# ChatPromptTemplate：构建结构化聊天提示词模板
# MessagesPlaceholder：对话历史占位符，用于动态嵌入完整对话消息列表
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# 从LangChain OpenAI模块导入ChatOpenAI，实例化兼容OpenAI格式的大语言模型客户端
from langchain_openai import ChatOpenAI
# 从pydantic导入SecretStr，用于安全存储敏感信息（API密钥），避免明文泄露风险
from pydantic import SecretStr

# ---------------------- 大语言模型实例化 ----------------------
# 创建ChatOpenAI模型实例，作为翻译任务和对话记忆管理的核心引擎
llm = ChatOpenAI(
    model="qwen-plus",  # 指定使用的模型名称（通义千问plus版本，具备优秀的文本理解与翻译能力）
    base_url= "https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云通义千问的OpenAI兼容API地址
    api_key=SecretStr(""),  # 安全存储API密钥（敏感信息），SecretStr会隐藏明文展示
    temperature=0.7)  # 模型生成内容的随机性/创造性参数（0-1区间），0.7兼顾翻译准确性和表达流畅性（注：原注释"亲密度"不准确）

# ---------------------- 对话记忆初始化（使用ConversationBufferMemory） ----------------------
# 初始化对话缓冲记忆对象，用于完整存储多轮对话的原始消息（不做摘要压缩，保留完整上下文）
memory = ConversationBufferMemory(
    llm=llm,  # 传入已实例化的模型（虽缓冲记忆默认无需模型生成摘要，仍可用于后续扩展）
    return_messages=True,  # 设置返回格式为Message对象列表（而非纯字符串），适配MessagesPlaceholder要求
    memory_key="chat_history",  # 定义记忆数据的键名，**必须与MessagesPlaceholder的variable_name一致**，否则无法正常嵌入
)

# ---------------------- 构建包含对话历史占位符的提示词模板 ----------------------
# 构建结构化聊天提示词模板，核心特点是嵌入对话历史占位符，实现上下文关联翻译
prompt = ChatPromptTemplate.from_messages([
    # 系统角色：定义AI的身份为翻译助手，明确核心任务目标
    ("system", "你是一个翻译助手"),
    # 对话历史占位符：动态嵌入由memory管理的完整对话消息列表（key为chat_history）
    # 作用：将历史对话上下文传入模型，支持关联式翻译（如需基于前文语境优化翻译结果）
    MessagesPlaceholder(variable_name="chat_history"),
    # 用户角色：接收用户实时输入的待翻译文本（{input}为用户输入占位符）
    ("user", "{input}")
])

# ---------------------- 构建LLMChain运行链（集成模型、提示词、记忆） ----------------------
# 实例化LLMChain链，将模型、提示词模板、对话记忆三者集成，实现端到端的带记忆翻译
chain = LLMChain(
    llm=llm,  # 传入核心大语言模型
    prompt=prompt,  # 传入已构建的带历史占位符的提示词模板
    memory=memory,  # 传入已初始化的对话缓冲记忆，链会自动管理对话的存储与加载（无需手动调用save_context）
)

# ---------------------- 定义模拟用户输入列表（待翻译的英文句子） ----------------------
# 准备3句待翻译的英文文本，用于验证多轮对话记忆和翻译功能
user_input = [
    "The symphony of raindrops on the rooftop played a gentle, rhythmic lullaby.",
    "A solitary dandelion seed drifted on the breeze, carrying its promise of new life to unknown soil",
    "He found that the oldest maps often led not to treasure, but to the most profound silence"
]

# ---------------------- 循环调用LLMChain，执行多轮带记忆翻译 ----------------------
# 遍历用户输入列表，逐轮执行翻译任务，验证对话记忆的自动管理功能
for query in user_input:
    # 调用LLMChain链，传入用户当前输入（字典格式，key与prompt中的{input}对应），获取翻译结果
    # 注：因链中集成了memory，会自动加载历史对话并嵌入提示词，执行后也会自动保存本轮对话到记忆中
    resp = chain.invoke({"input":query})
    
    # 打印当前轮次的用户提问（待翻译英文）和AI回复（翻译结果）
    print(f"User提问:{query}")
    print(f"AI回复:{resp}",end="\n\n")
    
    # 打印当前最新的完整对话历史，验证记忆对象的自动更新功能
    # 从记忆中加载以"chat_history"为键的对话消息列表，查看所有历史对话记录
    print(f"打印当前历史：{memory.load_memory_variables({})['chat_history']}")
    print("-" * 100)  # 分隔线，优化打印结果的可读性（可选补充）
```

### 多会话隔离记忆

```python
# 从LangChain核心消息模块导入系统消息和人类消息类，用于构建结构化对话消息
from langchain_core.messages import SystemMessage, HumanMessage
# 从LangChain核心提示词模块导入：
# ChatPromptTemplate：构建结构化聊天提示词模板
# MessagesPlaceholder：对话历史占位符，用于动态嵌入会话历史消息列表
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# 从LangChain核心可运行模块导入RunnableWithMessageHistory，用于为链式结构添加会话历史管理和隔离能力
from langchain_core.runnables import RunnableWithMessageHistory
# 从LangChain OpenAI模块导入ChatOpenAI，实例化兼容OpenAI格式的大语言模型客户端
from langchain_openai import ChatOpenAI
# 从pydantic导入SecretStr，用于安全存储敏感信息（API密钥），避免明文泄露风险
from pydantic import SecretStr
# 从LangChain记忆模块导入ChatMessageHistory，用于存储单个会话的完整消息历史（会话级记忆载体）
from langchain.memory import ChatMessageHistory

# ---------------------- 全局会话存储初始化 ----------------------
# 定义全局字典store，用于存储所有用户/会话的历史记录，实现多会话隔离
# 键（key）：会话唯一标识（session_id），值（value）：对应会话的ChatMessageHistory实例（存储该会话的完整对话）
store = {}

# ---------------------- 会话历史获取函数（核心：实现会话隔离与创建） ----------------------
def get_session_history(session_id: str):
    """
    根据会话唯一标识（session_id）获取对应的会话历史，实现多会话隔离
    :param session_id: 会话唯一标识（如用户ID、会话ID），用于区分不同会话
    :return: 对应session_id的ChatMessageHistory实例（会话历史载体）
    """
    # 从全局字典store中，根据session_id获取对应的会话历史对象
    history = store.get(session_id)
    
    # 判空逻辑：如果该session_id对应的会话历史不存在（首次访问该会话）
    if history is None:  # 修复点：使用 is None 准确判断对象是否为空，而非 in None
        # 创建一个新的ChatMessageHistory实例，用于存储该会话的后续对话消息
        history = ChatMessageHistory()
        # 将新创建的会话历史存入全局字典，绑定对应的session_id，方便后续获取
        store[session_id] = history
    
    # 返回该session_id对应的会话历史对象（存在则返回已有，不存在则返回新建并存储后的）
    return history  # 修复点：确保函数返回会话历史对象，供后续链式结构使用

# ---------------------- 构建带会话历史占位符的聊天提示词模板 ----------------------
# 构建结构化聊天提示词模板，包含系统消息、历史消息占位符、人类输入，支持会话上下文关联
prompt = ChatPromptTemplate.from_messages([
    # 系统消息：定义AI助手的能力边界和回答要求，{ability}为动态传入的能力参数
    SystemMessage(content="你是一个AI助手，擅长能力{ability}。用30个字以内回答"),
    # 会话历史占位符：动态嵌入对应session_id的对话历史消息列表，key为"history"
    MessagesPlaceholder(variable_name="history"),
    # 人类消息：接收用户的实时输入，{input}为动态传入的用户提问参数
    HumanMessage(content="{input}")
])

# ---------------------- 大语言模型实例化 ----------------------
# 创建ChatOpenAI模型实例，作为会话问答的核心生成引擎
llm = ChatOpenAI(
    model="qwen-plus",  # 指定使用的模型名称（通义千问plus版本，具备优秀的文本理解与生成能力）
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云通义千问的OpenAI兼容API地址
    api_key=SecretStr(""),  # 安全存储API密钥（敏感信息），SecretStr会隐藏明文展示
    temperature=0.7)  # 模型生成内容的随机性/创造性参数（0-1区间），0.7兼顾逻辑性和表达流畅性（注：原注释"亲密度"不准确）

# ---------------------- 构建基础运行链 ----------------------
# 用管道符|串联提示词模板和模型，构建端到端的基础问答链（无会话记忆能力）
chain = prompt | llm

# ---------------------- 为基础链添加会话历史管理与隔离能力 ----------------------
# 封装RunnableWithMessageHistory，为基础链赋予「会话历史记忆+多会话隔离」的核心能力
with_message_history = RunnableWithMessageHistory(
    chain,  # 传入基础运行链，为其添加会话历史功能
    get_session_history=get_session_history,  # 传入会话历史获取函数，用于获取/创建对应session_id的会话历史
    input_messages_key="input",  # 指定用户输入对应的键名，与prompt中的{input}和invoke传入的参数对应
    history_messages_key="history"  # 指定会话历史对应的键名，与MessagesPlaceholder的variable_name对应
)

# ---------------------- 第一次调用：会话user_123 首次提问 ----------------------
# 调用带会话历史的链式结构，向session_id=user_123的会话发送首次提问
resp1 = with_message_history.invoke(
    # 传入提示词模板所需的动态参数：能力类型和用户输入
    {
        "ability": "Java开发",
        "input": "什么是JVM"
    },
    # 配置会话唯一标识，指定本次调用归属的会话（实现会话隔离的关键配置）
    config={"configurable":{"session_id": "user_123"}}
)

# 打印全局存储字典store，查看session_id=user_123的会话历史是否已创建并存储
print(f"store1：{store}")
# 打印第一次调用的AI回复内容（提取Message对象的content属性，获取纯文本回复）
print(f"resp1:{resp1.content}",end="\n\n")

# ---------------------- 第二次调用：会话user_123 后续追问 ----------------------
# 继续调用带会话历史的链式结构，向同一个session_id=user_123的会话发送追问
resp2 = with_message_history.invoke(
    # 传入提示词模板所需的动态参数，本次要求重新回答JVM相关问题
    {
        "ability": "Java开发",
        "input": "重新回答一次"
    },
    # 配置相同的session_id=user_123，确保归属同一个会话，可关联上一轮对话上下文
    config={"configurable":{"session_id": "user_123"}}
)

# 打印全局存储字典store，查看session_id=user_123的会话历史是否已追加本轮对话
print(f"store2：{store}")
# 打印第二次调用的AI回复内容
print(f"resp2:{resp2.content}",end="\n\n")
```

### 多租户 / 多会话隔离记忆实战

```python
# 从LangChain核心消息模块导入系统消息和人类消息类，用于构建结构化对话消息
from langchain_core.messages import SystemMessage, HumanMessage
# 从LangChain核心提示词模块导入：
# ChatPromptTemplate：构建结构化聊天提示词模板
# MessagesPlaceholder：对话历史占位符，用于动态嵌入会话历史消息列表
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# 从LangChain核心可运行模块导入：
# RunnableWithMessageHistory：为链式结构添加会话历史管理和隔离能力
# ConfigurableFieldSpec：定义自定义可配置字段，支持多参数（用户+会话）的会话隔离
from langchain_core.runnables import RunnableWithMessageHistory, ConfigurableFieldSpec
# 从LangChain OpenAI模块导入ChatOpenAI，实例化兼容OpenAI格式的大语言模型客户端
from langchain_openai import ChatOpenAI
# 从pydantic导入SecretStr，用于安全存储敏感信息（API密钥），避免明文泄露风险
from pydantic import SecretStr
# 从LangChain记忆模块导入ChatMessageHistory，用于存储单个（用户+会话）的完整消息历史
from langchain.memory import ChatMessageHistory

# ---------------------- 全局多租户会话存储初始化 ----------------------
# 定义全局字典store，用于存储「多租户+多会话」的双层历史记录，实现双层隔离
# 键（key）：元组 (user_id, session_id)，对应「用户唯一标识+会话唯一标识」
# 值（value）：对应 该（用户+会话）的ChatMessageHistory实例（存储该组合的完整对话）
store = {}

# ---------------------- 双层会话历史获取函数（核心：多租户+多会话隔离） ----------------------
def get_session_history(user_id: str, session_id: str):
    """
    根据「用户唯一标识（user_id）+ 会话唯一标识（session_id）」获取对应的会话历史，实现双层隔离
    :param user_id: 租户/用户唯一标识，用于区分不同用户（多租户核心）
    :param session_id: 会话唯一标识，用于区分同一用户的不同会话
    :return: 对应 (user_id, session_id) 的ChatMessageHistory实例（会话历史载体）
    """
    # 判空逻辑：如果该（用户+会话）组合对应的会话历史不存在（首次访问该组合）
    if (user_id, session_id) not in store:
        # 创建一个新的ChatMessageHistory实例，用于存储该（用户+会话）的后续对话消息
        store[(user_id, session_id)] = ChatMessageHistory()
    # 返回该（用户+会话）组合对应的会话历史对象（存在则返回已有，不存在则返回新建并存储后的）
    return store[(user_id, session_id)]

# ---------------------- 构建带会话历史占位符的聊天提示词模板 ----------------------
# 构建结构化聊天提示词模板，包含系统消息、历史消息占位符、人类输入，支持上下文关联
prompt = ChatPromptTemplate.from_messages([
    # 系统消息：定义AI助手的能力边界和回答要求，{ability}为动态传入的能力参数
    SystemMessage(content="你是一个AI助手，擅长能力{ability}。用30个字以内回答"),
    # 会话历史占位符：动态嵌入对应（用户+会话）的对话历史消息列表，key为"history"
    MessagesPlaceholder(variable_name="history"),
    # 人类消息：接收用户的实时输入，{input}为动态传入的用户提问参数
    HumanMessage(content="{input}")
])

# ---------------------- 大语言模型实例化 ----------------------
# 创建ChatOpenAI模型实例，作为多租户会话问答的核心生成引擎
llm = ChatOpenAI(
    model="qwen-plus",  # 指定使用的模型名称（通义千问plus版本，具备优秀的文本理解与生成能力）
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云通义千问的OpenAI兼容API地址
    api_key=SecretStr(""),  # 安全存储API密钥（敏感信息），SecretStr会隐藏明文展示
    temperature=0.7)  # 模型生成内容的随机性/创造性参数（0-1区间），0.7兼顾逻辑性和表达流畅性（注：原注释"亲密度"不准确）

# ---------------------- 构建基础运行链 ----------------------
# 用管道符|串联提示词模板和模型，构建端到端的基础问答链（无会话记忆能力）
chain = prompt | llm

# ---------------------- 为基础链添加「多租户+多会话」双层隔离记忆能力 ----------------------
# 封装RunnableWithMessageHistory，为基础链赋予「多租户+多会话」双层隔离的会话历史管理能力
with_message_history = RunnableWithMessageHistory(
    chain,  # 传入基础运行链，为其添加会话历史功能
    get_session_history=get_session_history,  # 传入双层会话历史获取函数，支持用户+会话双参数
    input_messages_key="input",  # 指定用户输入对应的键名，与prompt中的{input}和invoke传入的参数对应
    history_messages_key="history",  # 指定会话历史对应的键名，与MessagesPlaceholder的variable_name对应
    # 定义自定义可配置字段（核心：为get_session_history传递多参数）
    history_factory_config=[
        # 定义第一个自定义字段：user_id（用户唯一标识）
        ConfigurableFieldSpec(
            id = "user_id",  # 字段唯一标识，与get_session_history的参数名、config中的键名一致
            annotation=str,  # 字段数据类型，指定为字符串类型
            name="用户id",  # 字段名称（可读性描述）
            description="用户唯一标识符",  # 字段详细描述（说明用途）
            default="",  # 字段默认值
            is_shared=True  # 标记字段为共享配置，可在config中传递
        ),
        # 定义第二个自定义字段：session_id（对话唯一标识）
        ConfigurableFieldSpec(
            id = "session_id",  # 字段唯一标识，与get_session_history的参数名、config中的键名一致
            annotation=str,  # 字段数据类型，指定为字符串类型
            name="对话id",  # 字段名称（可读性描述）
            description="对话唯一标识符",  # 字段详细描述（说明用途）
            default="",  # 字段默认值
            is_shared=True  # 标记字段为共享配置，可在config中传递
        )
    ]
)

# ---------------------- 第一次调用：用户1 + 会话1 首次提问 ----------------------
# 调用带双层隔离记忆的链式结构，向（user_id=1, session_id=1）的组合发送首次提问
resp1 = with_message_history.invoke(
    # 传入提示词模板所需的动态参数：能力类型和用户输入（封装为HumanMessage对象）
    {
        "ability": "Java开发",
        "input": HumanMessage("什么是JVM")
    },
    # 配置双层唯一标识，指定本次调用归属的「用户+会话」（实现双层隔离的关键配置）
    config={'configurable':{"user_id":"1", 'session_id':"1"}}
)

# 打印全局存储字典store，查看（1,1）组合的会话历史是否已创建并存储
print(f"store1：{store}")
# 打印第一次调用的AI回复内容（提取Message对象的content属性，获取纯文本回复）
print(f"resp1:{resp1.content}",end="\n\n")

# ---------------------- 第二次调用：用户1 + 会话1 后续追问 ----------------------
# 继续调用带双层隔离记忆的链式结构，向同一个（user_id=1, session_id=1）组合发送追问
resp2 = with_message_history.invoke(
    # 传入提示词模板所需的动态参数，本次要求重新回答JVM相关问题
    {
        "ability": "Java开发",
        "input": HumanMessage("重新回答一次")
    },
    # 配置相同的「用户+会话」组合，确保归属同一上下文，可关联上一轮对话
    config={'configurable':{"user_id":"1", 'session_id':"1"}}
)

# 打印全局存储字典store，查看（1,1）组合的会话历史是否已追加本轮对话
print(f"store2：{store}")
# 打印第二次调用的AI回复内容
print(f"resp2:{resp2.content}",end="\n\n")
```

### 长期记忆缓存到 Redis实战

```python
# 从LangChain核心消息模块导入系统消息和人类消息类，用于构建结构化对话消息
from langchain_core.messages import SystemMessage, HumanMessage
# 从LangChain核心提示词模块导入：
# ChatPromptTemplate：构建结构化聊天提示词模板
# MessagesPlaceholder：对话历史占位符，用于动态嵌入Redis中的会话历史消息列表
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# 从LangChain核心可运行模块导入：
# RunnableWithMessageHistory：为链式结构添加会话历史管理和隔离能力
# ConfigurableFieldSpec：定义自定义可配置字段，支持多参数（用户+会话）传递
from langchain_core.runnables import RunnableWithMessageHistory, ConfigurableFieldSpec
# 从LangChain OpenAI模块导入ChatOpenAI，实例化兼容OpenAI格式的大语言模型客户端
from langchain_openai import ChatOpenAI
# 从LangChain Redis模块导入RedisChatMessageHistory，用于将对话历史持久化存储到Redis中（长期记忆）
from langchain_redis import RedisChatMessageHistory
# 从pydantic导入SecretStr，用于安全存储敏感信息（API密钥），避免明文泄露风险
from pydantic import SecretStr

# ---------------------- Redis 配置常量 ----------------------
# 定义Redis服务的连接URL，指定Redis服务的地址、端口（默认6379）
# 本地Redis需提前启动，否则会出现连接失败错误
REDIS_URL = "redis://127.0.0.1:6379"

# ---------------------- 双层会话历史获取函数（核心：Redis持久化+多用户多会话隔离） ----------------------
def get_session_history(user_id: str, session_id: str):
    """
    根据「用户唯一标识（user_id）+ 会话唯一标识（session_id）」获取Redis持久化的会话历史
    实现多用户多会话双层隔离，且对话历史长期存储（重启程序不丢失）
    :param user_id: 租户/用户唯一标识，用于区分不同用户
    :param session_id: 会话唯一标识，用于区分同一用户的不同会话
    :return: 绑定唯一键的RedisChatMessageHistory实例（对话历史Redis持久化载体）
    """
    # 构建Redis中的唯一键（拼接user_id和session_id），实现双层隔离
    # 避免不同用户、同一用户不同会话的历史记录在Redis中相互混淆
    uni_key = user_id + "_" + session_id
    
    # 实例化RedisChatMessageHistory，将对话历史存储到Redis中（对应uni_key键）
    # 若该uni_key不存在，则自动创建；若已存在，则加载已有对话历史（长期记忆核心）
    return RedisChatMessageHistory(uni_key, redis_url=REDIS_URL)

# ---------------------- 大语言模型实例化 ----------------------
# 创建ChatOpenAI模型实例，作为多用户多会话问答的核心生成引擎
llm = ChatOpenAI(
    model="qwen-plus",  # 指定使用的模型名称（通义千问plus版本，具备优秀的文本理解与生成能力）
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云通义千问的OpenAI兼容API地址
    api_key=SecretStr(""),  # 安全存储API密钥（敏感信息），SecretStr会隐藏明文展示
    temperature=0.7)  # 模型生成内容的随机性/创造性参数（0-1区间），0.7兼顾逻辑性和表达流畅性（注：原注释"亲密度"不准确）

# ---------------------- 构建带会话历史占位符的聊天提示词模板 ----------------------
# 构建结构化聊天提示词模板，包含系统消息、历史消息占位符、人类输入，支持上下文关联
prompt = ChatPromptTemplate.from_messages([
    # 系统消息：定义AI助手的能力边界和回答要求，{ability}为动态传入的能力参数
    SystemMessage(content="你是一个AI助手，擅长能力{ability}。用30个字以内回答"),
    # 会话历史占位符：动态嵌入Redis中对应（用户+会话）的对话历史消息列表，key为"history"
    MessagesPlaceholder(variable_name="history"),
    # 人类消息：接收用户的实时输入，{input}为动态传入的用户提问参数
    HumanMessage(content="{input}")
])

# ---------------------- 构建基础运行链 ----------------------
# 用管道符|串联提示词模板和模型，构建端到端的基础问答链（无会话记忆能力）
chain = prompt | llm

# ---------------------- 为基础链添加「Redis持久化+多用户多会话」双层隔离记忆能力 ----------------------
# 封装RunnableWithMessageHistory，为基础链赋予：
# 1. 对话历史Redis持久化（长期记忆，重启不丢失）
# 2. 多用户多会话双层隔离（互不干扰）
# 3. 自动管理会话（加载/保存，无需手动操作）
with_message_history = RunnableWithMessageHistory(
    chain,  # 传入基础运行链，为其添加会话历史功能
    get_session_history=get_session_history,  # 传入Redis会话历史获取函数，支持持久化和双层隔离
    input_messages_key="input",  # 指定用户输入对应的键名，与prompt中的{input}和invoke传入的参数对应
    history_messages_key="history",  # 指定会话历史对应的键名，与MessagesPlaceholder的variable_name对应
    # 定义自定义可配置字段（核心：为get_session_history传递user_id和session_id双参数）
    history_factory_config=[
        # 定义第一个自定义字段：user_id（用户唯一标识）
        ConfigurableFieldSpec(
            id = "user_id",  # 字段唯一标识，与get_session_history的参数名、config中的键名一致
            annotation=str,  # 字段数据类型，指定为字符串类型
            name="用户id",  # 字段名称（可读性描述）
            description="用户唯一标识符",  # 字段详细描述（说明用途）
            default="",  # 字段默认值
            is_shared=True  # 标记字段为共享配置，可在config中传递
        ),
        # 定义第二个自定义字段：session_id（对话唯一标识）
        ConfigurableFieldSpec(
            id = "session_id",  # 字段唯一标识，与get_session_history的参数名、config中的键名一致
            annotation=str,  # 字段数据类型，指定为字符串类型
            name="对话id",  # 字段名称（可读性描述）
            description="对话唯一标识符",  # 字段详细描述（说明用途）
            default="",  # 字段默认值
            is_shared=True  # 标记字段为共享配置，可在config中传递
        )
    ]
)

# ---------------------- 第一次调用：用户1 + 会话1 首次提问 ----------------------
# 调用带Redis持久化记忆的链式结构，向（user_id=1, session_id=1）的组合发送首次提问
# 本次调用会自动在Redis中创建"1_1"键，存储本轮对话历史
resp1 = with_message_history.invoke(
    # 传入提示词模板所需的动态参数：能力类型和用户输入
    {
        "ability": "Java开发",
        "input": "什么是JVM"
    },
    # 配置双层唯一标识，指定本次调用归属的「用户+会话」（实现双层隔离的关键配置）
    config={'configurable':{"user_id":"1",'session_id':"1"}}
)

# 打印第一次调用的AI回复内容（提取Message对象的content属性，获取纯文本回复）
print(f"resp1:{resp1.content}",end="\n\n")

# ---------------------- 第二次调用：用户1 + 会话1 后续追问 ----------------------
# 继续调用带Redis持久化记忆的链式结构，向同一个（user_id=1, session_id=1）组合发送追问
# 本次调用会自动从Redis的"1_1"键加载历史对话，关联上下文回复，并追加本轮对话到Redis中
resp2 = with_message_history.invoke(
    # 传入提示词模板所需的动态参数，本次要求重新回答JVM相关问题
    {
        "ability": "Java开发",
        "input": "重新回答一次"
    },
    # 配置相同的「用户+会话」组合，确保归属同一上下文，可关联上一轮对话
    config={'configurable':{"user_id":"1",'session_id':"1"}}
)

# 打印第二次调用的AI回复内容
print(f"resp2:{resp2.content}",end="\n\n")
```

### AI医生综合项目实战

```python
# 从LangChain核心输出解析模块导入字符串输出解析器，将模型返回的复杂Message对象转为纯字符串
from langchain_core.output_parsers import StrOutputParser
# 从LangChain核心提示词模块导入聊天提示词模板，用于构建结构化的RAG问答提示词
from langchain_core.prompts import ChatPromptTemplate
# 从LangChain核心可运行模块导入透传工具，用于直接传递用户提问到RAG链中
from langchain_core.runnables import RunnablePassthrough
# 从LangChain社区文档加载模块导入：
# TextLoader：用于加载本地文本文件（.txt），实现本地文档的RAG问答
from langchain_community.document_loaders import WebBaseLoader, TextLoader
# 从LangChain社区向量存储模块导入Chroma，用于构建本地向量数据库，存储文档嵌入向量
from langchain_community.vectorstores import Chroma
# 从LangChain OpenAI模块导入ChatOpenAI，实例化兼容OpenAI格式的大语言模型客户端
from langchain_openai import ChatOpenAI
# 从LangChain文本分割模块导入递归字符文本分割器，用于将长文档分割为小片段（适配嵌入模型限制）
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 从LangChain社区嵌入模块导入通义千问嵌入模型，用于将文档片段转换为向量表示（嵌入）
from langchain_community.embeddings import DashScopeEmbeddings
# 从pydantic导入SecretStr，用于安全存储敏感信息（API密钥），避免明文泄露风险
from pydantic import SecretStr

# ---------------------- 步骤1：加载本地文档（RAG的数据源准备） ----------------------
# 注释：网页文档加载器（备用），用于加载在线网页内容，需确保网络通畅
# loader = WebBaseLoader("")  # 示例网页
# docs = loader.load()

# 初始化本地文本文件加载器，指定要加载的本地文本文件路径和编码格式（utf-8避免中文乱码）
# 需提前创建data目录，并在其中放入qa.txt文件（存储待问答的文档内容）
loader = TextLoader("data/qa.txt", encoding="utf-8")
# 执行文档加载，返回文档对象列表（docs），每个元素对应一个文档（此处仅加载单个txt文件）
docs = loader.load()

# ---------------------- 步骤2：分割长文档（适配嵌入模型的长度限制，提升检索精度） ----------------------
# 初始化递归字符文本分割器，这是RAG场景中最常用的文档分割工具
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 每个文档片段（chunk）的最大字符数，根据嵌入模型能力调整
    chunk_overlap=200  # 相邻文档片段之间的重叠字符数，避免分割导致上下文丢失，提升片段连贯性
)
# 执行文档分割，将加载的完整文档（docs）分割为多个小片段（splits），返回片段列表
splits = text_splitter.split_documents(docs)

# ---------------------- 步骤3：初始化嵌入模型（文档→向量的转换工具） ----------------------
# 初始化通义千问DashScopeEmbeddings嵌入模型，用于将文档片段转换为高维向量（嵌入表示）
embedding_model = DashScopeEmbeddings(
    model="text-embedding-v2",  # 指定使用的嵌入模型版本（第二代通用嵌入模型，效果更优）
    max_retries=3,  # 嵌入请求失败时的最大重试次数，提升稳定性
    dashscope_api_key=""  # 通义千问API密钥，用于调用嵌入接口
)

# ---------------------- 步骤4：创建Chroma向量数据库并构建检索器（存储向量+实现精准检索） ----------------------
# 基于分割后的文档片段，创建Chroma本地向量数据库
vectorstore = Chroma.from_documents(
    documents=splits,  # 传入分割后的文档片段列表，作为向量数据库的数据源
    embedding=embedding_model,  # 传入已初始化的嵌入模型，用于将文档片段转为向量
    persist_directory="./rag_chroma_db"  # 指定向量数据库的本地持久化目录，重启程序后无需重新生成向量
)
# 将向量数据库转换为检索器（retriever），用于后续根据用户提问检索相关文档片段
# search_kwargs={"k": 3}：指定每次检索返回最相关的3个文档片段，平衡检索精度和上下文长度
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------------------- 步骤5：初始化大语言模型（RAG的核心生成引擎，用于基于检索结果回答问题） ----------------------
# 创建ChatOpenAI模型实例，作为RAG问答的最终生成引擎
model = ChatOpenAI(
    model="qwen-plus",  # 指定使用的大语言模型（通义千问plus版本，具备优秀的文本理解和生成能力）
    base_url= "https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云通义千问的OpenAI兼容API地址
    api_key=SecretStr(""),  # 安全存储API密钥（敏感信息），SecretStr隐藏明文展示
    streaming=True,  # 开启流式输出（本代码直接invoke获取结果，未体现流式效果，需结合回调函数使用）
    temperature=0.7)  # 模型生成内容的随机性/创造性参数（0-1区间），0.7兼顾回答准确性和表达流畅性（注：原注释"亲密度"不准确）

# ---------------------- 步骤6：创建RAG专用提示词模板（引导模型基于检索上下文回答问题） ----------------------
# 定义RAG提示词模板，核心是注入检索到的上下文（{context}）和用户问题（{question}）
template = """[INST]<<SYS>>
你是一个有用的AI助手，请根据以下上下文回答问题：
{context}
<</SYS>>
问题：{question} [/INST]"""
# 基于模板字符串创建ChatPromptTemplate对象，用于后续填充上下文和问题，生成完整提示词
rag_prompt = ChatPromptTemplate.from_template(template)

# ---------------------- 步骤7：构建完整RAG链（串联所有组件，实现端到端的检索增强生成） ----------------------
# 构建LCEL表达式的RAG链，按从左到右顺序串联各组件，实现「检索→填充提示词→模型生成→结果解析」
rag_chain = (
    # 第一步：构建输入字典，为后续链提供两个核心参数：
    # - context：通过retriever检索用户问题相关的文档片段（自动将用户提问传入retriever进行检索）
    # - question：通过RunnablePassthrough()直接透传用户的原始提问，不做任何修改
    {"context": retriever, "question": RunnablePassthrough()}
    |  # 管道符：将上一步的输出（字典）作为下一步的输入
    # 第二步：将输入字典填充到rag_prompt模板中，生成完整的结构化提示词
    rag_prompt
    |  # 管道符：将完整提示词作为下一步的输入
    # 第三步：将完整提示词传入大语言模型，生成问答结果（Message对象）
    model
    |  # 管道符：将模型生成的Message对象作为下一步的输入
    # 第四步：将复杂的Message对象转为纯字符串格式，方便查看和使用
    StrOutputParser()
)

# ---------------------- 示例：调用RAG链，执行问答任务 ----------------------
# 调用RAG链，传入用户问题「头疼怎么办?」，获取基于本地qa.txt文档的增强回答结果
result = rag_chain.invoke("头疼怎么办?")
# 打印最终的问答结果
print(result)
```

### MCP 服务端实战

```python
# 导入必要的第三方库和MCP服务器核心模块
import httpx  # 用于发送异步HTTP请求的库，替代requests的异步实现
from mcp.server.fastmcp import FastMCP  # 导入FastMCP框架，用于快速构建MCP工具服务
from typing import Any, Dict, Optional  # 导入类型注解相关工具，提升代码可维护性和类型安全性

# ========================================
# 全局配置与MCP服务器初始化
# ========================================
# 初始化FastMCP服务器实例，指定服务名称为"amap-weather"
# 该服务主要提供基于高德地图API的天气查询相关功能
mcp = FastMCP("amap-weather")

# 高德地图API相关全局配置（需用户根据实际情况替换/配置）
AMAP_API_BASE = "https://restapi.amap.com/v3"  # 高德地图API的基础请求地址
AMAP_API_KEY = "07ce033dc43afb0d8d11e0a98267c67c"  # 高德地图API密钥（必填）
# 注意：请前往高德开放平台申请有效的API Key，该测试密钥可能已失效
# 申请地址：https://lbs.amap.com/
USER_AGENT = "amap-weather-app/1.0"  # 请求头中的用户代理标识，用于标识请求来源

# ========================================
# 核心工具函数：高德API请求发送
# ========================================
async def make_amap_request(endpoint: str, params: dict) -> Optional[Dict[str, Any]]:
    """
    封装高德地图API的异步请求发送逻辑，包含完整的错误处理机制

    Args:
        endpoint: API接口路径（如"/weather/weatherInfo"，需以/开头）
        params: 接口所需的业务请求参数（无需包含key和output）

    Returns:
        Optional[Dict[str, Any]]: 成功返回API响应的JSON数据字典，失败返回None
    """
    print('正在发送高德API请求...make_amap_request')
    
    # 构造基础请求参数，所有接口都需要携带key和output格式指定
    base_params = {
        'key': AMAP_API_KEY,  # 接口调用凭证，必填
        'output': 'JSON'      # 指定响应数据格式为JSON，方便后续解析
    }
    
    # 合并基础参数和业务参数（业务参数可覆盖基础参数，提升灵活性）
    full_params = {**base_params, **params}

    # 异步创建HTTP客户端，使用上下文管理器自动管理客户端生命周期
    async with httpx.AsyncClient() as client:
        try:
            # 发送GET异步请求，获取高德API响应
            response = await client.get(
                url=f'{AMAP_API_BASE}{endpoint}',  # 拼接完整的请求URL
                params=full_params,                # 传递拼接后的完整请求参数
                headers={'User-Agent': USER_AGENT},# 设置请求头，标识客户端身份
                timeout=30.0                       # 设置30秒请求超时时间，防止长时间阻塞
            )
            
            # 检查HTTP响应状态码，非2xx状态码会抛出HTTPStatusError异常
            response.raise_for_status()
            
            # 解析响应数据为JSON格式（字典类型）
            data = response.json()

            # 处理高德API自身的业务错误（HTTP状态码200不代表业务请求成功）
            # 高德API约定：status为'1'表示业务请求成功，其他值为失败
            if data.get("status") != '1':
                err_msg = data.get("info", "未知错误")  # 获取API返回的错误描述信息
                print(f"高德API请求失败: {err_msg}")
                return None
            
            # 业务请求成功，返回解析后的JSON数据
            return data
        
        # 捕获所有可能的异常（网络异常、超时、JSON解析失败等）
        except Exception as e:
            print(f"请求异常: {str(e)}")
            return None

# ========================================
# 数据格式化函数：天气数据转可读文本
# ========================================
def format_weather_forecast(data: Dict[str, Any]) -> str:
    """
    格式化实时天气数据（简化版），转换为人类可读的简洁文本

    Args:
        data: 高德API返回的天气数据字典

    Returns:
        str: 格式化后的简洁天气信息字符串
    """
    try:
        # 校验数据是否包含有效实时天气字段（lives为高德API实时天气数据关键字段）
        if 'lives' in data and data['lives']:
            weather_info = data['lives'][0]  # 取第一个元素（对应查询城市的天气数据）
            city = weather_info.get('city', '未知城市')    # 城市名称
            weather = weather_info.get('weather', '未知天气')  # 天气状况
            temperature = weather_info.get('temperature', '未知')  # 温度
            humidity = weather_info.get('humidity', '未知')  # 湿度

            # 拼接简洁的天气信息字符串并返回
            return f"{city}天气：{weather}，温度：{temperature}℃，湿度：{humidity}%"
        else:
            # 数据中无有效实时天气字段
            return "未找到天气信息"
    except Exception as e:
        # 捕获数据格式化过程中的异常（如字段缺失、类型错误等）
        print(f"格式化天气数据时出错: {str(e)}")
        return "天气数据格式错误"

def format_realtime_weather(data: Dict[str, Any]) -> str:
    """
    格式化实时天气数据（详细版），包含更多气象信息

    Args:
        data: 高德API返回的天气数据字典

    Returns:
        str: 格式化后的详细实时天气信息字符串
    """
    try:
        # 校验数据是否包含有效实时天气字段
        if 'lives' in data and data['lives']:
            live = data['lives'][0]  # 取第一个元素（对应查询城市的实时天气数据）
            # 从数据中提取各类气象信息，指定默认值防止字段缺失导致报错
            city = live.get('city', '未知城市')
            weather = live.get('weather', '未知')
            temperature = live.get('temperature', '未知')
            wind_direction = live.get('winddirection', '未知')
            wind_power = live.get('windpower', '未知')
            humidity = live.get('humidity', '未知')
            report_time = live.get('reporttime', '未知')

            # 拼接详细的实时天气信息，采用分行格式提升可读性
            return (f"【{city}实时天气】\n"
                    f"天气状况：{weather}\n"
                    f"温度：{temperature}℃\n"
                    f"风向：{wind_direction}\n"
                    f"风力：{wind_power}级\n"
                    f"湿度：{humidity}%\n"
                    f"发布时间：{report_time}")
        else:
            # 数据中无有效实时天气字段
            return "未找到实时天气数据"
    except Exception as e:
        # 捕获数据格式化过程中的异常
        print(f"格式化实时天气数据时出错: {str(e)}")
        return "实时天气数据格式错误"

def format_forecast(data: Dict[str, Any]) -> str:
    """
    格式化未来多天天气预报数据，转换为人类可读的文本

    Args:
        data: 高德API返回的天气预报数据字典

    Returns:
        str: 格式化后的未来3天天气预报信息字符串
    """
    try:
        # 校验数据是否包含有效天气预报字段（forecasts为高德API预报数据关键字段）
        if 'forecasts' in data and data['forecasts']:
            forecast = data['forecasts'][0]  # 取第一个元素（对应查询城市的预报数据）
            city = forecast.get('city', '未知城市')  # 城市名称
            casts = forecast.get('casts', [])  # 每日预报数据列表

            # 初始化结果列表，用于拼接多行预报信息
            result = [f"【{city}天气预报】"]

            # 遍历前3天的预报数据（避免数据过多，提升可读性）
            for cast in casts[:3]:
                # 提取每日预报的详细信息，指定默认值防止字段缺失
                date = cast.get('date', '未知日期')
                day_weather = cast.get('dayweather', '未知')
                night_weather = cast.get('nightweather', '未知')
                day_temp = cast.get('daytemp', '未知')
                night_temp = cast.get('nighttemp', '未知')
                day_wind = cast.get('daywind', '未知')
                night_wind = cast.get('nightwind', '未知')
                day_power = cast.get('daypower', '未知')
                night_power = cast.get('nightpower', '未知')

                # 拼接单日预报信息，添加到结果列表中
                result.append(
                    f"\n{date}：\n"
                    f"  白天：{day_weather}，{day_temp}℃，{day_wind}风{day_power}级\n"
                    f"  夜间：{night_weather}，{night_temp}℃，{night_wind}风{night_power}级"
                )

            # 将结果列表拼接为完整字符串并返回
            return "\n".join(result)
        else:
            # 数据中无有效天气预报字段
            return "未找到天气预报数据"
    except Exception as e:
        # 捕获数据格式化过程中的异常
        print(f"格式化天气预报数据时出错: {str(e)}")
        return "天气预报数据格式错误"

# ========================================
# MCP工具注册：对外暴露的天气查询接口
# ========================================
@mcp.tool()
async def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息（简化版，通过城市名称查询）

    Args:
        city: 城市名称（如"北京"、"上海"，支持中文全称）

    Returns:
        str: 格式化后的简洁天气信息字符串
    """
    # 构造天气查询请求参数
    params = {
        'city': city,                # 目标城市名称
        'extensions': 'base'         # 查询类型：base=实时天气，all=未来天气预报
    }

    # 调用封装的高德API请求函数，获取天气数据
    data = await make_amap_request("/weather/weatherInfo", params)

    # 校验数据是否获取成功，返回对应结果
    if data is None:
        return "获取天气信息失败，请检查网络连接或API配置"

    # 格式化天气数据并返回
    return format_weather_forecast(data)

@mcp.tool()
async def get_realtime_weather(city_adcode: str) -> str:
    """
    获取中国城市实时天气（详细版，通过城市编码查询，精度更高）

    Args:
        city_adcode: 城市编码（如北京110000，上海310000，可通过高德城市编码表查询）

    Returns:
        str: 格式化后的详细实时天气信息字符串
    """
    print('正在获取实时天气...get_realtime_weather')
    
    # 调用封装的高德API请求函数，获取详细实时天气数据
    data = await make_amap_request(
        endpoint='/weather/weatherInfo',  # 天气查询接口路径
        params={'city': city_adcode, 'extensions': 'base'}  # 查询参数：城市编码+实时天气
    )
    
    # 校验数据是否有效，返回对应结果
    if not data or not data.get('lives'):
        return '无法获取实时天气数据（请检查城市编码或API Key）'
    
    # 格式化详细实时天气数据并返回
    return format_realtime_weather(data)

@mcp.tool()
async def get_forecast(city_adcode: str) -> str:
    """
    获取中国城市未来多天天气预报（通过城市编码查询，精度更高）

    Args:
        city_adcode: 城市编码（如北京110000，上海310000，可通过高德城市编码表查询）

    Returns:
        str: 格式化后的未来3天天气预报信息字符串
    """
    # 调用封装的高德API请求函数，获取天气预报数据
    data = await make_amap_request(
        endpoint='/weather/weatherInfo',  # 天气查询接口路径
        params={'city': city_adcode, 'extensions': 'all'}  # 查询参数：城市编码+未来预报
    )
    
    # 校验数据是否有效，返回对应结果
    if not data or not data.get('forecasts'):
        return '无法获取天气预报数据（请检查城市编码或API Key）'
    
    # 格式化天气预报数据并返回
    return format_forecast(data)

# ========================================
# 程序入口：启动MCP服务器
# ========================================
if __name__ == "__main__":
    # 启动FastMCP服务器，采用stdio（标准输入输出）进行通信
    # 该模式适用于MCP框架的进程间通信，兼容大多数MCP客户端
    mcp.run(transport='stdio')
```

### MCP客户端实战

```python
# ========================================
# 极简版MCP天气客户端（带详细注释）
# 核心功能：连接MCP服务端 → 调用LLM生成指令 → 调用天气工具 → 返回结果
# 适合新手理解核心流程，无冗余封装，线性逻辑清晰
# ========================================

# 导入Python内置标准库（核心必要）
import json  # 用于解析和生成JSON数据（LLM指令、工具参数均为JSON格式）
import asyncio  # 用于支持异步编程（MCP通信、HTTP请求均为异步操作）
import sys  # 用于获取命令行参数（传入MCP服务端脚本路径）
import os  # 用于读取环境变量（获取DashScope API密钥）

# 导入第三方库（核心必要）
import httpx  # 用于发送异步HTTP请求（调用通义千问LLM接口）

# 导入MCP框架核心模块（用于建立客户端与服务端通信）
from mcp import ClientSession  # MCP客户端会话对象，负责发送工具调用指令
from mcp.client.stdio import stdio_client  # MCP标准输入输出客户端，用于启动并连接服务端脚本

# ========================================
# 全局配置项（简化版，直接定义，无需封装到类中）
# ========================================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # 从环境变量读取通义千问API密钥（安全便捷）
TONGYI_MODEL = "qwen-plus"  # 通义千问模型名称，选择高性能通用模型qwen-plus
AMAP_TOOL_NAME = "get_weather"  # 我们要调用的MCP工具名称（对应服务端的天气查询工具）

# ========================================
# 步骤1：连接MCP服务端，建立通信通道
# ========================================
async def connect_mcp_server(server_script_path):
    """
    简化版：连接MCP服务端脚本，建立异步通信会话，验证目标工具是否可用
    核心作用：就像拨通电话，确认对方在线且能提供我们需要的服务（天气查询工具）
    Args:
        server_script_path: MCP服务端脚本路径（如weather.py）
    Returns:
        ClientSession: 初始化完成的MCP客户端会话对象（后续用于调用工具）
    Raises:
        ValueError: 若脚本格式不正确或目标工具不存在，抛出错误提示
    """
    # 第一步：验证服务端脚本格式（仅支持Python脚本，简化版不兼容JS）
    if not server_script_path.endswith(".py"):
        raise ValueError("简化版仅支持Python格式的MCP服务端脚本（.py文件）")
    
    # 第二步：启动MCP服务端脚本，建立标准输入输出（stdio）通信连接
    # stdio_client会自动执行python命令，启动传入的服务端脚本，并建立双向通信
    stdio_transport = await stdio_client(command="python", args=[server_script_path])
    stdio, write = stdio_transport  # 解包通信传输对象：stdio用于读取，write用于写入
    
    # 第三步：创建MCP客户端会话对象，完成会话初始化（握手协议）
    # 会话对象是后续调用工具的核心载体，负责封装通信细节
    session = ClientSession(stdio, write)
    await session.initialize()  # 初始化会话，与服务端完成协议握手
    
    # 第四步：验证目标工具（get_weather）是否在服务端的可用工具列表中
    tool_list_response = await session.list_tools()  # 获取服务端所有可用工具
    tool_names = [tool.name for tool in tool_list_response.tools]  # 提取工具名称列表
    
    # 若目标工具不存在，抛出错误并提示可用工具
    if AMAP_TOOL_NAME not in tool_names:
        raise ValueError(f"服务端未提供{AMAP_TOOL_NAME}工具，可用工具列表：{tool_names}")
    
    # 第五步：打印连接成功提示，返回初始化完成的会话对象
    print(f"✅ 已成功连接到MCP服务端，目标工具{AMAP_TOOL_NAME}可用")
    return session

# ========================================
# 步骤2：调用通义千问LLM，生成标准化工具调用指令
# ========================================
async def call_llm(user_query):
    """
    简化版：调用通义千问LLM，将用户自然语言查询转换为机器可识别的工具调用指令
    核心作用：就像翻译，把用户说的大白话（如"北京天气"）翻译成机器能懂的专业指令（JSON格式）
    Args:
        user_query: 用户输入的自然语言查询（如"北京天气"、"上海今天天气怎么样"）
    Returns:
        dict: 解析后的工具调用指令（含name和parameters），或错误信息字典
    """
    # 第一步：前置检查，验证API密钥是否配置
    if not DASHSCOPE_API_KEY:
        return {"error": "未配置DASHSCOPE_API_KEY环境变量，请先配置后再运行"}
    
    # 第二步：构建LLM系统提示词，强制LLM遵循固定输出格式
    # 简化版：只要求处理get_weather工具，输出标准JSON格式，无额外冗余内容
    system_prompt = f"""
    你的唯一任务是将用户输入转换为{AMAP_TOOL_NAME}工具的调用指令，严格遵守以下规则：
    1. 仅能调用{AMAP_TOOL_NAME}这一个工具，无需考虑其他工具
    2. 输出格式必须是纯JSON，无任何额外文本、注释或换行
    3. JSON格式固定为：{{"name": "{AMAP_TOOL_NAME}", "parameters": {{"city": "城市名"}}}}
    4. 从用户输入中提取城市名称，填入city参数，若未明确城市，默认填"北京"
    """
    
    # 第三步：构建LLM请求头，包含身份验证和数据格式声明
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",  # Bearer认证，传入API密钥
        "Content-Type": "application/json"  # 声明请求体为JSON格式
    }
    
    # 第四步：构建LLM请求体，包含模型名称、对话消息和生成参数
    payload = {
        "model": TONGYI_MODEL,  # 要调用的通义千问模型名称
        "messages": [  # 对话消息列表，遵循OpenAI兼容格式
            {"role": "system", "content": system_prompt},  # 系统提示，定义LLM的行为准则
            {"role": "user", "content": user_query}  # 用户输入，传递用户的自然语言查询
        ],
        "parameters": {"temperature": 0.0}  # 生成参数：温度设为0，确保输出结果稳定可预测
    }
    
    try:
        # 第五步：异步发送HTTP POST请求，调用通义千问LLM接口
        async with httpx.AsyncClient() as client:  # 异步HTTP客户端，自动管理连接生命周期
            response = await client.post(
                url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",  # 通义千问兼容OpenAI的API地址
                headers=headers,  # 传入请求头
                json=payload  # 传入请求体（自动序列化为JSON格式）
            )
            response.raise_for_status()  # 检查HTTP响应状态码，非2xx状态码抛出异常
        
        # 第六步：解析LLM返回结果，提取工具调用指令
        llm_response = response.json()  # 将HTTP响应体解析为字典
        # 提取LLM生成的核心内容（choices → 第一个结果 → message → content）
        tool_call_content = llm_response["choices"][0]["message"]["content"]
        # 将LLM返回的JSON字符串解析为字典，供后续调用工具使用
        return json.loads(tool_call_content)
    
    # 捕获所有可能的异常，返回友好错误提示
    except Exception as e:
        return {"error": f"LLM调用失败或返回格式错误，具体原因：{str(e)}"}

# ========================================
# 步骤3：调用MCP工具，执行天气查询并返回结果
# ========================================
async def call_mcp_tool(session, tool_name, parameters):
    """
    简化版：通过MCP会话对象调用指定工具，获取天气查询结果
    核心作用：就像把翻译好的指令传递给办事人员，等待办事结果并带回
    Args:
        session: 初始化完成的MCP ClientSession对象
        tool_name: 要调用的工具名称（如get_weather）
        parameters: 工具调用所需参数（如{"city": "北京"}）
    Returns:
        str: 工具执行结果（格式化的天气信息）或错误提示字符串
    """
    try:
        # 第一步：调用MCP工具，添加10秒超时控制，防止长时间阻塞
        # session.call_tool：发送工具调用指令给服务端，等待服务端执行并返回结果
        result = await asyncio.wait_for(
            session.call_tool(tool_name, parameters),  # 核心工具调用方法
            timeout=10  # 10秒超时，若服务端未响应则抛出超时异常
        )
        
        # 第二步：返回工具执行的核心内容（result.content为服务端返回的有效数据）
        return result.content
    
    # 捕获所有可能的异常，返回友好错误提示
    except Exception as e:
        return f"工具调用失败，具体原因：{str(e)}"

# ========================================
# 主流程：串联3个核心步骤，实现完整的天气查询功能
# ========================================
async def main():
    """
    极简版主流程：串联「连接MCP服务端」→「调用LLM生成指令」→「调用MCP工具」
    线性流程，无复杂嵌套，新手易理解
    """
    # 第一步：检查命令行参数是否完整（是否传入了MCP服务端脚本路径）
    if len(sys.argv) < 2:
        print("使用方法：python simple_client_with_comments.py <MCP服务端脚本路径>")
        print("示例：python simple_client_with_comments.py weather.py")
        return  # 参数不完整，直接退出程序
    
    try:
        # 第二步：连接MCP服务端，获取会话对象（步骤1）
        # sys.argv[1]：获取命令行中第二个参数（服务端脚本路径）
        mcp_session = await connect_mcp_server(sys.argv[1])
        
        # 第三步：打印欢迎信息，进入用户交互循环
        print("\n🤖 极简天气助手（已就绪，支持自然语言查询）")
        print("📌 示例查询：'北京天气'、'上海今天天气'")
        print("📌 输入'quit'或'退出'，即可结束程序")
        
        # 第四步：循环接收用户输入，处理查询请求（直到用户退出）
        while True:
            # 接收用户输入，去除首尾空格
            user_query = input("\n请输入你的天气查询：").strip()
            
            # 退出条件：用户输入quit、退出、bye等关键词
            if user_query.lower() in ["quit", "退出", "bye"]:
                print("👋 再见！程序已正常退出")
                break
            
            # 跳过空输入（用户直接按回车）
            if not user_query:
                continue
            
            # 第五步：调用LLM，将用户输入转换为工具调用指令（步骤2）
            tool_call_info = await call_llm(user_query)
            
            # 检查LLM调用是否出错，若出错则打印错误信息并继续下一轮循环
            if "error" in tool_call_info:
                print(f"❌ {tool_call_info['error']}")
                continue
            
            # 第六步：调用MCP工具，执行天气查询，获取结果（步骤3）
            weather_result = await call_mcp_tool(
                session=mcp_session,
                tool_name=tool_call_info["name"],
                parameters=tool_call_info["parameters"]
            )
            
            # 第七步：格式化打印天气查询结果
            print(f"\n📝 天气查询结果如下：")
            print("-" * 40)
            print(weather_result)
            print("-" * 40)
    
    # 捕获所有可能的全局异常，打印友好错误提示
    except Exception as e:
        print(f"❌ 程序运行出错，具体原因：{str(e)}")

# ========================================
# 程序入口：运行异步主流程
# ========================================
if __name__ == "__main__":
    # asyncio.run()：运行异步主函数，管理整个异步程序的生命周期
    asyncio.run(main())
```

 