import os
from operator import itemgetter
from typing import Dict, List, Optional, Sequence

import weaviate
from constants import WEAVIATE_DOCS_INDEX_NAME
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ingest import get_embeddings_model
from langchain_community.vectorstores import Weaviate
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI  # 使用 OpenAI 客户端兼容魔塔/通义
from langsmith import Client
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# --- 配置 ---
# 请确保环境变量中有 MODELSCOPE_API_KEY (魔塔/通义的Key)
# WEAVIATE_URL 和 WEAVIATE_API_KEY 也必须存在
WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
MODELSCOPE_API_KEY = os.environ.get("MODELSCOPE_API_KEY")

if not MODELSCOPE_API_KEY:
    raise ValueError("请在 .env 文件中配置 MODELSCOPE_API_KEY")

# --- 提示词模板 (Prompt) ---
# 这里保留了原版的英文提示词，如果你希望它用中文回答，建议把 "Generate... in English" 改为 "用中文回答"
RESPONSE_TEMPLATE = """\
You are an expert programmer and problem-solver, tasked with answering any question \
about Langchain.

Generate a comprehensive and informative answer of 80 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] notation. Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the sentence or paragraph that reference them - do not put them all at the end. If \
different results refer to different entities within the same name, write separate \
answers for each entity.

You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." Don't try to make up an answer.

Anything between the following `context` html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. 

<context>
    {context} 
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer.
"""

REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""


client = Client()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]


# --- 核心修改 1: 获取检索器 ---
def get_retriever() -> BaseRetriever:
    # 建立 Weaviate 客户端
    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    )
    # 初始化向量库
    vectorstore = Weaviate(
        client=client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=get_embeddings_model(), # 注意：这里要确保 ingest.py 里的 embedding 模型和你现在能用的匹配
        by_text=False,
        attributes=["source", "title"],
    )
    return vectorstore.as_retriever(search_kwargs=dict(k=6))


# --- 核心修改 2: 链的构建 ---
def create_retriever_chain(
    llm: LanguageModelLike, retriever: BaseRetriever
) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    
    # 使用 LLM 重新改写问题（处理多轮对话）
    condense_question_chain = (
        CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )
    
    conversation_chain = condense_question_chain | retriever
    
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
            RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            )
            | retriever
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def serialize_history(request: ChatRequest):
    chat_history = request.chat_history or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


def create_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
    retriever_chain = create_retriever_chain(
        llm,
        retriever,
    ).with_config(run_name="FindDocs")
    
    context = (
        RunnablePassthrough.assign(docs=retriever_chain)
        .assign(context=lambda x: format_docs(x["docs"]))
        .with_config(run_name="RetrieveDocs")
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    
    response_synthesizer = (prompt | llm | StrOutputParser()).with_config(
        run_name="GenerateResponse"
    )
    
    return (
        RunnablePassthrough.assign(chat_history=serialize_history)
        | context
        | response_synthesizer
    )

# --- 核心修改 3: 初始化魔塔/通义 LLM ---
# 使用 OpenAI 兼容模式连接 DashScope
# model 可以选 "qwen-turbo" (通常免费/便宜), "qwen-plus", "qwen-max"
modelscope_llm = ChatOpenAI(
    model="qwen-turbo", 
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key=MODELSCOPE_API_KEY,
    temperature=0,
    streaming=True
)

# 初始化检索器
retriever = get_retriever()

# 构建最终的链
answer_chain = create_chain(modelscope_llm, retriever)

# FastAPI 路由
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # 将 Pydantic model 转为 dict 传给 chain
    input_dict = {"question": request.question, "chat_history": request.chat_history}
    return await answer_chain.ainvoke(input_dict)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)