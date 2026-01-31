import logging
import os
from typing import List

import weaviate
from constants import WEAVIATE_DOCS_INDEX_NAME
from langchain.indexes import SQLRecordManager, index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Weaviate
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_embeddings_model() -> Embeddings:
    # 保持原样， text-embedding-3-small 性价比很高
    return OpenAIEmbeddings(model="text-embedding-3-small")
from langchain.embeddings import HuggingFaceEmbeddings



def load_local_txt_docs(path: str) -> List[Document]:
    """
    从本地目录加载所有 .txt 文件
    """
    logger.info(f"正在从目录加载文件: {path}")
    # glob="./**/*.txt" 表示递归查找目录下所有 txt
    loader = DirectoryLoader(
        path, 
        glob="./**/*.txt", 
        loader_cls=TextLoader,
        show_progress=True,
        loader_kwargs={'encoding': 'utf-8'}  # 关键：指定 UTF-8 编码
    )
    return loader.load()

def ingest_docs():
    # 环境变量配置
    WEAVIATE_URL = os.environ["WEAVIATE_URL"]
    WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
    RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]

    # 1. 初始化模型和工具
    # embedding = get_embeddings_model()
    embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # 2. 连接 Weaviate
    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    )
    
    vectorstore = Weaviate(
        client=client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=embedding,
        by_text=False,
        attributes=["source", "title"],
    )

    # 3. 增量索引管理器 (非常重要，避免重复入库)
    record_manager = SQLRecordManager(
        f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
    )
    record_manager.create_schema()

    # 4. 加载本地数据
    # 假设你的 txt 文件放在当前目录下的 'data' 文件夹中
    raw_docs = load_local_txt_docs("E:\\Program\\chat-langchain-study\\data")
    
    # 5. 转换与清洗
    docs_transformed = text_splitter.split_documents(raw_docs)
    
    # 补充元数据补丁（Weaviate 要求查询的属性必须存在）
    for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = doc.metadata.get("path", "local_file")
        if "title" not in doc.metadata:
            # 取文件名作为标题
            file_name = os.path.basename(doc.metadata.get("source", ""))
            doc.metadata["title"] = file_name

    # 6. 执行索引
    indexing_stats = index(
        docs_transformed,
        record_manager,
        vectorstore,
        cleanup="full", # 如果本地文件删除了，数据库也会同步删除
        source_id_key="source",
    )

    logger.info(f"索引统计结果: {indexing_stats}")

if __name__ == "__main__":
    ingest_docs()