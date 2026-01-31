print("hello world")
import os

from langchain_openai import ChatOpenAI
print("检查环境变量：")
print(f"WEAVIATE_URL: {os.getenv('WEAVIATE_URL', '未设置！')}")
print(f"WEAVIATE_API_KEY: {'已设置' if os.getenv('WEAVIATE_API_KEY') else '未设置！'}")
# 测试调用
modelscope_qwen = ChatOpenAI(
    model="Qwen/Qwen3-8B",  # modelscope/qwen-turboQwen/Qwen3-8B 可选：qwen-max（最强）、qwen-plus（均衡）、qwen-long（长文本）
    temperature=0,
    openai_api_key=os.environ.get("MODELSCOPE_API_KEY", "not_provided"),
    openai_api_base="https://api-inference.modelscope.cn/v1",
    streaming=True,
)
try:
    response = modelscope_qwen.invoke("你好，请介绍一下你自己。")
    print(response.content)
except Exception as e:
    print("调用失败:", str(e))