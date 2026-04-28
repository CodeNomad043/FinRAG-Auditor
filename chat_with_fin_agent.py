import os
import yfinance as yf
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_community.llms import Ollama
from langchain.tools import tool
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 

# --- [工具定义] ---
@tool
def get_realtime_price(ticker: str) -> str:
    """查询指定股票代码的当前实时价格。输入应为股票代码如 'AAPL'。"""
    try:
        stock = yf.Ticker(ticker)
        price = stock.fast_info['last_price']
        return f"{ticker}当前股价: {price:.2f} USD"
    except: return "价格查询失败"

@tool
def financial_calculator(expression: str) -> str:
    """金融计算器。输入必须是纯数字运算式，例如 '31370 / 267.61'。不要输入字母。"""
    try:
        # 使用安全的 eval 处理纯数学运算
        res = eval(expression, {"__builtins__": None}, {})
        return f"计算结果: {res}"
    except: return "计算格式错误，请确保只输入数字和运算符"

def main():
    model_name = "BAAI/bge-small-en-v1.5"
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})

    vector_db = Milvus(
        embedding_function=embeddings,
        connection_args={"host": "127.0.0.1", "port": "19530"},
        collection_name="Financial_Reports",
    )

    # 核心优化：提高模型预测稳定性
    llm = Ollama(
        model="qwen2.5:3b", 
        base_url="http://127.0.0.1:11434",
        temperature=0,  # 强制降低随机性
        stop=["\nObservation:"] # 强制停止标志，防止模型替工具说话
    ) 

    retriever_tool = create_retriever_tool(
        vector_db.as_retriever(search_kwargs={"k": 5}),
        "search_docs",
        "搜索财报中的年度研发投入(R&D)、营收等历史数字。"
    )
    
    tools = [retriever_tool, get_realtime_price, financial_calculator]

    # --- [核心修改：换成更简单的 ReAct 协议] ---
    # react-agent 比 structured-chat 更适合小型模型
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=5 # 限制最大尝试次数，防止死循环
    )

    print("\n🚀 Fin-Agent (稳定性增强版) 已就绪！")
    
    while True:
        query = input("\n👤 指令: ")
        if query.lower() == 'exit': break
        try:
            # 这里的输入变量名必须是 "input"
            agent_executor.invoke({"input": query})
        except Exception as e:
            print(f"❌ 运行异常: {e}")

if __name__ == "__main__":
    main()