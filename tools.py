import yfinance as yf
from langchain.tools import tool

@tool
def get_realtime_price(ticker: str) -> str:
    """
    查询指定股票代码的当前实时价格。
    参数 ticker 应该是股票的缩写代码，例如：苹果是 AAPL，微软是 MSFT，英伟达是 NVDA。
    """
    try:
        stock = yf.Ticker(ticker)
        # 获取最新收盘价或当前价格
        price = stock.fast_info['last_price']
        currency = stock.fast_info['currency']
        return f"{ticker} 的当前实时股价为 {price:.2f} {currency}。"
    except Exception as e:
        return f"查询股价时出错: {e}"

@tool
def financial_calculator(expression: str) -> str:
    """
    一个简单的金融计算器，用于处理加减乘除。输入应该是数学表达式，例如 '391035 * 0.05'。
    """
    try:
        # 注意：生产环境建议用更安全的计算方式
        result = eval(expression)
        return f"计算结果为: {result}"
    except:
        return "无法计算，请确保输入的是数学表达式。"