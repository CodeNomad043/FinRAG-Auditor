import json
from production_audit import FinancialAuditor  # 引用你之前的逻辑
from langfuse import Langfuse

def run_batch_test():
    auditor = FinancialAuditor()
    langfuse = Langfuse()
    
    with open("data/test_cases.json", "r", encoding="utf-8") as f:
        tests = json.load(f)

    for case in tests:
        print(f"正在测试: {case['query']}")
        # 这里的 trace 会自动关联到你在 production_audit.py 中配置的 Langfuse 逻辑
        response = auditor.audit_task(case['query'])
        print(f"响应结果: {response}\n")

if __name__ == "__main__":
    run_batch_test()