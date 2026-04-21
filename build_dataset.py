import json

# 这里模拟从财报中提取的 5 组核心指标，实际操作中你可以根据阶段一的线索多写点
raw_facts = [
    {"q": "2025年iPhone收入", "i": "iPhone net sales $209,586 million, up 4%", "o": '{"product":"iPhone","rev":209586,"change":"+4%"}'},
    {"q": "2025年iPad表现", "i": "iPad net sales $28,023 million, up 5%", "o": '{"product":"iPad","rev":28023,"change":"+5%"}'},
    {"q": "2025年Mac表现", "i": "Mac net sales $33,708 million, up 12%", "o": '{"product":"Mac","rev":33708,"change":"+12%"}'},
    {"q": "2025年服务收入", "i": "Services net sales $109,158 million, up 14%", "o": '{"product":"Services","rev":109158,"change":"+14%"}'},
    {"q": "2025年净利润", "i": "Net income $112,010 million in 2025", "o": '{"metric":"Net Income","val":112010,"unit":"M"}'}
]

dataset = []
for item in raw_facts:
    dataset.append({
        "instruction": "作为审计助手，请从文本中提取财务指标并以JSON格式输出。",
        "input": item["i"],
        "output": item["o"]
    })

with open("audit_train_data.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"✅ 成功生成 {len(dataset)} 条训练数据！")