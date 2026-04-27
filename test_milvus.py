from pymilvus import connections, db

# 1. 连接到 Milvus
# 注意：因为 Python 是运行在 Windows 主机上的，
# 所以这里要填 localhost
print("正在连接 Milvus...")
try:
    connections.connect("default", host="localhost", port="19530")
    print("✅ 连接成功！你的向量数据库已准备就绪。")
    
    # 2. 查看当前的数据库列表
    databases = db.list_database()
    print(f"当前数据库列表: {databases}")
    
except Exception as e:
    print(f"❌ 连接失败: {e}")