import streamlit as st
from production_audit import FinancialAuditor
import time

# 设置页面配置
st.set_page_config(page_title="AI 智能财报审计员", page_icon="⚖️", layout="wide")

st.title("⚖️ AI 智能财报审计系统")
st.markdown("---")

# 初始化后端引擎（使用 st.cache_resource 避免页面刷新时重新加载模型）
@st.cache_resource
def get_auditor():
    return FinancialAuditor()

with st.spinner("正在初始化审计引擎并加载模型，请稍候..."):
    auditor = get_auditor()

# 侧边栏：状态监控
with st.sidebar:
    st.header("🏢 系统状态面板")
    st.success("✅ 模型已就绪 (Qwen-Audit)")
    st.info("✅ 监控已接入 (Langfuse)")
    st.divider()
    st.write("**当前审计文档：**")
    st.code("apple_2025_10k.pdf")
    
    if st.button("清除对话缓存"):
        if 'audit_result' in st.session_state:
            del st.session_state['audit_result']
        st.rerun()

# 主界面：左右布局
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 发起审计指令")
    user_query = st.text_area("请输入您的审计需求：", 
                             placeholder="例如：提取文档中 2024 财年的净销售额和研发费用。",
                             height=150)
    
    if st.button("🚀 开始执行深度审计"):
        if user_query:
            with st.spinner("正在检索文档并解析财务指标..."):
                start_time = time.time()
                # 调用后端 FinancialAuditor 执行 RAG
                response_obj = auditor.audit_task(user_query)
                end_time = time.time()
                
                # 存储结果对象和耗时
                st.session_state['audit_result'] = response_obj
                st.session_state['process_time'] = end_time - start_time
        else:
            st.warning("⚠️ 请输入具体的审计指令。")

with col2:
    st.subheader("🔍 审计报告输出")
    if 'audit_result' in st.session_state:
        res = st.session_state['audit_result']
        duration = st.session_state['process_time']
        
        st.success(f"审计任务执行成功！耗时: {duration:.2f}s")
        
        # --- [重点修改：结果清洗] ---
        st.markdown("### 📝 审计意见")
        # 提取 Response 对象中的文本内容，避免显示 NodeWithScore 等原始数据
        if hasattr(res, 'response'):
            st.info(res.response)
        else:
            st.write(str(res))

        # --- [重点修改：数据溯源] ---
        with st.expander("📌 查看数据来源 (Source Nodes)"):
            st.caption("以下是模型给出上述回答所参考的原始文档片段：")
            if hasattr(res, 'source_nodes'):
                for i, node in enumerate(res.source_nodes):
                    score = getattr(node, 'score', 0)
                    page = node.node.metadata.get('page_label', '未知')
                    
                    st.markdown(f"**来源片段 {i+1}** (相关度: {score:.2f} | 第 {page} 页)")
                    st.text_area(f"Node ID: {node.node.node_id}", 
                                 value=node.node.get_content(), 
                                 height=100, 
                                 key=f"node_{i}")
                    st.divider()
    else:
        st.info("💡 请在左侧输入指令，点击按钮获取审计结果。")

st.markdown("---")
st.caption("基于 RAG 架构与 Qwen2.5-7B 微调模型构建 | 审计数据受 Langfuse 实时监控")