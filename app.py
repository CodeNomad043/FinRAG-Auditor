import streamlit as st
from production_audit import FinancialAuditor
import time
import os
from pathlib import Path

# 设置页面配置
st.set_page_config(page_title="AI 智能财报审计员", page_icon="⚖️", layout="wide")

# --- 1. 样式与初始化 ---
st.title("⚖️ AI 智能多文档审计系统")
st.markdown("---")

@st.cache_resource
def get_auditor():
    return FinancialAuditor()

with st.spinner("正在初始化审计引擎..."):
    auditor = get_auditor()

# 临时文件存放目录
UPLOAD_DIR = Path("/root/autodl-tmp/temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# --- 2. 侧边栏：文件上传与状态 ---
with st.sidebar:
    st.header("📂 审计文档管理")
    
    # 文件上传组件
    uploaded_file = st.file_uploader("上传 PDF 财报文档", type=["pdf"])
    
    if uploaded_file:
        file_path = UPLOAD_DIR / uploaded_file.name
        # 将上传的文件保存到本地
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"文件已存至服务器: {uploaded_file.name}")
        
        # 核心功能：点击后重新构建 RAG 索引
        if st.button("🔨 为新文档构建索引", use_container_width=True):
            with st.spinner("正在解析文档并重建向量空间..."):
                auditor._prepare_index(str(file_path))
                st.session_state['current_doc'] = uploaded_file.name
                st.success("索引重建完成！")
                
    st.divider()
    st.header("🏢 系统状态面板")
    st.success("✅ 模型: Qwen-Audit")
    st.info("✅ 监控: Langfuse 接入")
    
    if 'current_doc' in st.session_state:
        st.write(f"当前活动文档: `{st.session_state['current_doc']}`")
    
    if st.button("🧹 清除所有缓存"):
        if 'audit_result' in st.session_state:
            del st.session_state['audit_result']
        st.rerun()

# --- 3. 主界面：审计交互 ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 发起审计指令")
    # 动态调整默认问题
    default_q = "请提取文档中 2024 财年的净销售额和研发费用。"
    user_query = st.text_area("请输入您的审计需求：", 
                             placeholder=default_q,
                             height=150)
    
    if st.button("🚀 开始深度审计", use_container_width=True):
        if not auditor.query_engine:
            st.error("❌ 请先在左侧上传文档并点击‘构建索引’！")
        elif user_query:
            with st.spinner("正在分析中..."):
                start_time = time.time()
                response_obj = auditor.audit_task(user_query)
                end_time = time.time()
                
                st.session_state['audit_result'] = response_obj
                st.session_state['process_time'] = end_time - start_time
        else:
            st.warning("⚠️ 请输入指令。")

with col2:
    st.subheader("🔍 审计报告输出")
    if 'audit_result' in st.session_state:
        res = st.session_state['audit_result']
        duration = st.session_state['process_time']
        
        st.success(f"任务执行成功！耗时: {duration:.2f}s")
        st.markdown("### 📝 审计结果")
        
        # 结果文本提取
        if hasattr(res, 'response'):
            st.info(res.response)
        else:
            st.write(str(res))

        # 数据溯源
        with st.expander("📌 查看数据来源 (Source Nodes)"):
            if hasattr(res, 'source_nodes'):
                for i, node in enumerate(res.source_nodes):
                    score = getattr(node, 'score', 0)
                    page = node.node.metadata.get('page_label', '未知')
                    st.markdown(f"**片段 {i+1}** (相关度: {score:.2f} | 第 {page} 页)")
                    st.caption(node.node.get_content())
                    st.divider()
    else:
        st.info("💡 请上传文档并输入指令。")

st.markdown("---")
st.caption("基于 RAG 架构与 Qwen2.5-7B 微调模型 | 审计数据受 Langfuse 实时监控")