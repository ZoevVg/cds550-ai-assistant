import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 0. 页面设置
# ==========================================
st.set_page_config(page_title="CDS550 AI Workstation", page_icon="🎓", layout="wide")
st.title("🎓 CDS550 复合模型协作工作台")
st.markdown("**思考 → 生成 → 批判 → 修订**：展示完整的 AI 思考过程，严格基于 PDF 内容。")

ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# ==========================================
# 1. 侧边栏：让访客输入自己的 API Key
# ==========================================
with st.sidebar:
    st.header("🔑 API Key 配置")
    st.caption("为了保护隐私和避免额度耗尽，请输入你自己的 API Key。")
    
    # 使用 type="password" 隐藏输入内容
    user_zhipu_key = st.text_input("智谱 API Key (必填)", type="password")
    user_deepseek_key = st.text_input("DeepSeek API Key (必填)", type="password")
    
    # 如果用户没有填写 Key，则暂停后续操作
    if not user_zhipu_key or not user_deepseek_key:
        st.warning("⚠️ 请先在左侧填写你的 API Key 才能使用本应用。")
        st.info("💡 如果你没有 Key，可以前往智谱AI (open.bigmodel.cn) 或 DeepSeek (platform.deepseek.com) 免费注册获取。")
        st.stop()  # 停止执行，不渲染后面的界面

    st.success("✅ API Key 已配置")
    st.markdown("---")
    st.header("📁 上传课件")
    uploaded_file = st.file_uploader("请上传 Lecture PDF", type="pdf")
    st.info("上传后，系统会自动解析并建立知识库。")


if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 2. 核心逻辑：PDF 处理与模型初始化
# ==========================================
if uploaded_file is not None:
    # 使用 tempfile 保证云端和本地兼容的临时文件写入
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.getbuffer())
        temp_pdf_path = temp_pdf.name

    @st.cache_resource(show_spinner="正在解析 PDF 并建立知识库...")
    def process_pdf(file_path, zhipu_key):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        
        # 关键修改：将访客的 Key 显式传递给 Embedding
        embeddings = OpenAIEmbeddings(
            model="embedding-2", 
            openai_api_base=ZHIPU_BASE_URL,
            api_key=zhipu_key 
        )
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore

    # 将用户填写的 Key 传入处理函数
    vectorstore = process_pdf(temp_pdf_path, user_zhipu_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 关键修改：使用访客填写的 Key 初始化模型
    model_zhipu = ChatOpenAI(
        model="glm-4-flash", 
        temperature=0, 
        openai_api_base=ZHIPU_BASE_URL,
        api_key=user_zhipu_key
    )
    
    model_deepseek = ChatOpenAI(
        model="deepseek-chat", 
        api_key=user_deepseek_key, 
        base_url=DEEPSEEK_BASE_URL, 
        temperature=0.1
    )

    tab1, tab2, tab3 = st.tabs(["💬 复合模型问答", "📝 随堂测验生成", "📄 课件一键总结"])

    # ==========================================
    # Tab 1: 复合模型问答（展示思考过程）
    # ==========================================
    with tab1:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("输入问题（例如：What are the three gates in an LSTM?）"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # 阶段 1：生成初稿
                with st.spinner("第 1 步：智谱正在思考并起草初稿..."):
                    docs = retriever.invoke(prompt)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    draft_prompt = ChatPromptTemplate.from_template("""
                    You are a helpful teaching assistant. Use ONLY the following context to answer the question.
                    If you don't know the answer, say "I cannot find the answer in the provided lecture notes."
                    Answer in English.
                    
                    Context: {context}
                    Question: {question}
                    
                    Please format your response exactly like this:
                    [Thinking Process]
                    (Write your brief reasoning process here)
                    
                    [Draft Answer]
                    (Write your actual answer here)
                    """)
                    draft_chain = draft_prompt | model_zhipu | StrOutputParser()
                    draft_result = draft_chain.invoke({"context": context, "question": prompt})
                
                # 阶段 2：审查与批判
                with st.spinner("第 2 步：DeepSeek 正在审查初稿是否有幻觉或遗漏..."):
                    critique_prompt = ChatPromptTemplate.from_template("""
                    You are a strict reviewer. Compare the DRAFT with the CONTEXT.
                    Identify any hallucinations (info not in context) or missing key concepts.
                    If the draft is perfect, state "NO ISSUES FOUND".
                    Output in English.
                    
                    Context: {context}
                    Question: {question}
                    Draft: {draft}
                    
                    Please format your response exactly like this:
                    [Reviewer's Thinking Process]
                    (Write your brief reasoning process here)
                    
                    [Critique]
                    (Write your critique or NO ISSUES FOUND here)
                    """)
                    critique_chain = critique_prompt | model_deepseek | StrOutputParser()
                    critique_result = critique_chain.invoke({"context": context, "question": prompt, "draft": draft_result})
                
                # 阶段 3：修订与终审
                with st.spinner("第 3 步：智谱正在根据审查意见进行修订..."):
                    revise_prompt = ChatPromptTemplate.from_template("""
                    You are the final editor. Based on the CRITIQUE, revise the DRAFT.
                    Ensure the final answer is strictly based on the Context, accurate, and in English.
                    
                    Context: {context}
                    Question: {question}
                    Draft: {draft}
                    Critique: {critique}
                    
                    Final Revised Answer:
                    """)
                    revise_chain = revise_prompt | model_zhipu | StrOutputParser()
                    final_answer = revise_chain.invoke({
                        "context": context, 
                        "question": prompt, 
                        "draft": draft_result, 
                        "critique": critique_result
                    })
                    
                    # 前端展示
                    st.markdown("### ✅ 最终修订答案 (Final Revised Answer)")
                    st.success(final_answer)
                    
                    st.write("---")
                    st.markdown("### 🧠 复合模型内部思考与协作全过程")
                    with st.expander("📝 阶段 1：智谱 GLM-4-Flash (起草与思考)", expanded=True):
                        st.info(draft_result)
                    with st.expander("🧐 阶段 2：DeepSeek (审查与思考)", expanded=True):
                        st.warning(critique_result)
                    with st.expander("📖 查看参考出处 (Proof)"):
                        for i, doc in enumerate(docs):
                            page_num = doc.metadata.get('page', 0) + 1
                            st.markdown(f"**Source {i+1}** (Page {page_num}): {doc.page_content[:150]}...")
            
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

    # ==========================================
    # Tab 2: 随堂测验生成
    # ==========================================
    with tab2:
        st.subheader("📝 Generate Quiz from Lecture")
        st.write("Strictly generate multiple-choice questions based on the lecture content.")
        if st.button("✨ Generate Quiz"):
            with st.spinner("AI is generating questions..."):
                docs = retriever.invoke("core concepts definitions")
                context = "\n".join([doc.page_content for doc in docs])
                quiz_prompt = ChatPromptTemplate.from_template("""
                You are a professor. Based STRICTLY on the following lecture content, generate 3 multiple-choice questions. 
                Do not use any outside knowledge. Output in English.
                
                Lecture Content: 
                {context}
                """)
                quiz_chain = quiz_prompt | model_zhipu | StrOutputParser()
                quiz = quiz_chain.invoke({"context": context})
                st.success(quiz)

    # ==========================================
    # Tab 3: 课件一键总结
    # ==========================================
    with tab3:
        st.subheader("📄 Summarize Lecture")
        if st.button("🚀 Summarize"):
            with st.spinner("AI is summarizing..."):
                docs = retriever.invoke("key concepts summary")
                context = "\n".join([doc.page_content for doc in docs])
                summary_prompt = ChatPromptTemplate.from_template("""
                Summarize the core concepts of the following lecture content. 
                Provide a brief summary and list 3 key concepts in bullet points. 
                You MUST answer STRICTLY based on the provided context and output in English.
                
                Lecture Content:
                {context}
                """)
                summary_chain = summary_prompt | model_zhipu | StrOutputParser()
                summary = summary_chain.invoke({"context": context})
                st.success(summary)
