import os
from dotenv import load_dotenv
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# 加载环境变量
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

# 导入通义千问相关的包
try:
    from langchain_community.embeddings import DashScopeEmbeddings
    from langchain_community.chat_models import ChatTongyi
except ImportError:
    # 如果导入失败，尝试安装
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "dashscope"])
    from langchain_community.embeddings import DashScopeEmbeddings
    from langchain_community.chat_models import ChatTongyi

DB_DIR = 'faiss_db/'

# 初始化通义千问嵌入模型
EMBEDDING_MODEL = DashScopeEmbeddings(
    model="text-embedding-v4",  # 通义千问的嵌入模型
    dashscope_api_key=os.getenv('DASHSCOPE_API_KEY')  # 从环境变量读取API key
)


def save_vectors_db():
    """构建向量数据库，并保存到磁盘"""
    if os.path.exists(DB_DIR):
        print('向量数据库已经构建，直接读取就ok!')
    else:
        with open('sales_datas.txt', encoding='utf-8') as f:
            contents = f.read()
        # 把文本内容切割成一个个的doc
        text_splitter = CharacterTextSplitter(
            separator=r'\d+\.',
            is_separator_regex=True,
            chunk_size=100,
            chunk_overlap=0,
            length_function=len
        )
        docs = text_splitter.create_documents([contents])
        print(f"文档分割数量: {len(docs)}")

        # 使用通义千问嵌入模型构建向量数据库
        db = FAISS.from_documents(docs, EMBEDDING_MODEL)
        db.save_local(DB_DIR)
        print('向量数据库构建完成并已保存!')


def init_chain():
    """最终得到的一个chain"""
    # 第一步:加载向量数据库
    db = FAISS.load_local(DB_DIR, EMBEDDING_MODEL, allow_dangerous_deserialization=True)

    # 第二步:创建一个提示模版
    system_prompt = """你是一位专业的AI汽车销售顾问，同时也是公司的销售冠军。
    你融合了顶尖销售的经验、心理学知识和客户对汽车需求的深刻理解。你的名字是“小智汽车顾问”。
    你的核心价值不是机械地回答，而是通过对话，与客户建立信任，深度挖掘他们的真实需求，并引导他们完成从购车咨询到车辆购买的整个流程。
    使用以下检索到的上下文片段来回答问题。如果你不知道答案，就说:"这个问题,我建议你直接问人工客服!"。\n
    {context}
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ('system', system_prompt),
            ('human', '{input}')
        ]
    )

    # 第三步：创建一个chain
    # 修改检索器：不使用分数阈值，而是使用相似度搜索
    retriever = db.as_retriever(
        search_type="similarity",  # 使用相似度搜索
        search_kwargs={'k': 3}  # 返回前3个最相关的结果
    )

    # 使用通义千问的聊天模型
    llm = ChatTongyi(
        model="qwen3-max",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        temperature=0.2
    )

    # 将检索到的结果（多个docs）输入到提示模版中
    chain1 = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
    return create_retrieval_chain(retriever=retriever, combine_docs_chain=chain1)


if __name__ == '__main__':
    save_vectors_db()
    chain = init_chain()
