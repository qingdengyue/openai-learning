import gradio as gr
import os
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter




## 生成一个List 包含 婴幼儿教育,电器,书籍,家装 四个内容 每个内容 包含 category 和 file_path 两个key,category是内容本身,file_path是文件路径
category_list_with_file_path = [
    {"key":1,"category": "婴幼儿教育", "file_path": "real_baby_edu_data.txt","db":{},"context":{}},
    {"key":2,"category": "电器", "file_path": "real_dianqi_data.txt","db":{},"context":{}},
    {"key":3,"category": "书籍", "file_path": "real_book_data.txt","db":{},"context":{}},
    {"key":4,"category": "家装", "file_path": "real_home_data.txt","db":{},"context":{}}
]

text_splitter = CharacterTextSplitter(        
    separator = r'\d+\.',
    chunk_size = 120,
    chunk_overlap  = 0,
    length_function = len,
    is_separator_regex = True,
)


def init_knowledge_db():
    ## 判断每一个  category_list_with_file_path中 file_path 在当前路径存在,如果存在则使用FAISS加载,如果不存在则初始化数据
    for category_dict in category_list_with_file_path:
        file_path = category_dict["file_path"]
        category = category_dict["category"]
        db_object = category_dict["db"]
        ## file_path 去除txt后缀
        file_path_dir = file_path.replace(".txt", "")
        if os.path.exists(file_path_dir):
            db = FAISS.load_local(file_path_dir, OpenAIEmbeddings(),allow_dangerous_deserialization=True)
            category_dict["db"]=db
        else:
            with open(file_path) as f:
                file_content = f.read()
                docs = text_splitter.create_documents([file_content])
                print(f"已加载:{len(docs)}条数据")
                db = FAISS.from_documents(docs, OpenAIEmbeddings())
                db.save_local(file_path_dir)
                print(f"{category}:初始化数据成功")
                category_dict["db"]=db





## 初始化各个领域的对象
def initialize_sales_bot():
    for category_dict in category_list_with_file_path:
        db = category_dict["db"]
        category = category_dict["category"]
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,verbose=True)
        context=RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
        context.return_source_documents = True
        category_dict["context"]=context
    



def sales_chat_1(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    for category_dict in category_list_with_file_path:
        key = category_dict["key"]
        if key==1:
            SALES_BOT = category_dict["context"]
    
    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if len(ans["source_documents"])>0 or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"
    

def sales_chat_2(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    for category_dict in category_list_with_file_path:
        key = category_dict["key"]
        if key==2:
            SALES_BOT = category_dict["context"]
    
    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if len(ans["source_documents"])>0 or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"
    


def sales_chat_3(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    for category_dict in category_list_with_file_path:
        key = category_dict["key"]
        if key==3:
            SALES_BOT = category_dict["context"]
    
    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if len(ans["source_documents"])>0 or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"
    

def sales_chat_4(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    for category_dict in category_list_with_file_path:
        key = category_dict["key"]
        if key==4:
            SALES_BOT = category_dict["context"]
    
    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if len(ans["source_documents"])>0 or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"
    

def launch_gradio():



    ## 初始tabs
    tabs=[]
    tabs_title=[]


    ## 生成每一个tab
    for category_dict in category_list_with_file_path:
        category = category_dict["category"]
        key = category_dict["key"]
        if key ==1 :
            sales_chat = sales_chat_1
        elif key ==2 :
            sales_chat = sales_chat_2
        elif key ==3 :
            sales_chat = sales_chat_3
        elif key ==4 :
            sales_chat = sales_chat_4

        tab = gr.ChatInterface(
            fn=sales_chat,
            title=category,
            # retry_btn=None,
            # undo_btn=None,
            chatbot=gr.Chatbot(height=600),
        )
        tabs.append(tab)
        tabs_title.append(category)

    demo=gr.TabbedInterface(tabs,tabs_title)

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":

    # 初始化数据库
    init_knowledge_db()

    # 初始化房产销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()