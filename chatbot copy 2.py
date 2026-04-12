import streamlit as st
#from langchain_ollama.llms import OllamaLLM
#from langchain_ollama import ChatOllama
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.retrieval_qa.base import RetrievalQA
#from langchain_classic.chains.retrieval_qa.base import RetrievalQA
#from langchain_classic.memory import ConversationBufferMemory
#from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
# from streamlit_image_select import image_select
# import time
# from langchain.retrievers import TFIDFRetriever

GOOGLE_API_KEY = "AIzaSyDumWBWL17xjaf6YUhksEVzc2AeY5yE3dI" #Pass your key here

images = ["human.png",
          "https://cdn-icons-png.flaticon.com/512/4712/4712027.png",
          "https://thumbs.dreamstime.com/b/blue-smiley-face-button-15880828.jpg"]




st.header("Demo ChatBot")



with  st.sidebar:
    st.title("Powered By Gemini")
    #file = st.file_uploader(" Upload a PDf file and start asking questions", type="pdf")

#Extract the text


# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",streaming=True ,google_api_key=GOOGLE_API_KEY, temperature=0.2)




template = """
You are a retrieval-based AI assistant. You must answer ONLY using the provided context.

======================
🚨 STRICT RULES (MANDATORY)
======================
1. You are ONLY allowed to use the information present in the CONTEXT section.
2. You MUST NOT use prior knowledge, memory, or assumptions.
3. If the answer is not explicitly present in the context, you MUST reply EXACTLY with:
   "I'm sorry, but that question is outside the scope of the provided information."
4. Do NOT attempt to guess or infer.
5. If context is empty or irrelevant, treat it as no answer found.

======================
📚 CONTEXT
======================
{context}

======================
💬 CHAT HISTORY
======================
{history}

======================
❓ USER QUESTION
======================
{question}

======================
🤖 RESPONSE
======================
"""


# prompt = PromptTemplate(
#     input_variables=["history", "context", "question"],
#     template=template,
# )


contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given chat history and latest user question, reformulate it into a standalone question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a helpful assistant. Use ONLY the provided context to answer.\n"
     "You are allowed to greet or greet back to humans. Be polite.\n"
     "Make analysis if asked to do so and be very polite.\n"
     "If answer is not in context, say: 'This is outside my knowledge base.'\n\n"
     "Context:\n{context}"
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])



    # generating embedding
#embedding_model = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
# loaded_vectors=FAISS.load_local("vectors_pdf", embedding_model,allow_dangerous_deserialization=True)
# faiss_retriever = loaded_vectors.as_retriever(search_type="mmr",search_kwargs={"k":5,"fetch_k":339})

# llm2  = Ollama(model="llama2")
# llm2  = Ollama(model="llama2")


if "llm" not in st.session_state:
    # llm = ChatOllama(
    # model="phi3",
    # base_url="http://localhost:11434",
    # temperature=0.2,
    # )

    #llm = OllamaLLM(model="llama2",base_url="http://localhost:11435", temperature=0.2)

    # print("LLM:",llm)
    # response = llm.invoke("Say hello in one sentence")
    # print("Response:",response)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",streaming=True , convert_system_message_to_human=True, google_api_key=GOOGLE_API_KEY, temperature=0.2)
    # print("LLM2:",llm2)
    st.session_state.llm = llm

else:
    llm = st.session_state.llm


if "embeddingModel" not in st.session_state:
    embedding_model = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    # embedding_model = HuggingFaceEmbeddings(
    # model_name="sentence-transformers/all-mpnet-base-v2",
    # model_kwargs={"device": "cpu"}
    # )
    st.session_state.embeddingModel = embedding_model

else:
    embedding_model = st.session_state.embeddingModel


if "loadedVectors" not in st.session_state:
    loaded_vectors=FAISS.load_local("vectors_pdf", embedding_model,allow_dangerous_deserialization=True)
    st.session_state.loadedVectors = loaded_vectors

else:
    loaded_vectors = st.session_state.loadedVectors


if "retriever" not in st.session_state:
    faiss_retriever = loaded_vectors.as_retriever(search_type="mmr",search_kwargs={"k":2})  #k=5
    st.session_state.retriever = faiss_retriever

else:
    faiss_retriever = st.session_state.retriever

        #output results
        #chain -> take the question, get relevant document, pass it to the LLM, generate the output


if "historyRetriever" not in st.session_state:
    history_aware_retriever = create_history_aware_retriever(
    llm,
    faiss_retriever,
    contextualize_q_prompt
    )
    st.session_state.historyRetriever = history_aware_retriever
else:
    history_aware_retriever = st.session_state.historyRetriever


if "QAChain" not in st.session_state:
    question_answer_chain = create_stuff_documents_chain(
    llm,
    qa_prompt
    )
    st.session_state.QAChain = question_answer_chain

else:
    question_answer_chain = st.session_state.QAChain

if "RAGChain" not in st.session_state:
    rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain
    )
    st.session_state.RAGChain = rag_chain

else:
    rag_chain = st.session_state.RAGChain



if "chatHistory" not in st.session_state:
    chat_history = []

else:
    chat_history = st.session_state.chatHistory


# if "model" not in st.session_state:
#     print("\n***Model not initiated***\n")
#     qa_with_sources_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever = faiss_retriever,
#         return_source_documents=False,
#         chain_type_kwargs={
#         "verbose": True,
#         "prompt": prompt,
#         "memory": ConversationBufferMemory(
#             memory_key="history",
#             input_key="question")
#     }
#     )
#     st.session_state.model = qa_with_sources_chain

# else:
#     print('\n***Model exists***\n')
#     qa_with_sources_chain = st.session_state.model

       
if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"],avatar=message['avatar']):
        st.markdown(message["content"])


#memory = ConversationBufferMemory(memory_key='query',return_messages=True,output_key='result')    




    #embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # creating vector store - FAISS
  

    # get user question

# with st.chat_message("user"):
#     st.markdown(user_question)



    # do similarity search
    
        #st.write(match)

        #define the LLM
       
#model = genai.GenerativeModel("gemini-2.5-flash")
#genai.configure(api_key="AIzaSyAeUbPohT7c-e41g6t-CvmAPnD7ljUsDxo")



#print("\n***********Sourav*********\n")
user_question = st.chat_input("Type Your question here")

if user_question:
    
    st.session_state.messages.append({"role": "user", "content": user_question,"avatar":images[0]})
    with st.chat_message("user",avatar=images[0]):
        st.markdown(user_question)      
    
    
   
    #searchDocs = loaded_vectors.similarity_search(user_question,fetch_k=339)
    #chain = load_qa_chain(llm, chain_type="stuff")
    
   # print(response)
    #print(qa_with_sources_chain.combine_documents_chain.memory)
    with st.chat_message("assistant", avatar=images[1]):
    #     stream = llm.chat.completions.create(
    #         model=st.session_state["messages"],
    #         messages=[
    #             {"role": m["role"], "content": m["content"]}
    #             for m in st.session_state.messages
    #         ],
    #         stream=True,
    #     )
    #     response = st.write_stream(stream)
        #response_box = st.empty()
        

        response = rag_chain.invoke({
        "input": user_question,
        "chat_history": chat_history
        })

        print(response["answer"])

        #response = qa_with_sources_chain.run({"query":user_question})
        
        #st.write_stream(response)
        st.markdown(response['answer'])
        chat_history.append(("human", user_question))
        chat_history.append(("ai", response['answer']))
    
    st.session_state.chatHistory = chat_history
    st.session_state.messages.append({"role": "assistant", "content": response['answer'],"avatar":images[1]})
    # st.session_state.model = qa_with_sources_]chain
    #response = chain.run(input_documents = searchDocs, question = user_question)
    #st.write(response['result'])
