import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
# Importações novas e mais seguras:
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configurações do Ollama
LLM_MODEL = "phi3" 
EMBEDDING_MODEL = "nomic-embed-text"

st.set_page_config(page_title="Chatbot PI-V", page_icon="🤖")
st.title("🤖 Chatbot RAG Local - PI-V")
st.caption(f"Rodando 100% offline com Ollama ({LLM_MODEL})")

DOC_DIR = "documentos"

@st.cache_resource
def load_and_process_document():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # 1. O "Pulo do Gato": Tenta carregar o banco se ele já existir (Rápido!)
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # 2. Se o banco não existir, vai ler os PDFs (Demorado, mas só acontece 1 vez)
    if not os.path.exists(DOC_DIR) or not os.listdir(DOC_DIR):
        return None
    
    with st.spinner('Vetorizando documentos (Isso só vai demorar desta vez!)...'):
        loader = PyPDFDirectoryLoader(DOC_DIR)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        splits = text_splitter.split_documents(docs)
        
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        # Salva o banco processado em uma pasta local
        vectorstore.save_local("faiss_index")
        
        return vectorstore

vectorstore = load_and_process_document()

if vectorstore is None:
    st.error(f"Arquivo '{DOC_DIR}' não encontrado. Coloque os PDFs na pasta 'documentos'.")
else:
    llm = Ollama(model=LLM_MODEL, temperature=0.2)
    retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 6, "fetch_k": 20}
)
    
    system_prompt = (
    "Você é um analista de dados especialista na documentação do projeto PI-V. "
    "Responda SEMPRE em Português do Brasil. "
    "Sua única fonte de verdade é o contexto fornecido abaixo. Analise-o cuidadosamente. "
    "Se a resposta não estiver explicitamente contida no contexto fornecido, "
    "diga EXATAMENTE a seguinte frase: 'Não encontrei informações suficientes no documento para responder a essa pergunta.' "
    "Não invente leis, não deduza informações e não utilize conhecimentos externos.\n\n"
    "Contexto recuperado:\n{context}"
)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Nova estrutura RAG que ignora o langchain.chains e usa LCEL puro
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Interface de Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Faça uma pergunta sobre a documentação..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                # Invoca a resposta diretamente da nova chain
                answer = rag_chain.invoke(user_input)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
