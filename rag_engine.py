import os
from dotenv import load_dotenv

# Carrega a chave GROQ_API_KEY do arquivo .env
load_dotenv()

from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

def build_vector_store(data_folder="data/scholarships"):
    """
    Lê todos os PDFs da pasta de editais de bolsas, fatias os textos em chunks
    e gera um banco vetorial local utilizando FAISS e Embeddings gratuitos da HuggingFace.
    """
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        raise FileNotFoundError(f"A pasta '{data_folder}' foi criada, mas está vazia. Adicione PDFs de editais lá.")

    # 1. Carrega todos os PDFs presentes na pasta data/scholarships
    loader = PyPDFDirectoryLoader(data_folder)
    documents = loader.load()

    if not documents:
        raise ValueError(f"Nenhum arquivo PDF foi encontrado dentro da pasta '{data_folder}'.")

    print(f"📖 {len(documents)} páginas de editais carregadas com sucesso.")

    # 2. Divide os documentos em pedaços menores (chunks) com sobreposição para manter contexto
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    docs_chunks = text_splitter.split_documents(documents)
    print(f"✂️ Documentos fatiados em {len(docs_chunks)} fragmentos (chunks).")

    # 3. Inicializa os embeddings 100% locais (gratuitos e rodando direto na sua CPU)
    print("⏳ Gerando embeddings locais com HuggingFace (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # 4. Criando e armazenando no banco de dados vetorial FAISS
    vector_store = FAISS.from_documents(docs_chunks, embeddings)
    print("✅ Banco de dados vetorial FAISS criado com sucesso!")

    return vector_store

def create_scholarship_rag_chain(vector_store):
    """
    Conecta o retriever do banco vetorial ao modelo LLM do Groq Cloud.
    """
    # 1. Configura o modelo Groq com LLaMA 3.3 70B
    llm = ChatGroq(
        model_name="openai/gpt-oss-120b",
        temperature=0.2, # Temperatura baixa para respostas mais fidedignas ao edital
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # 2. Transforma o banco vetorial em um buscador (retriever) dos 4 trechos mais relevantes
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # 3. Monta a cadeia RAG de perguntas e respostas
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain


# Block para testes locais via terminal
if __name__ == "__main__":
    try:
        print("🚀 Iniciando a RAG Engine...")
        db = build_vector_store("data/scholarships")
        rag_chain = create_scholarship_rag_chain(db)

        # Teste de pergunta
        pergunta = "Quais são os requisitos gerais para se candidatar às bolsas de estudo?"
        print(f"\n❓ Pergunta de teste: {pergunta}")

        resposta = rag_chain.invoke({"query": pergunta})

        print("\n💡 Resposta da IA (Groq + RAG):")
        print(resposta["result"])

        print("\n📄 Fontes Consultadas nos PDFs:")
        for doc in resposta["source_documents"]:
            fonte = doc.metadata.get("source", "PDF")
            pagina = doc.metadata.get("page", 0) + 1
            print(f"- Arquivo: {os.path.basename(fonte)} | Página: {pagina}")

    except Exception as e:
        print(f"\n⚠️ Erro durante o teste: {e}")
