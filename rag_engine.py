import glob
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

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_FOLDER = "data/faiss_index"


def get_embeddings():
    """Embeddings 100% locais (gratuitos e rodando na CPU)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def _latest_pdf_mtime(data_folder):
    """Data de modificação do PDF mais recente (0.0 se não houver PDFs)."""
    pdfs = glob.glob(os.path.join(data_folder, "*.pdf"))
    return max((os.path.getmtime(p) for p in pdfs), default=0.0)


def build_vector_store(
    data_folder="data/scholarships",
    index_folder=DEFAULT_INDEX_FOLDER,
    force_rebuild=False,
):
    """
    Lê todos os PDFs da pasta de editais de bolsas, fatias os textos em chunks
    e gera um banco vetorial local utilizando FAISS e Embeddings gratuitos da HuggingFace.

    O índice é persistido em disco (``index_folder``) e reutilizado nas chamadas
    seguintes. Só é reconstruído quando um PDF é adicionado/modificado ou quando
    ``force_rebuild=True``.
    """
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        raise FileNotFoundError(f"A pasta '{data_folder}' foi criada, mas está vazia. Adicione PDFs de editais lá.")

    embeddings = get_embeddings()
    index_file = os.path.join(index_folder, "index.faiss")

    # Reutiliza o índice existente se ele estiver atualizado em relação aos PDFs.
    if (
        not force_rebuild
        and os.path.exists(index_file)
        and _latest_pdf_mtime(data_folder) <= os.path.getmtime(index_file)
    ):
        print("✅ Índice FAISS carregado do disco (sem re-gerar embeddings).")
        return FAISS.load_local(
            index_folder, embeddings, allow_dangerous_deserialization=True
        )

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

    # 3. Criando o banco de dados vetorial FAISS
    print("⏳ Gerando embeddings locais com HuggingFace (all-MiniLM-L6-v2)...")
    vector_store = FAISS.from_documents(docs_chunks, embeddings)

    # 4. Persistindo o índice para reutilização
    os.makedirs(index_folder, exist_ok=True)
    vector_store.save_local(index_folder)
    print(f"✅ Banco de dados vetorial FAISS criado e guardado em '{index_folder}'.")

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
