import glob
import os

import streamlit as st
from rag_engine import build_vector_store, create_scholarship_rag_chain
from database import init_db, get_all_scholarships, add_scholarship


def _folder_signature(folder):
    """Assinatura dos PDFs (caminho + mtime) para invalidar o cache do Streamlit."""
    return tuple(
        sorted((f, os.path.getmtime(f)) for f in glob.glob(os.path.join(folder, "*.pdf")))
    )


@st.cache_resource(show_spinner=False)
def get_cached_vector_store(folder, signature):
    # O índice é lido do disco pelo rag_engine; o cache evita recarregá-lo a cada clique.
    return build_vector_store(folder)

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Study Abroad & Scholarship AI Finder",
    page_icon="🎓",
    layout="wide"
)

# Inicializa a base de dados SQLite
init_db()

st.title("🎓 Study Abroad & Scholarship AI Finder (Madrid 2027)")
st.markdown("Plataforma inteligente desenvolvida para centralizar a busca, análise de editais e gestão de candidaturas a bolsas de estudo.")

# Criação de separadores (Tabs) para organizar a aplicação
tab1, tab2 = st.tabs(["🔍 Pesquisa Inteligente (RAG)", "📋 Gestão de Bolsas (SQLite)"])

with tab1:
    st.header("Assistente Virtual de Editais")
    st.markdown("Faça perguntas em linguagem natural sobre os editais de bolsas guardados na pasta do projeto.")

    pergunta = st.text_input(
        "O que gostaria de saber sobre as bolsas?",
        value="Quais são os requisitos gerais para se candidatar às bolsas de estudo?"
    )

    if st.button("Consultar IA", type="primary"):
        scholarships_folder = "data/scholarships"
        if not os.path.exists(scholarships_folder) or not os.listdir(scholarships_folder):
            st.warning(f"⚠️ Nenhum ficheiro PDF encontrado na pasta `{scholarships_folder}`. Adicione os editais primeiro.")
        else:
            with st.spinner("A analisar editais e a gerar resposta com IA..."):
                try:
                    # Carrega (ou constrói) o vector store e a cadeia RAG
                    vector_store = get_cached_vector_store(
                        scholarships_folder, _folder_signature(scholarships_folder)
                    )
                    rag_chain = create_scholarship_rag_chain(vector_store)

                    resposta = rag_chain.invoke({"query": pergunta})

                    st.success("Resposta gerada com sucesso!")
                    st.write(resposta["result"])

                    st.subheader("📄 Fontes Consultadas nos Editais:")
                    for doc in resposta["source_documents"]:
                        fonte = doc.metadata.get("source", "PDF")
                        pagina = doc.metadata.get("page", 0) + 1
                        st.markdown(f"- **Ficheiro:** `{os.path.basename(fonte)}` | **Página:** {pagina}")

                except Exception as e:
                    st.error(f"Ocorreu um erro durante a consulta: {e}")

with tab2:
    st.header("Gestão de Oportunidades e Prazos")
    st.markdown("Registe novas bolsas ou consulte o estado das candidaturas armazenadas na base de dados relacional.")

    # Formulário para adicionar nova bolsa
    with st.form("add_scholarship_form"):
        st.subheader("Adicionar Nova Bolsa de Estudo")
        col1, col2 = st.columns(2)

        with col1:
            title = st.text_input("Título da Bolsa / Programa")
            institution = st.text_input("Instituição / Universidade")
            deadline = st.text_input("Prazo Limite (Ex: 2027-03-15)")

        with col2:
            pdf_source = st.text_input("Nome do Ficheiro PDF de Referência")
            status = st.selectbox("Estado da Candidatura", ["Pendente", "Em Análise", "Candidatura Enviada", "Aprovada"])

        requirements_summary = st.text_area("Resumo dos Requisitos Principais")

        submitted = st.form_submit_button("Guardar Bolsa na Base de Dados")
        if submitted:
            if title:
                add_scholarship(title, institution, deadline, requirements_summary, pdf_source, status)
                st.success(f"Bolsa '{title}' registrada com sucesso!")
            else:
                st.error("O campo do título é obrigatório.")

    st.divider()
    st.subheader("Bolsas Registradas")

    bolsas = get_all_scholarships()
    if bolsas:
        for b in bolsas:
            with st.expander(f"📌 {b[1]} ({b[2]}) - Estado: {b[6]}"):
                st.write(f"**Prazo Limite:** {b[3]}")
                st.write(f"**Requisitos:** {b[4]}")
                st.write(f"**Fonte PDF:** {b[5]}")
    else:
        st.info("Ainda não existem bolsas registradas na base de dados.")
