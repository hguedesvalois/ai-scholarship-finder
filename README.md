## 📌 Descrição do Projeto
Um assistente inteligente de recomendação e busca de bolsas de estudo na Europa e Estados Unidos. O sistema combina dados estruturados de programas e universidades armazenados em um banco de dados local **SQLite** (consultados via **SQL**) com documentos detalhados de edital processados via **RAG**. A inferência do modelo de linguagem é realizada utilizando os modelos open-weights da **Groq Cloud**, sem custos de API.

---

## 🛠️ Competências Aprendidas & Praticadas
- **SQL (Banco de Dados Relacional):** Criação de tabelas, inserção de dados e consultas com SQL (`SELECT`, `WHERE`, `JOIN`).
- **Python & SQLite:** Utilização da biblioteca `sqlite3` para integrar Python a um banco de dados relacional.
- **RAG com LLM Open-Source:** Integração do LangChain com a **Groq API** (`langchain-groq`).
- **APIs & Web Requests:** Consumo e gerenciamento de endpoints de inferência em nuvem.
- **Engenharia de Prompts:** Formatação de instruções para modelos como LLaMA 3.

---

## 📂 Estrutura de Arquivos

```text
scholarship-ai-finder/
│
├── data/                  # Editais e documentos em PDF/Texto
│   ├── scholarships/
│   ├── faiss_index/       # Índice vetorial FAISS (gerado automaticamente)
│   └── scholarships.db    # Banco de dados SQLite local
│
├── .env                   # Variáveis de ambiente (GROQ_API_KEY)
├── .gitignore             # Arquivos ignorados pelo Git
├── requirements.txt       # Dependências do projeto
├── README.md              # Documentação do projeto para o GitHub
├── rag_engine.py          # Processamento de editais e busca vetorial
├── database.py            # Funções de consulta SQL ao banco de dados
└── app.py                 # Interface Streamlit
```

---

## 📦 Dependências (`requirements.txt`)

```text
streamlit==1.32.0
langchain==0.1.20
langchain-community==0.0.38
langchain-core==0.1.52
langchain-text-splitters==0.0.2
langchain-groq==0.0.1
sentence-transformers==2.5.1
faiss-cpu==1.15.1
pypdf==4.0.0
python-dotenv==1.0.0
```

> **Nota:** instale o PyTorch **CPU-only** antes do `requirements.txt`, caso contrário o pip
> descarrega ~5 GB de bibliotecas CUDA/NVIDIA que o projeto não utiliza (os embeddings
> correm na CPU):
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

---

## 🚀 Instalação e Execução

1. **Clonar o repositório e acessar a pasta:**
   ```bash
   git clone https://github.com/hguedesvalois/ai-scholarship-finder.git
   cd ai-scholarship-finder
   ```

2. **Criar e ativar o ambiente virtual (venv):**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

3. **Instalar dependências (PyTorch CPU-only primeiro):**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

4. **Configurar a API Key no `.env`:**
   ```env
   GROQ_API_KEY="gsk_sua_chave_groq_aqui"
   ```

5. **Executar a aplicação:**
   ```bash
   streamlit run app.py
   ```
