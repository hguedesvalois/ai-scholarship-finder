import sqlite3
import os

# Define o caminho onde o ficheiro da base de dados será guardado
DB_PATH = "data/scholarships.db"

def get_connection():
    """Garante que a pasta 'data' existe e retorna uma conexão com o SQLite."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    """Cria a tabela de bolsas de estudo caso ela ainda não exista."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scholarships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            institution TEXT,
            deadline TEXT,
            requirements_summary TEXT,
            pdf_source TEXT,
            status TEXT DEFAULT 'Pendente'
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ Base de dados relacional SQLite inicializada com sucesso!")

def add_scholarship(title, institution, deadline, requirements_summary, pdf_source, status="Pendente"):
    """Adiciona um novo registo de bolsa de estudo na base de dados."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO scholarships (title, institution, deadline, requirements_summary, pdf_source, status)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (title, institution, deadline, requirements_summary, pdf_source, status))

    conn.commit()
    conn.close()
    print(f"📌 Bolsa '{title}' inserida com sucesso!")

def get_all_scholarships():
    """Retorna todas as bolsas guardadas na base de dados."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM scholarships')
    rows = cursor.fetchall()

    conn.close()
    return rows


# Bloco de teste direto no terminal
if __name__ == "__main__":
    print("🚀 A configurar a base de dados do Projeto 2...")
    init_db()

    # Inserção de exemplo para teste
    add_scholarship(
        title="Becas de Postgrado - Fundación Carolina",
        institution="Universidad Politécnica de Madrid",
        deadline="2027-03-15",
        requirements_summary="Nacionalidade brasileira, diploma universitário e proficiência em espanhol.",
        pdf_source="edital_fundacion_carolina.pdf",
        status="Pendente"
    )

    print("\n📋 Listagem de Bolsas na Base de Dados:")
    bolsas = get_all_scholarships()
    for bolsa in bolsas:
        print(f"- ID: {bolsa[0]} | Título: {bolsa[1]} | Instituição: {bolsa[2]} | Prazo: {bolsa[3]} | Estado: {bolsa[6]}")
