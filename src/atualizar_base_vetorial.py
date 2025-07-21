import sqlite3
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

def atualizar_base_com_feedback_sqlite(db_path):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT question, answer, comment
            FROM feedback
            WHERE feedback_type = 'dislike'
        """)
        rows = cursor.fetchall()

    if not rows:
        print("Nenhum feedback negativo com comentário encontrado.")
        return

    docs = [
        Document(
            page_content=f"Pergunta: {q}\nResposta ideal: {(c or '').strip() or 'Sem comentário'}",
            metadata={"tipo": "feedback_correção"}
        )
        for q, a, c in rows
    ]

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("vectorstore_feedback_patch")
    print(f"Base vetorial atualizada com {len(docs)} documentos!")

if __name__ == "__main__":
    atualizar_base_com_feedback_sqlite("feedback_data/feedback.db")
