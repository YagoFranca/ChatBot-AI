import os
import glob
import sqlite3
import re
import json
import torch
from datetime import datetime
from typing import Optional, Dict, List

import language_tool_python
import yaml
from flask import Flask, request, jsonify
from textblob import TextBlob
from spellchecker import SpellChecker

# LangChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.docstore.document import Document
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Nome do banco de dados para dados de suporte
DATABASE_PATH = "atendimentos.db"
FEEDBACK_DB_PATH = "feedback.db"
CONTEXT_DB_PATH = "context.db"

spell = SpellChecker(language='pt')
app = Flask(__name__)

# Memória e Contexto aprimorados
client_memories = {}
client_context = {}
client_conversation_history = {}  # Histórico completo das conversas

tool = language_tool_python.LanguageTool('pt-BR')

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.device_count())

# Palavras-chave para diferentes tipos de consulta - MELHORADAS
KEYWORDS_MAPPING = {
    'status_atendimento': ['status do atendimento', 'situação do atendimento', 'andamento do atendimento',
                           'como está o atendimento', 'estado do atendimento'],
    'defeito_atendimento': ['defeito do atendimento', 'problema do atendimento', 'erro do atendimento',
                            'falha do atendimento', 'avaria do atendimento', 'dano do atendimento'],
    'data_atendimento': ['data do atendimento', 'quando foi o atendimento', 'dia do atendimento',
                         'prazo do atendimento', 'tempo do atendimento'],
    'descricao_atendimento': ['descrição do atendimento', 'detalhes do atendimento',
                              'informações do atendimento', 'explicação do atendimento'],
    'garantia_atendimento': ['garantia do atendimento', 'cobertura do atendimento'],

    # Palavras-chave para consultas gerais sobre produtos (NÃO relacionadas a atendimento específico)
    'especificacao_produto': ['especificações técnicas', 'características do produto', 'funcionalidades',
                              'recursos do relógio', 'especificações do relógio', 'detalhes técnicos'],
    'preco_produto': ['preço', 'valor', 'custo', 'quanto custa', 'preço do relógio'],
    'garantia_produto': ['garantia', 'cobertura', 'prazo de garantia', 'política de garantia'],
    'produto_geral': ['relógio', 'produto', 'modelo', 'linha', 'coleção']
}


def corrigir_texto_pt(texto):
    """Corrige erros gramaticais e ortográficos em português"""
    try:
        matches = tool.check(texto)
        return language_tool_python.utils.correct(texto, matches)
    except Exception as e:
        print(f"Erro na correção gramatical: {e}")
        return texto


def init_context_db():
    """Inicializa banco de dados para armazenar contexto das conversas"""
    conn = sqlite3.connect(CONTEXT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS conversation_context (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        client_id TEXT NOT NULL,
        context_data TEXT NOT NULL,
        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    conn.close()


def save_context(client_id: str, context_data: Dict):
    """Salva contexto da conversa no banco de dados"""
    conn = sqlite3.connect(CONTEXT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""INSERT OR REPLACE INTO conversation_context 
                     (client_id, context_data, last_updated) 
                     VALUES (?, ?, ?)""",
                   (client_id, json.dumps(context_data), datetime.now()))
    conn.commit()
    conn.close()


def load_context(client_id: str) -> Dict:
    """Carrega contexto da conversa do banco de dados"""
    conn = sqlite3.connect(CONTEXT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT context_data FROM conversation_context WHERE client_id = ?", (client_id,))
    result = cursor.fetchone()
    conn.close()

    if result:
        return json.loads(result[0])
    return {}


def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_feedback_db_connection():
    conn = sqlite3.connect(FEEDBACK_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_feedback_db():
    conn = get_feedback_db_connection()
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        client_id TEXT NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        feedback_type TEXT,
        comment TEXT,
        context_info TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    conn.close()


# Inicializa bancos de dados
init_feedback_db()
init_context_db()


def is_atendimento_related_question(question: str) -> bool:
    """Verifica se a pergunta é especificamente sobre um atendimento"""
    question_lower = question.lower()

    # Palavras que indicam consulta sobre atendimento específico
    atendimento_indicators = [
        'atendimento', 'número do atendimento', 'atendimento número',
        'status do atendimento', 'situação do atendimento',
        'defeito do atendimento', 'problema do atendimento',
        'data do atendimento', 'quando foi o atendimento'
    ]

    # Verifica se contém número de atendimento
    has_atendimento_number = bool(re.search(r'atendimento\s*(?:número|de número|n°|nº)?\s*(\d+)', question_lower))

    # Verifica se contém indicadores de atendimento
    has_atendimento_indicators = any(indicator in question_lower for indicator in atendimento_indicators)

    return has_atendimento_number or has_atendimento_indicators


def classify_question_intent(question: str) -> str:
    """Classifica a intenção da pergunta com maior precisão"""
    question_lower = question.lower()

    # Primeiro verifica se é sobre atendimento específico
    if is_atendimento_related_question(question):
        # Classifica o tipo de consulta sobre atendimento
        for intent, keywords in KEYWORDS_MAPPING.items():
            if intent.endswith('_atendimento'):
                if any(keyword in question_lower for keyword in keywords):
                    return intent
        return 'atendimento_geral'

    # Se não é sobre atendimento, classifica como consulta geral
    for intent, keywords in KEYWORDS_MAPPING.items():
        if intent.endswith('_produto'):
            if any(keyword in question_lower for keyword in keywords):
                return intent

    return 'geral'


def extract_entities(question: str) -> Dict:
    """Extrai entidades da pergunta (números, datas, etc.)"""
    entities = {}

    # Extrai números de atendimento
    atendimento_match = re.search(r'atendimento\s*(?:número|de número|n°|nº)?\s*(\d+)', question, re.IGNORECASE)
    if atendimento_match:
        entities['atendimento_id'] = int(atendimento_match.group(1))

    # Extrai outros números que podem ser relevantes apenas se mencionou atendimento
    if 'atendimento' in question.lower():
        numbers = re.findall(r'\b\d+\b', question)
        if numbers and 'atendimento_id' not in entities:
            entities['possible_numbers'] = [int(n) for n in numbers]

    return entities


def get_conversation_history(client_id: str) -> List[Dict]:
    """Obtém histórico de conversas do cliente"""
    return client_conversation_history.get(client_id, [])


def add_to_conversation_history(client_id: str, question: str, answer: str, context: Dict):
    """Adiciona interação ao histórico de conversas"""
    if client_id not in client_conversation_history:
        client_conversation_history[client_id] = []

    client_conversation_history[client_id].append({
        'timestamp': datetime.now().isoformat(),
        'question': question,
        'answer': answer,
        'context': context
    })

    # Mantém apenas as últimas 10 interações
    if len(client_conversation_history[client_id]) > 10:
        client_conversation_history[client_id] = client_conversation_history[client_id][-10:]


def buscar_atendimento_inteligente(question: str, client_id: str) -> Optional[str]:
    """Busca atendimento com melhor compreensão de contexto e precisão"""

    # Primeiro verifica se é realmente uma pergunta sobre atendimento
    if not is_atendimento_related_question(question):
        return None

    conn = get_db_connection()
    cursor = conn.cursor()

    # Classifica a intenção da pergunta
    intent = classify_question_intent(question)
    entities = extract_entities(question)

    # Carrega contexto salvo
    context = load_context(client_id)

    # Busca ID do atendimento
    atendimento_id = None
    if 'atendimento_id' in entities:
        atendimento_id = entities['atendimento_id']
    elif 'possible_numbers' in entities and len(entities['possible_numbers']) == 1:
        # Se há apenas um número na pergunta e mencionou atendimento, assume que é o ID
        atendimento_id = entities['possible_numbers'][0]
    elif 'current_atendimento' in context:
        atendimento_id = context['current_atendimento']
    elif client_id in client_context:
        atendimento_id = client_context[client_id]

    if atendimento_id:
        cursor.execute("SELECT * FROM atendimentos WHERE id = ?", (atendimento_id,))
        atendimento = cursor.fetchone()

        if atendimento:
            # Atualiza contexto
            context['current_atendimento'] = atendimento_id
            context['last_query_intent'] = intent
            context['atendimento_data'] = dict(atendimento)

            # Salva contexto
            save_context(client_id, context)
            client_context[client_id] = atendimento_id

            # Responde baseado na intenção específica
            if intent == 'status_atendimento':
                return f"O status do atendimento {atendimento_id} é: **{atendimento['status']}**"
            elif intent == 'defeito_atendimento':
                return f"O defeito registrado no atendimento {atendimento_id} é: **{atendimento['defeito']}**"
            elif intent == 'descricao_atendimento':
                return f"A descrição do atendimento {atendimento_id} é: **{atendimento['descricao']}**"
            elif intent == 'data_atendimento':
                return f"A data do atendimento {atendimento_id} é: **{atendimento['data']}**"
            elif intent == 'garantia_atendimento':
                garantia_info = f"Informações de garantia para o atendimento {atendimento_id}:"
                if atendimento.get('garantia'):
                    garantia_info += f"\n**Garantia:** {atendimento['garantia']}"
                else:
                    garantia_info += "\n**Garantia:** Consulte os documentos do produto para detalhes da garantia."
                return garantia_info
            else:
                # Resposta completa para consultas gerais sobre atendimento
                return (f"**Informações do atendimento {atendimento_id}:**\n"
                        f"**Cliente:** {atendimento['cliente_nome']}\n"
                        f"**Status:** {atendimento['status']}\n"
                        f"**Data:** {atendimento['data']}\n"
                        f"**Defeito:** {atendimento['defeito']}")
        else:
            return f"Não encontrei o atendimento número {atendimento_id}. Verifique se o número está correto."

    # Se mencionou atendimento mas não conseguiu identificar o número
    if "atendimento" in question.lower():
        return "Para consultar informações sobre atendimento, preciso do número do atendimento. Você pode me informar o número?"

    return None


def create_enhanced_retriever():
    """Cria um retriever melhorado com chunking inteligente"""
    documents = load_documents_enhanced("documents")

    # Usa chunking mais inteligente
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    split_documents = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cuda'}
    )

    vectorstore = FAISS.from_documents(split_documents, embeddings)

    # Retorna mais documentos relevantes
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )


def load_documents_enhanced(folder_path):
    """Carrega documentos com metadados aprimorados"""
    documents = []
    for filepath in glob.glob(os.path.join(folder_path, "*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            filename = os.path.basename(filepath)
            documents.append(Document(
                page_content=content,
                metadata={
                    "source": filepath,
                    "filename": filename,
                    "doc_type": "knowledge_base"
                }
            ))
    return documents


def create_enhanced_prompt():
    """Cria prompt melhorado para o modelo"""
    return PromptTemplate(
        template="""Você é um assistente virtual especializado no atendimento ao cliente para relógios Abyss Precision.

CONTEXTO DA CONVERSA:
{chat_history}

INFORMAÇÕES RELEVANTES:
{context}

PERGUNTA ATUAL: {question}

INSTRUÇÕES:
1. Responda SEMPRE em português brasileiro
2. Use as informações do contexto para dar respostas precisas
3. Para perguntas sobre especificações técnicas, características ou informações de produtos, use APENAS as informações dos documentos
4. Para perguntas sobre atendimentos específicos, essas são tratadas separadamente
5. Se não souber algo, seja honesto e ofereça ajuda alternativa
6. Mantenha tom profissional mas amigável
7. Considere o histórico da conversa para manter contexto

RESPOSTA:""",
        input_variables=["chat_history", "context", "question"]
    )


# Carrega documentos e cria retriever melhorado
retriever = create_enhanced_retriever()

# Modelo de chat otimizado
chat = ChatOllama(
    model="llama3",
    temperature=0.1,  # Pouca variação para mais consistência
    top_p=0.9,
    repeat_penalty=1.1
)

# Mensagem do sistema aprimorada
system_message = SystemMessage(content=(
    "Você é um assistente virtual especializado da Abyss Precision, marca premium de relógios. "
    "Você tem acesso a informações sobre produtos, garantias, manutenções e serviços técnicos. "
    "SEMPRE responda em português brasileiro, de forma clara e profissional. "
    "Use o contexto da conversa para dar respostas mais precisas e personalizadas. "
    "Para perguntas sobre especificações técnicas, características de produtos ou informações gerais, "
    "use APENAS as informações disponíveis nos documentos da base de conhecimento. "
    "Se não souber algo específico, seja honesto e ofereça alternativas de ajuda. "
    "Para perguntas fora do escopo (não relacionadas a relógios, atendimento ou produtos), "
    "responda: 'Desculpe, mas só posso ajudar com questões relacionadas aos nossos produtos e serviços.'"
))


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    if not data or "client_id" not in data or "question" not in data:
        return jsonify({"error": "Requisição inválida. Forneça 'client_id' e 'question'."}), 400

    client_id = data["client_id"]
    question_original = data["question"]
    question = corrigir_texto_pt(question_original)

    print(f"[Cliente {client_id}] Pergunta original: {question_original}")
    print(f"[Cliente {client_id}] Pergunta corrigida: {question}")

    # Verifica se é sobre atendimento específico com IA melhorada
    atendimento_resposta = buscar_atendimento_inteligente(question, client_id)
    if atendimento_resposta:
        # Adiciona ao histórico
        add_to_conversation_history(client_id, question, atendimento_resposta,
                                    load_context(client_id))
        return jsonify({"answer": atendimento_resposta})

    # Para perguntas gerais sobre produtos, usa o sistema de RAG
    # Recupera ou cria a memória de conversa
    if client_id not in client_memories:
        client_memories[client_id] = ConversationSummaryBufferMemory(
            llm=chat,
            max_token_limit=800,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

    memory = client_memories[client_id]

    # Adiciona mensagem do sistema se necessário
    if not memory.chat_memory.messages:
        memory.chat_memory.messages.append(system_message)

    # Cria cadeia de conversação melhorada
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True
    )

    # Executa a cadeia
    result = qa_chain.invoke({"question": question, "chat_history": memory})
    answer = result.get("answer", "")
    source_docs = result.get("source_documents", [])

    # Adiciona informações sobre fontes se relevante
    if source_docs and len(answer) > 50:  # Só para respostas substanciais
        answer += f"\n\n(Baseado em {len(source_docs)} fonte(s) de informação)"

    # Registra feedback para perguntas fora do escopo
    if "só posso ajudar com questões relacionadas" in answer:
        conn = get_feedback_db_connection()
        cursor = conn.cursor()
        context_info = json.dumps(load_context(client_id))
        cursor.execute(
            "INSERT INTO feedback (client_id, question, answer, feedback_type, comment, context_info) VALUES (?, ?, ?, ?, ?, ?)",
            (client_id, question_original, answer, "out_of_scope",
             "Pergunta identificada como fora do escopo.", context_info))
        conn.commit()
        conn.close()

    # Adiciona ao histórico de conversas
    add_to_conversation_history(client_id, question, answer, load_context(client_id))

    return jsonify({"answer": answer})


@app.route('/feedback', methods=['POST'])
def receive_feedback():
    try:
        data = request.get_json()
        required_fields = ["client_id", "question", "answer", "feedback_type"]

        if not data or not all(field in data for field in required_fields):
            return jsonify({
                "error": "Requisição inválida. Forneça 'client_id', 'question', 'answer' e 'feedback_type'."
            }), 400

        client_id = data["client_id"]
        question = data["question"]
        answer = data["answer"]
        feedback_type = data["feedback_type"]
        comment = data.get("comment", None)

        # Carrega contexto
        context_info = json.dumps(load_context(client_id))

        # Insere no banco
        conn = get_feedback_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO feedback (client_id, question, answer, feedback_type, comment, context_info) VALUES (?, ?, ?, ?, ?, ?)",
            (client_id, question, answer, feedback_type, comment, context_info)
        )
        conn.commit()
        conn.close()

        return jsonify({"status": "Feedback recebido com sucesso!"}), 200

    except Exception as e:
        # Loga erro no console
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Erro interno no servidor", "details": str(e)}), 500



if __name__ == '__main__':
    app.run(port=5000, debug=True)