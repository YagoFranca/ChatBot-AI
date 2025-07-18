import streamlit as st
import requests
import uuid
import json
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="Abyss Precision - Atendimento DeepDive 1000M",
    page_icon="⌚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para melhor aparência
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f4e79;
        margin-bottom: 2rem;
    }
    .sub-header {
        text-align: center;
        color: #2c5aa0;
        margin-bottom: 1rem;
    }
    .feedback-container {
        display: flex;
        gap: 10px;
        margin-top: 10px;
    }
    .context-info {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .stButton > button {
        border-radius: 20px;
        border: none;
        padding: 0.25rem 0.5rem;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Cabeçalho principal
st.markdown("<h1 class='main-header'>⌚ Abyss Precision</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='sub-header'>Atendimento Exclusivo para DeepDive 1000M</h3>", unsafe_allow_html=True)

# Layout com colunas
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔄 Nova Conversa", help="Reinicia a conversa"):
        # Gera novo client_id
        st.session_state.client_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# Sidebar com informações contextuais
with st.sidebar:
    st.markdown("### 📋 Informações da Sessão")

    # Gera um client_id único para a sessão
    if "client_id" not in st.session_state:
        st.session_state.client_id = str(uuid.uuid4())

    st.text(f"ID da Sessão: {st.session_state.client_id[:8]}...")

    # Mostra contexto atual se disponível
    if st.button("🔍 Ver Contexto"):
        try:
            context_response = requests.get(f"http://127.0.0.1:5000/context/{st.session_state.client_id}")
            if context_response.status_code == 200:
                context_data = context_response.json()
                with st.expander("Contexto da Conversa"):
                    st.json(context_data)
        except Exception as e:
            st.error(f"Erro ao buscar contexto: {e}")

    st.markdown("### 💡 Dicas de Uso")
    st.markdown("""
    - Para consultar atendimento, informe o número
    - Seja específico nas perguntas
    - Use feedback para melhorar as respostas
    - O sistema lembra do contexto da conversa
    """)

    st.markdown("### 🔧 Exemplos de Perguntas")
    examples = [
        "Qual o status do atendimento 123?",
        "Qual a garantia do DeepDive 1000M?",
        "Como fazer manutenção preventiva?",
        "Quais são as especificações técnicas?"
    ]

    for example in examples:
        if st.button(f"💬 {example}", key=f"example_{example[:20]}"):
            st.session_state.example_question = example

# Inicializa histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []


# Funções para interagir com a API
def ask_question(question):
    """Envia pergunta para a API e retorna resposta"""
    url = "http://127.0.0.1:5000/ask"
    payload = {
        "client_id": st.session_state.client_id,
        "question": question
    }

    try:
        response = requests.post(url, json=payload, timeout=100)
        if response.status_code == 200:
            return response.json().get("answer", "Erro ao obter resposta.")
        else:
            return f"Erro {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Erro de conexão: {e}"


def send_feedback(question, answer, feedback_type, comment=None):
    """Envia feedback para a API"""
    url = "http://127.0.0.1:5000/feedback"
    payload = {
        "client_id": st.session_state.client_id,
        "question": question,
        "answer": answer,
        "feedback_type": feedback_type,
        "comment": comment
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def format_answer(answer):
    """Formata a resposta para melhor visualização"""
    # Se a resposta tem informações estruturadas, formata melhor
    if "Informações do atendimento" in answer:
        lines = answer.split('\n')
        formatted = []
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                formatted.append(f"**{key.strip()}:** {value.strip()}")
            else:
                formatted.append(line)
        return '\n'.join(formatted)

    return answer


# Área principal de chat
st.markdown("### 💬 Conversa")

# Exibe histórico de mensagens
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Formata e exibe resposta
            formatted_answer = format_answer(message["content"])
            st.markdown(formatted_answer)

            # Botões de feedback
            col1, col2, col3 = st.columns([1, 1, 6])
            with col1:
                if st.button("👍", key=f"like_{i}", help="Resposta útil"):
                    if send_feedback(st.session_state.messages[i - 1]["content"],
                                     message["content"], "like"):
                        st.success("✅ Feedback positivo enviado!")
                    else:
                        st.error("❌ Erro ao enviar feedback")

            with col2:
                if st.button("👎", key=f"dislike_{i}", help="Resposta não útil"):
                    if send_feedback(st.session_state.messages[i - 1]["content"],
                                     message["content"], "dislike"):
                        st.success("✅ Feedback negativo enviado!")
                    else:
                        st.error("❌ Erro ao enviar feedback")
        else:
            st.markdown(message["content"])

# Entrada do usuário
prompt_input = st.chat_input("Digite sua pergunta sobre o DeepDive 1000M...")

# Processa pergunta de exemplo se foi clicada
if "example_question" in st.session_state:
    prompt_input = st.session_state.example_question
    del st.session_state.example_question

if prompt_input:
    # Adiciona mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt_input})

    # Mostra mensagem do usuário
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Obtém resposta da API
    with st.chat_message("assistant"):
        with st.spinner("Processando sua pergunta..."):
            answer = ask_question(prompt_input)

        # Formata e exibe resposta
        formatted_answer = format_answer(answer)
        st.markdown(formatted_answer)

        # Adiciona resposta ao histórico
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Botões de feedback para nova resposta
        col1, col2, col3 = st.columns([1, 1, 6])
        with col1:
            if st.button("👍", key="like_new", help="Resposta útil"):
                if send_feedback(prompt_input, answer, "like"):
                    st.success("✅ Feedback positivo enviado!")
                else:
                    st.error("❌ Erro ao enviar feedback")

        with col2:
            if st.button("👎", key="dislike_new", help="Resposta não útil"):
                if send_feedback(prompt_input, answer, "dislike"):
                    st.success("✅ Feedback negativo enviado!")
                else:
                    st.error("❌ Erro ao enviar feedback")

# Rodapé com informações
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>
        Abyss Precision - Atendimento Inteligente | 
        Powered by IA | 
        Sessão: {session_id}
    </small>
</div>
""".format(session_id=st.session_state.client_id[:8]), unsafe_allow_html=True)

# Informações de debug (só em development)
if st.checkbox("🔧 Modo Debug"):
    st.markdown("### Debug Info")
    st.json({
        "client_id": st.session_state.client_id,
        "messages_count": len(st.session_state.messages),
        "timestamp": datetime.now().isoformat()
    })