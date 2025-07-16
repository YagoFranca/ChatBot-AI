# 🤖 Chatbot de Atendimento para Marca de Relógios

Este projeto é um assistente virtual inteligente desenvolvido em Python com Flask e LangChain. Ele é especializado em atendimento ao cliente para uma renomada marca de relógios, oferecendo respostas sobre produtos, garantias, manutenções e suporte técnico, além de consultar informações de atendimentos registrados em banco de dados.

---

## 🧠 Funcionalidades

- ✅ Suporte inteligente baseado em IA (modelo LLM via Ollama)
- 🔍 Busca semântica de conhecimento com FAISS + embeddings
- 📝 Correção ortográfica automática (TextBlob + LanguageTool + pyspellchecker)
- 🗃️ Integração com banco de dados SQLite para histórico de atendimentos
- 💬 Suporte a conversas com memória (LangChain Memory)
- 🌐 API HTTP com Flask para interação via POST `/ask`
- 🖥️ Interface amigável com **Streamlit**

---

## 🚀 Como executar localmente

### 1. Clone o repositório


git clone https://github.com/seu-usuario/chatbot-relogios.git
cd chatbot-relogios

### 2. Crie e ative um ambiente virtual
bash
Copy
Edit
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

### 3. Instale as dependências
bash
Copy
Edit
pip install -r requirements.txt

###  4. Baixe e rode o modelo via Ollama (ex: llama3)
bash
Copy
Edit
ollama pull llama3
ollama serve

### 🛠 Como usar

### 📡 Passo 1: Inicie o servidor de backend com Flask
bash
Copy
Edit
python src/app.py
O servidor estará disponível em http://localhost:5000/ask.

### 💬 Passo 2: Rode a interface do chatbot com Streamlit
bash
Copy
Edit
streamlit run chat.py
O chat.py se conecta ao servidor Flask via POST e permite interação em tempo real com o assistente virtual.

