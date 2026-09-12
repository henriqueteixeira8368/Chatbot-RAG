# Chatbot RAG - Documentação PI-V

Este projeto implementa um chatbot com arquitetura RAG rodando 100% localmente e offline.

## Stack Tecnológica Escolhida
* **Linguagem:** Python
* **Interface Web:** Streamlit
* **Framework RAG:** LangChain
* **Vector Store:** FAISS (Local em memória)
* **LLM:** Ollama (Modelo `llama3`)
* **Embeddings:** Ollama (Modelo `nomic-embed-text`)

**Justificativa:** Optamos por essa stack para garantir Custo ZERO, máxima privacidade de dados (rodando 100% local) e facilidade de deploy.

## Requisitos Prévios
1. Python 3.9+ instalado.
2. [Ollama](https://ollama.com/) instalado na sua máquina.
3. Um arquivo chamado `documentacao_PI-V.pdf` na raiz do projeto.

## Preparando os Modelos Locais
Antes de rodar a aplicação, abra o terminal e baixe os modelos necessários no Ollama:
```bash
ollama pull llama3
ollama pull nomic-embed-text