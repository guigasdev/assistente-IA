import os
import json
from flask import Flask, request, jsonify
from openai import AzureOpenAI
from dotenv import load_dotenv
from flask_cors import CORS

# Carregar variáveis de ambiente
load_dotenv()

# Configuração da API
endpoint = os.getenv("AZURE_OPENAI_API_BASE")
deployment = os.getenv("OPENIA_INSTANCIA", "gpt-4o")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

# Inicializar cliente Azure OpenAI
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2024-05-01-preview",
)

app = Flask(__name__)
CORS(app)

HISTORY_FILE = "conversas.json"

def save_conversation(pergunta, resposta):
    conversation = {"pergunta": pergunta, "resposta": resposta}
    with open(HISTORY_FILE, "a", encoding='utf-8') as file:
        json.dump(conversation, file, ensure_ascii=False)
        file.write("\n")

def load_history():
    history = []
    try:
        with open(HISTORY_FILE, "r", encoding='utf-8') as file:
            for line in file:
                history.append(json.loads(line))
    except FileNotFoundError:
        pass
    return history

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    role_system = data.get("role_system", "Você é um assistente de IA.")
    pergunta = data.get("pergunta", "")
    
    if not pergunta.strip():
        return jsonify({"erro": "Pergunta vazia."}), 400
    
    chat_prompt = [
        {"role": "system", "content": role_system},
        {"role": "user", "content": pergunta},
    ]
    
    try:
        completion = client.chat.completions.create(
            model=deployment,
            messages=chat_prompt,
            max_tokens=800,
            temperature=0.7
        )
        resposta = completion.choices[0].message.content.strip()
        save_conversation(pergunta, resposta)
        return jsonify({"pergunta": pergunta, "resposta": resposta})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route("/historico", methods=["GET"])
def historico():
    return jsonify(load_history())

if __name__ == "__main__":
    app.run(debug=True)
