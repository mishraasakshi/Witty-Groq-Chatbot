# 🤖 Witty Groq Chatbot

A fast, sarcastically smart chatbot built with the GenAI stack: **LangChain + Groq + Streamlit**.

## 🚀 Features
- **High-speed inference:** uses `openai/gpt-oss-120b` via the Groq API (switched from Llama 3.3 70B after Groq retired it in August 2026).
- **Witty persona:** a custom system prompt gives the bot a humorous, sarcastic personality.
- **Chat UI:** built with Streamlit's chat components.
- **Session memory:** remembers the conversation within a session.
- **Safe secret handling:** reads `GROQ_API_KEY` from a local `.env` file or Streamlit secrets - never from the code.

## 🛠️ Tech Stack
- **Language:** Python
- **Orchestration:** LangChain (`langchain-groq`)
- **LLM:** `openai/gpt-oss-120b` on Groq
- **Frontend:** Streamlit
- **Deployment:** Streamlit Community Cloud

## 📦 Run locally
1. Clone the repo:
   ```bash
   git clone https://github.com/mishraasakshi/Witty-Groq-Chatbot.git
   cd Witty-Groq-Chatbot
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your own Groq API key (get one free at console.groq.com):
   ```
   GROQ_API_KEY=your_key_here
   ```
4. Start the app:
   ```bash
   streamlit run app.py
   ```

## ☁️ Deploy on Streamlit Community Cloud
Add `GROQ_API_KEY` under **App settings -> Secrets**; the app reads it automatically.
