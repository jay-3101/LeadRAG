# 🧠 LeadRAG – Leader Embeddings with Keyword Corpus in RAG

**LeadRAG** is a secure and intelligent Retrieval-Augmented Generation (RAG) system that enhances information retrieval through leader embeddings and a keyword-driven corpus. It combines semantic and lexical techniques to provide high-quality, context-aware answers. Role-based access and admin capabilities ensure a controlled and customizable environment.

---

## 🚀 Features

- 🔍 **Hybrid Retrieval**: Combines leader embeddings with keyword search for improved document matching.
- 🧠 **RAG-based QA**: Retrieves context and uses LLMs to generate accurate answers.
- 🧱 **Modular Architecture**: Easy to integrate or extend with custom components.
- 🔐 **Role-Based Access**: Includes user and admin modes for controlled data access.
- 📊 **Dashboard Views**: Guided interactions and feedback visualizations

## 📦 Installation & Setup

### 1. Clone the Repository
### 1. Clone the Repository

```bash
git clone https://github.com/jay-3101/LeadRAG.git
cd RAG_Chatbot
```

### 2. Set Up Your GROQ API Key
This project uses GROQ’s free API to call the LLaMA model(Open Source Model.

Steps:
Sign up at https://console.groq.com/keys

Generate your GROQ API key


### 3.Running the Project
#### 1. Using shell Script
Use the run.sh script to automatically install dependencies, export your API key, and start the backend server:
```bash
chmod +x run.sh  # (Only once)
./run.sh your_groq_api_key_here
```

This will:

Export the GROQ_API_KEY to your environment

Install Python dependencies

Start the backend server on http://127.0.0.1:5000

#### 2. Manual Setup
In the env file Update

GROQ_API_KEY=your_groq_api_key_here

```bash
pip install -r requirements.txt
cd backend
python app.py
```

The server will start on http://127.0.0.1:5000
Follow the on-screen instructions for signup, login, and interacting with the system.

## 👤 Admin Access Instructions
By default, new users are registered as regular users.
To make a user an admin, follow these steps:

After the user signs up, open the file:
backend/user.json

Locate the user entry and modify their role like this:
{
  "username": "your_username",
  "password": "hashed_password",
  "role_type": "admin"
}

⚠️ Important: Modify role_type carefully to maintain system security. Admins may access sensitive features or data.


