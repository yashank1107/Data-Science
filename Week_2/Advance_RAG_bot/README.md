# 🚀 Enhanced Multi-modal RAG Chatbot

Welcome to the Enhanced Multi-modal RAG Chatbot, a sophisticated, full-stack application designed for advanced conversational AI. This project integrates multiple Large Language Models (LLMs), Retrieval-Augmented Generation (RAG) techniques, and real-time observability, all packaged into a responsive and user-friendly interface.

The system features a **Streamlit** frontend for dynamic user interaction and a robust **FastAPI** backend that orchestrates LLM calls, document processing, and state management.

![Project Demo GIF](https://path.to/your/demo.gif)
*(Suggestion: Add a GIF or screenshot of your application in action here.)*

---

## ✨ Key Features

This project is packed with features that make it a powerful and flexible platform for conversational AI:

#### 🤖 Conversational AI Core
- **Multi-LLM Support**: Seamlessly switch between different LLM providers, including **Google Gemini**, **Groq** (Llama 3.1, Gemma2), and **Cohere**.
- **Advanced RAG Variants**: Choose from multiple RAG strategies (`basic`, `knowledge_graph`, `hybrid`) to enhance responses with contextual data.
  - **Basic RAG**: A standard approach that retrieves relevant text chunks from a vector database to provide context.
  - **Knowledge Graph RAG**: Constructs a knowledge graph from documents to answer queries requiring multi-hop reasoning and understanding of relationships between entities.
  - **Hybrid RAG**: A sophisticated blend of vector search and knowledge graph querying, offering the benefits of both semantic similarity and structured data retrieval.
- **Multi-modal Input**: Interact with the chatbot using both **text and images**.
- **Internet Search Integration**: Empowers the chatbot to access real-time information from the web for up-to-date answers.

#### 📄 Document & State Management
- **Comprehensive Document Handling**: Upload and process various file types (`.pdf`, `.txt`, `.docx`, `.png`, `.jpg`, etc.).
- **Selective Document Context**: Choose specific uploaded documents to be used as context for the chat session.
- **Persistent Conversation Memory**: Maintains conversation history for coherent, multi-turn dialogues.

#### 🖥️ User Interface & Experience
- **Real-time System Dashboard**: A "System Status" panel provides live updates on the selected LLM and active features.
- **Dynamic Configuration**: Users can change the LLM, RAG variant, and other settings on-the-fly via the sidebar.
- **Interactive UI**: Built with Streamlit for a responsive and intuitive user experience.

#### 🛡️ Backend & Observability
- **Robust FastAPI Backend**: A scalable and efficient backend serving the AI logic.
- **Enhanced Guardrails**: Implements safety checks to validate and secure requests.
- **Full Observability Stack**: Integrated with **Opik (OpenTelemetry)**, **Jaeger**, and **Prometheus** for comprehensive monitoring of traces and metrics.
- **Containerized Services**: Uses Docker and Docker Compose for easy setup and deployment of the observability stack.

---

## 🏗️ Project Architecture

The application is built on a decoupled frontend-backend architecture, ensuring scalability and maintainability.

```
  +------------------------+      +-------------------------+
  |   Streamlit Frontend   |      |   Observability Stack   |
  | (frontend/app.py)      |<---->|   (Docker Compose)      |
  +------------------------+      | - Opik Collector        |
             |                    | - Jaeger (Traces)       |
             | (HTTP API Calls)   | - Prometheus (Metrics)  |
             v                    +-------------------------+
  +------------------------+                  ^
  |    FastAPI Backend     |                  | (OTLP)
  |  (backend/app/main.py) |------------------+
  +------------------------+
             |
             | (Orchestration)
             v
  +------------------------+
  |   Backend Services     |
  | - LLM Service          |
  | - RAG Service          |
  | - Document Processor   |
  | - Guardrails           |
  | - Conversation Memory  |
  +------------------------+
```

1.  **Frontend (Streamlit)**: The user interacts with the Streamlit UI to send messages, upload documents, and configure settings. It communicates with the backend via HTTP requests.
2.  **Backend (FastAPI)**: Receives requests from the frontend. It uses various services to process the request:
    - **LLM Service**: Connects to external LLM APIs (Gemini, Groq, Cohere).
    - **RAG Service**: Retrieves relevant context from documents or the internet.
    - **Document Processor**: Manages uploaded files and extracts content.
    - **Guardrails & Memory**: Ensure safety and maintain conversation history.
3.  **Observability Stack (Docker)**: The backend sends telemetry data (traces, metrics) to the **Opik OpenTelemetry Collector**, which then forwards it to Jaeger for tracing and Prometheus for metrics.

---

## 🤖 Available LLMs & Comparison

The application supports a variety of powerful LLMs, each with unique strengths. The models are dynamically loaded based on the API keys provided in your `.env` file.

| Provider | Model Name                  | Context Window | Key Strengths                                                                  |
| :------- | :-------------------------- | :------------- | :----------------------------------------------------------------------------- |
| **Groq** | `llama-3.1-8b-instant`      | 128k           | Blazing-fast inference speed, ideal for real-time conversational applications. |
|          | `gemma2-9b-it`              | 8k             | Strong reasoning and instruction-following capabilities from Google.             |
|          | `mixtral-8x7b-32768`        | 32k            | Excellent performance on a wide range of benchmarks; a powerful open model.    |
| **Google** | `gemini-1.5-flash`          | 1M             | Massive context window, multi-modal capabilities, and a balance of speed and performance. |
|          | `gemini-2.0-flash` (Custom) | ~128k (est.)   | *Assumed to be a custom or fine-tuned variant for specific flash-style tasks.* |
|          | `gemini-2.5-flash` (Custom) | ~128k (est.)   | *Assumed to be a custom or fine-tuned variant for specific flash-style tasks.* |
| **Cohere** | `command-r-plus-08-2024`    | 128k           | Enterprise-grade model optimized for RAG, tool use, and multi-lingual tasks.   |
|          | `command-a-vision-07-2025`  | ~128k (est.)   | *Assumed to be a custom or fine-tuned variant with a focus on vision capabilities.* |
|          | `command-a-03-2025`         | ~128k (est.)   | *Assumed to be a custom or fine-tuned variant for general-purpose tasks.*      |

*Note: Some model names in your configuration appear to be custom or speculative (e.g., with future dates). The stats provided are based on the closest publicly known base models. The context windows are token limits.*

---

## 🛠️ Tech Stack

| Category          | Technology                                       |
| ----------------- | ------------------------------------------------ |
| **Frontend**      | Streamlit                                        |
| **Backend**       | FastAPI, Pydantic, Uvicorn                        |
| **LLM Providers** | Google Gemini, Groq, Cohere                      |
| **Web Search**    | Serper API                                       |
| **Data & RAG**    | LangChain, Vector Store (e.g., ChromaDB/FAISS)   |
| **Observability** | OpenTelemetry, Opik, Jaeger, Prometheus          |
| **Containerization**| Docker, Docker Compose                           |
| **Language**      | Python 3.9+                                      |

---

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose
- An IDE like VS Code

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Final_project
```

### 2. Backend Setup

First, set up and run the FastAPI backend.

```bash
# Navigate to the backend directory
cd backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install backend dependencies
pip install -r requirements.txt

# Go back to the project root to create the .env file
cd ..
```

**Create the Environment File:**

In the root directory of `Final_project`, create a file named `.env` and populate it with your API keys. Use the provided `.env` file in the context as a template:

```env
# .env

# LLM API Keys
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIzaSy...
COHERE_API_KEY=jLlVvW...

# Serper API Key (for web search)
SERPER_API_KEY=153b2f...

# Observability
OPIK_API_KEY=5bM9Jz...
OPIK_ENDPOINT="http://localhost:4318"
```

**Run the Backend Server:**

```bash
# Navigate back to the backend app directory
cd backend/app

# Run the FastAPI server
uvicorn main:app --reload
```

The backend will be available at `http://localhost:8000`.

### 3. Frontend Setup

In a new terminal, set up and run the Streamlit frontend.

```bash
# Navigate to the frontend directory
cd frontend

# (Optional) Create and activate a separate virtual environment
# python -m venv venv
# source venv/bin/activate

# Install frontend dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

The frontend will open in your browser, usually at `http://localhost:8501`.

### 4. Observability Stack Setup (Optional)

To visualize traces and metrics, run the observability services using Docker Compose.

**Prerequisite**: Make sure Docker Desktop is running.

```bash
# In the root directory of Final_project

# Set the OPIK_API_KEY for the Docker environment
# (On Linux/macOS)
export OPIK_API_KEY="your_opik_api_key_from_.env"

# (On Windows PowerShell)
$env:OPIK_API_KEY="your_opik_api_key_from_.env"

# Run the services
docker-compose -f docker-compose.opik.yml up -d
```

The following services will be available:
- **Jaeger UI** (for traces): `http://localhost:16686`
- **Prometheus UI** (for metrics): `http://localhost:9090`

---

## 📂 File Structure

```
Final_project/
├── backend/
│   ├── app/
│   │   ├── main.py         # FastAPI application logic
│   │   ├── config.py       # Pydantic settings and configuration
│   │   ├── services
│   │   │    ├── document_processor.py # Manage type and upload docs
│   │   │    ├── guradrails.py         # To prevent unwanted data to flow
│   │   │    ├── llm_service.py        # Calling all LLm's
│   │   │    ├── memory.py             # To store chat messages
│   │   │    ├── rag_service.py        # Calling all RAG's
│   │   │── models
│   │        ├── models.py   # Basemodel defined 
│   └── requirements.txt    # Backend dependencies
├── frontend/
│   ├── app.py              # Streamlit UI application
│   └── requirements.txt    # Frontend dependencies
├── .env                    # Environment variables (API keys)
├── docker-compose.opik.yml # Docker Compose for observability
└── README.md               # This file
```

---

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.