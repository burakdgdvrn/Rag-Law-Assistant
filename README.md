# ⚖️ Local RAG Law Assistant (Turkish Code of Obligations)

An offline, privacy-first legal AI assistant designed to answer questions regarding **Turkish Rental Law**, utilizing **Retrieval-Augmented Generation (RAG)** architecture.

This project leverages **Llama 3.2 (3B)** via **Ollama** to provide legally grounded answers without relying on external cloud APIs, ensuring data privacy and low latency.

## 🎯 Project Scope
* **Current Domain:** Turkish Code of Obligations (TBK) - **Rental Law (Articles 299-356)**.
* **Architecture:** Fully local RAG pipeline (No OpenAI/Cloud dependence).
* **Status:** Prototype / Active Development.

## ⚙️ Tech Stack
* **LLM:** Llama 3.2:3b (Running locally via Ollama)
* **Orchestration:** LangChain / Python
* **Vector Database:** ChromaDB / FAISS
* **Embeddings:** HuggingFace (Multilingual)
* **Data Source:** Official Turkish Legal Texts (TBK)

## 🧪 Example Scenario

 **User Query**: What can the landlord do if the tenant fails to pay?

 **System Output**:
"Based on TBK Article 315: If the tenant fails to pay the rent, the landlord may give a written deadline (min. 30 days for residential properties). If payment is not made within this period, the landlord is entitled to terminate the contract..."

## 🔄 How It Works
The system does not just "generate" text; it retrieves actual legal articles before answering to minimize hallucinations.

`User Query` ➔ `Vector Search (Semantic)` ➔ `Retrieve Relevant Articles` ➔ `Llama 3.2 Generation` ➔ `Answer`

## 🚀 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/burakdgdvrn/Rag-Law-Assistant.git](https://github.com/burakdgdvrn/Rag-Law-Assistant.git)
    cd Rag-Law-Assistant
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Local Model:**
    Ensure [Ollama](https://ollama.com/) is installed and running.
    ```bash
    ollama pull llama3.2:3b
    ```

4.  **Run the application:**
    ```bash
    python model_run.py
    ```
## 🗺️ Roadmap
* **[x] Phase 1**: Core RAG pipeline setup with LangChain & Ollama.

* **[x] Phase 2**: Ingestion of Rental Law module (TBK 299-356).

* **[ ] Phase 3**: Expansion to Family Law and General Provisions.

* **[ ] Phase 4**: UI Development (Streamlit/Gradio interface).

* **[ ] Phase 5**: Evaluation metrics for citation accuracy.
## ⚠️ Disclaimer
This project is for educational and experimental purposes only. The generated outputs are not professional legal advice.

### 👨‍💻 Developed by **Burak Dağdeviren**
