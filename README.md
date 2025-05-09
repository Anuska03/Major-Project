# **Retrieval-Augmented Generation (RAG) Project**

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system for processing user queries using document content (PDFs, CSVs) and structured data. It leverages **docling**, **FAISS**, **BM25**, and **DPR** for efficient document retrieval,  **DuckDB** and **pandas** for efficient csv parsing and **GPT-4** along with **COT prompting** for response generation.

---

## Setup Instructions

### **Requirements**

- Python 3.8+

Install dependencies:

```bash
pip install -r requirements.txt
````

---

### **Steps to Run the Project**

1. **Clone the Repo**:

   ```bash
   git clone https://github.com/Anuska03/Major-Project.git
   cd Major-Project
   ```

2. **Create a `.env` file** in the root directory with the following content:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   HUGGINGFACE_HUB_TOKEN=your_huggingface_api_key
   ```

   Replace the placeholders with your actual API keys from **OpenAI** and **HuggingFace**.

3. **Run the Streamlit App**:

   ```bash
   streamlit run app.py
   ```
---

## How It Works

* **File Upload**: Upload PDF or CSV files.
* **Ingestion**: The system processes the uploaded files, extracting text and metadata.
* **Query Processing**: Enter queries to retrieve relevant information from the ingested data.
* **Response Generation**: **GPT-4** generates answers based on the retrieved content.

---

## Project Structure

* `app.py`: Main Streamlit app.
* `csv_ingest.py`: Handles CSV ingestion.
* `pdf_ingest.py`: Parses and chunks PDFs.
* `pdf_agent.py`: Manages PDF retrieval and image interpretation.
* `sql_agent.py`: Converts queries into SQL.
* `views.py`: Manages the LangGraph workflow.

---

## Troubleshooting

* **Missing `.env` file**: Make sure to create it as mentioned above.
* **ModuleNotFoundError**: Install all dependencies via:

  ```bash
  pip install -r requirements.txt
  ```




