# Advanced RAG PDF Chatbot

An advanced document question-answering application that combines a
**React frontend** with a **FastAPI backend** and a Retrieval-Augmented
Generation (RAG) pipeline. Users can upload PDF documents, build a
searchable knowledge base, ask questions, and inspect the passages
retrieved to support an answer.

The backend combines **BM25 keyword retrieval**, **FAISS semantic
retrieval**, **hybrid retrieval**, and **FlashRank reranking**. It
returns answers with source document and page information, along with
retrieval diagnostics.

## Features

-   **PDF upload and management** --- upload, select, deselect, and
    remove PDF documents.
-   **Two-step knowledge-base workflow** --- upload documents first,
    then explicitly build the index for selected documents.
-   **Hybrid retrieval** --- combine BM25 keyword search with FAISS
    vector search.
-   **Semantic embeddings** --- embed document chunks for
    similarity-based retrieval.
-   **Reranking** --- use FlashRank to rerank candidate passages before
    answer generation.
-   **Source-aware answers** --- return source document names, page
    numbers, and passage previews.
-   **Retrieval diagnostics** --- expose BM25 hits, semantic hits, merge
    statistics, reranked results, and context statistics through the
    API.
-   **Knowledge-base status** --- report index readiness,
    document/page/chunk counts, and component status.
-   **React frontend** --- provides the user interface; the backend
    exposes a REST API for frontend integration.

## Architecture

### Application Architecture

The React frontend communicates with the FastAPI backend over HTTP. The
backend handles PDF ingestion, knowledge-base construction, retrieval,
reranking, and answer generation.

``` text
React Frontend
      |
      | HTTP requests
      v
FastAPI Backend
      |
      +--> Document upload and management
      +--> Knowledge-base build
      +--> Question answering
      +--> Source references and retrieval diagnostics
```

### Backend Architecture

The backend processes uploaded PDFs, builds both semantic and keyword
indexes, and combines their results to retrieve relevant context for
answering questions.

``` mermaid
flowchart TD
    A[PDF Documents] --> B[PDF Text Extraction<br/>PyMuPDF]
    B --> C[Text Cleaning and Chunking]
    C --> D[Document Chunks]
    D --> E[Embedding Generation]
    E --> F[Build Retrieval Indexes]

    F --> G[FAISS Index<br/>Semantic Search]
    F --> H[BM25 Index<br/>Keyword Search]

    G --> I[Hybrid Retrieval]
    H --> I
    I --> J[Merge and Deduplicate]
    J --> K[FlashRank Reranking]
    K --> L[Context Optimization]
    L --> M[LLM Answer Generation]
    M --> N[Answer with Source Details]
```

#### Pipeline Stages

1.  **PDF text extraction:** PyMuPDF extracts text and page information
    from uploaded PDFs.
2.  **Text cleaning and chunking:** Extracted text is cleaned and
    divided into manageable chunks.
3.  **Embedding generation:** A sentence-transformer model converts
    chunks into vectors.
4.  **Build retrieval indexes:** The backend creates a FAISS vector
    index and a BM25 keyword index.
5.  **Hybrid retrieval:** Semantic search and keyword search contribute
    candidate passages.
6.  **Merge and deduplicate:** Candidate results are combined and
    duplicate chunks are removed.
7.  **FlashRank reranking:** Candidate passages are reranked by
    relevance when the reranker is available.
8.  **Context optimization:** The highest-ranked passages are selected
    for the LLM context.
9.  **LLM answer generation:** The configured answer-generation provider
    uses the question and selected context to generate a response.
10. **Answer with source details:** The API returns the answer, source
    document/page information, and retrieval diagnostics.

## Technology Stack

  Layer               Technology
  ------------------- -------------------------------------------
  Frontend            React
  Backend API         FastAPI
  API server          Uvicorn
  Language            Python
  PDF extraction      PyMuPDF
  Text embeddings     Sentence Transformers
  Vector retrieval    FAISS
  Keyword retrieval   BM25
  Reranking           FlashRank
  Answer generation   LLM integration configured by the backend
  Configuration       Python settings / environment variables

The embedding model is configured through `settings.EMBEDDING_MODEL`.
Check `config.py` for the model name and other runtime settings.

## Repository Structure

``` text
Rag-pdf-chatbot/
├── advanced-rag/
│   ├── app.py                 # FastAPI application entry point
│   ├── config.py              # Backend settings (if present here)
│   ├── src/
│   │   ├── evaluation/        # Evaluation utilities
│   │   ├── generation/        # Answer generation
│   │   ├── ingestion/         # PDF extraction, cleaning, chunking
│   │   ├── reranking/         # FlashRank reranker
│   │   ├── retrieval/          # BM25, embeddings, hybrid and vector search
│   │   └── utils/              # Shared utilities
│   └── ...
├── react-app/
│   └── frontend/              # React frontend
├── output/                    # Output artifacts
└── README.md
```

This is a representative structure based on the backend modules. Adjust
filenames or paths if your checked-out repository differs.

## Prerequisites

-   Python 3.10 or newer (use the version supported by your
    dependencies)
-   Node.js and npm for running the React frontend
-   Git
-   An API key for the answer-generation provider configured by your
    backend, if generation is enabled

## Setup

### 1. Clone the repository

``` bash
git clone https://github.com/Anurag-singh-6f27/Rag-pdf-chatbot.git
cd Rag-pdf-chatbot
```

### 2. Set up the backend

Navigate to the backend directory:

``` bash
cd advanced-rag
```

Create and activate a virtual environment.

**Windows PowerShell:**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux or macOS:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the backend dependencies. If the backend has a
`requirements.txt` file, use it:

``` bash
pip install -r requirements.txt
```

If no requirements file is present, install the API server dependencies
and then install the remaining libraries imported by the project:

``` bash
pip install fastapi "uvicorn[standard]" python-multipart
```

The RAG pipeline also depends on packages used by the modules in `src/`,
such as the PDF-processing, embedding, vector-search, BM25, and
reranking libraries. Install the versions specified by your project
dependency file when available.

### 3. Configure environment variables

The backend's fallback response indicates that answer generation is
enabled by setting `GROQ_API_KEY` in a `.env` file. If you use Groq,
create an `.env` file in the backend's expected working directory:

``` env
GROQ_API_KEY=your_groq_api_key_here
```

Confirm the required variable names and configuration path in
`src/generation/answer_generator.py` and `config.py`. Never commit
`.env` or API keys to Git.

If generation is not configured, the backend can return relevant
retrieved passages instead of a generated answer.

### 4. Run the backend

From the `advanced-rag` directory, run:

``` bash
uvicorn app:app --reload
```

Uvicorn starts the FastAPI application. Keep this terminal running while
using the frontend.

You can open the interactive API documentation at:

``` text
http://127.0.0.1:8000/docs
```

> The command above assumes `app.py` is directly inside `advanced-rag/`
> and imports resolve from that directory.

### 5. Run the React frontend

Open a second terminal and navigate to the frontend directory from the
repository root:

``` bash
cd react-app/frontend
```

Install dependencies and start the development server:

``` bash
npm install
npm run dev
```

Use the local URL printed by the frontend development server. Make sure
the frontend's API base URL points to the running FastAPI backend
(usually `http://127.0.0.1:8000` during local development).

If the frontend uses a different package manager or start script, follow
its `package.json` scripts.

## API Endpoints

The backend exposes the following routes:

  ----------------------------------------------------------------------------
  Method               Endpoint                           Purpose
  -------------------- ---------------------------------- --------------------
  `GET`                `/api/documents`                   List uploaded
                                                          documents

  `POST`               `/api/documents/upload`            Upload PDF files

  `DELETE`             `/api/documents/{doc_id}`          Remove a document

  `POST`               `/api/documents/{doc_id}/toggle`   Toggle a document's
                                                          selection

  `POST`               `/api/documents/toggle-all`        Select or deselect
                                                          all documents

  `GET`                `/api/kb/status`                   Get knowledge-base
                                                          and component status

  `POST`               `/api/kb/build`                    Build indexes for
                                                          selected documents

  `POST`               `/api/chat/ask`                    Retrieve context and
                                                          answer a question
  ----------------------------------------------------------------------------

Open `http://127.0.0.1:8000/docs` while the backend is running to
inspect request schemas and try the endpoints.

### Example: Ask a question

`POST /api/chat/ask`

``` json
{
  "query": "What are the main findings in this document?",
  "selected_doc_ids": []
}
```

`selected_doc_ids` can be used to restrict retrieval to a subset of
indexed documents. Check the API documentation for the exact response
schema.

## How to Use

1.  Start the FastAPI backend.
2.  Start the React frontend.
3.  Upload one or more PDF files.
4.  Select the documents you want included in the knowledge base.
5.  Choose **Build Knowledge Base** to extract chunks, generate
    embeddings, and build the BM25 and FAISS indexes.
6.  Ask a question in the chat interface.
7.  Review the answer, source document/page references, and retrieval
    details.

**Important:** Uploading a PDF does not automatically index it. Build
the knowledge base after selecting documents before asking questions.

## Research and Evaluation

The architecture supports investigation of how retrieval and reranking
affect document question-answering. Possible configurations to compare
include:

  ---------------------------------------------------------------------
  Configuration                      Pipeline
  ---------------------------------- ----------------------------------
  Baseline 1                         FAISS semantic retrieval

  Baseline 2                         BM25 keyword retrieval

  Baseline 3                         Hybrid BM25 + FAISS retrieval

  Proposed pipeline                  Hybrid retrieval + FlashRank
                                     reranking + selected context
  ---------------------------------------------------------------------

Potential metrics include Recall@k, Mean Reciprocal Rank (MRR), answer
faithfulness, answer relevance, and end-to-end latency, depending on the
evaluation implementation.

Do not claim performance improvements until experiments have been run
and results recorded. The `src/evaluation/` directory contains
evaluation-related modules; inspect the current implementation for
supported metrics and commands.

## Notes and Limitations

-   Uploaded-document and knowledge-base state is held in application
    memory in the shown backend implementation. Restarting the backend
    can clear the in-memory document registry and indexes; uploaded PDF
    files may remain on disk.
-   The backend attempts to initialize FlashRank at startup. If it is
    unavailable, reranking may be skipped and hybrid retrieval order
    used instead.
-   If the answer-generation API is not configured, the backend returns
    retrieved passages as a fallback.
-   The `groundedness` field is a retrieval-score proxy in this
    implementation, not a verified measure of answer faithfulness or
    hallucination.
-   Do not upload confidential documents unless your data-handling
    requirements and LLM provider configuration permit it.

## Future Improvements

-   Add automated tests for ingestion, retrieval, API endpoints, and
    answer generation.
-   Profile each pipeline stage and optimize latency.
-   Evaluate retrieval quality and answer faithfulness on a documented
    dataset.
-   Add persistent storage for document metadata and indexes if needed.
-   Improve handling of scanned PDFs, tables, and complex layouts.
-   Add robust checks for questions that cannot be answered from the
    uploaded documents.

