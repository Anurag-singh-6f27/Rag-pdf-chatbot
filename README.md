# Advanced RAG PDF Chatbot

An intelligent PDF question-answering system built using
**Retrieval-Augmented Generation (RAG)**. The application allows users
to upload PDF documents, ask questions in natural language, and receive
context-grounded answers with document and page references.

The project explores how hybrid retrieval, reranking, and context
optimization can improve the relevance of retrieved information and the
reliability of LLM-generated answers.

## Overview

Traditional large language models (LLMs) may generate incorrect answers
when they lack relevant information about a document. This project
addresses that limitation by retrieving relevant passages from uploaded
PDFs and providing them as context to a language model before generating
an answer.

The system combines semantic search and keyword-based retrieval to find
relevant document passages, reranks the retrieved candidates, and
selects a smaller set of relevant context for answer generation.

## Key Features

-   **PDF Question Answering:** Ask natural-language questions about
    uploaded PDF documents.
-   **Multi-PDF Support:** Build a knowledge base from multiple
    documents.
-   **Hybrid Retrieval:** Combine dense vector search using FAISS with
    keyword-based search using BM25.
-   **Semantic Embeddings:** Represent document chunks as vectors using
    Sentence Transformers.
-   **Reranking:** Use FlashRank to improve the ordering of retrieved
    passages by relevance.
-   **Context Optimization:** Reduce irrelevant or redundant context
    before passing it to the LLM.
-   **Source References:** Present document and page information to help
    users verify answers.
-   **Retrieval Transparency:** Inspect intermediate retrieval results
    and relevance scores, where available.
-   **Persistent Indexing:** Reuse saved retrieval indexes instead of
    rebuilding the knowledge base each time, if enabled in the current
    implementation.

## Technology Stack

  Component              Technology
  ---------------------- ------------------------------------------
  Programming Language   Python
  User Interface         Streamlit
  PDF Processing         PyMuPDF
  Text Embeddings        Sentence Transformers
  Embedding Model        `sentence-transformers/all-MiniLM-L6-v2`
  Vector Search          FAISS
  Keyword Search         BM25
  Reranking              FlashRank
  LLM                    Google Gemini API
  Retrieval Approach     Hybrid Retrieval-Augmented Generation

> Verify that each listed technology and feature is present in the
> current implementation before publishing.

## System Architecture

``` text
                 PDF Documents
                       |
                       v
              PDF Text Extraction
                  (PyMuPDF)
                       |
                       v
             Text Cleaning and Chunking
                       |
                       v
                Document Chunks
                       |
                       v
              Embedding Generation
                       |
                       v
              Build Retrieval Indexes
                  /           \
                 v             v
            FAISS Index     BM25 Index
          Semantic Search  Keyword Search
                 \             /
                  v           v
                 Hybrid Retrieval
                       |
                       v
              Merge the Deduplicates
                       |
                       v
                  FlashRank
                   Reranking
                       |
                       v
               Context Optimization
                       |
                       v
             LLM Answer Generation
                       |
                       v
             Answer with Source Details
```

## How It Works

### 1. Document Processing

PDF files are processed using PyMuPDF to extract text page by page. The
extracted text is cleaned and divided into chunks, ideally preserving
page metadata so that answers can be traced back to their source.

### 2. Embedding Generation

Each chunk is converted into a numerical vector using a Sentence
Transformers embedding model. These vectors represent semantic
information and enable similarity-based retrieval.

### 3. Hybrid Retrieval

The system retrieves relevant passages through two complementary
approaches:

-   **FAISS:** Finds semantically similar passages using vector
    similarity.
-   **BM25:** Finds passages that match important query keywords.

The retrieved candidates are combined and deduplicated to create a
broader set of potentially relevant passages.

### 4. Reranking

FlashRank reranks the candidate passages according to their relevance to
the user's query. This helps prioritize useful information before it
reaches the language model.

### 5. Context Optimization

The most relevant passages are selected for the final context. Reducing
irrelevant and redundant information can help control token usage and
improve the quality of the context provided to the LLM.

### 6. Answer Generation

The selected passages and user query are supplied to Google Gemini
through a grounded prompt. The application generates an answer based on
the retrieved document context and presents available source references.

## Installation and Setup

### Prerequisites

-   Python installed on your system
-   Git
-   A Google Gemini API key
-   Internet access for API calls and any initial dependency or model
    downloads

### 1. Clone the Repository

``` bash
git clone https://github.com/Anurag-singh-6f27/Rag-pdf-chatbot.git
cd Rag-pdf-chatbot
```

### 2. Create a Virtual Environment

**Windows (PowerShell):**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux or macOS:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Navigate to the directory containing the project's `requirements.txt`
file, then run:

``` bash
pip install -r requirements.txt
```

If the requirements file is inside a subdirectory, use its path instead,
for example:

``` bash
pip install -r advanced-rag/requirements.txt
```

Use the path that actually exists in the repository.

### 4. Configure the Gemini API Key

Obtain an API key from [Google AI Studio](https://aistudio.google.com/).

If the application reads the key from an environment variable named
`GEMINI_API_KEY`, configure it as follows.

**PowerShell:**

``` powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

**Linux or macOS:**

``` bash
export GEMINI_API_KEY="your_api_key_here"
```

Alternatively, use a `.env` file if the application supports it. Add
`.env` to `.gitignore` and never commit API keys or other secrets. Use
the exact environment variable name expected by your code.

### 5. Run the Application

From the directory containing the Streamlit entry-point file, run:

``` bash
streamlit run app.py
```

Replace `app.py` with the actual entry-point filename if your project
uses a different one. Open the local URL shown in the terminal.

## Usage

1.  Launch the Streamlit application.
2.  Upload one or more PDF documents.
3.  Build or load the document knowledge base, if required by the
    application.
4.  Enter a question about the uploaded content.
5.  Review the generated answer and available source references.
6.  Inspect retrieval details, if supported, to understand how passages
    were selected and reranked.

For best results, ask specific questions that can be answered using the
content of the uploaded documents.

## Research and Evaluation

This project can be used to investigate whether combining retrieval
techniques and context optimization improves document question-answering
quality.

Potential evaluation dimensions include:

-   **Retrieval quality:** Whether relevant passages appear among the
    retrieved results.
-   **Answer faithfulness:** Whether generated answers are supported by
    the supplied context.
-   **Answer relevance:** Whether responses address the user's question.
-   **Context efficiency:** How much irrelevant information is removed
    before generation.
-   **Latency:** How long each pipeline stage takes to produce an
    answer.

A useful experimental comparison includes the following configurations:

  -----------------------------------------------------------------------
  Configuration                       Retrieval Pipeline
  ----------------------------------- -----------------------------------
  Baseline 1                          FAISS-only retrieval

  Baseline 2                          BM25-only retrieval

  Baseline 3                          BM25 + FAISS hybrid retrieval

  Proposed approach                   Hybrid retrieval + FlashRank
                                      reranking + context optimization
  -----------------------------------------------------------------------

Report quantitative improvements only after running experiments and
recording actual results. Possible metrics include Recall@k, Mean
Reciprocal Rank (MRR), answer faithfulness, answer relevance, and
end-to-end latency, provided the evaluation setup supports them.

## Project Structure

The repository's visible top-level structure includes these directories:

``` text
Rag-pdf-chatbot/
├── advanced-rag/       # Advanced RAG implementation
├── output/             # Output artifacts
├── react-app/
│   └── frontend/       # Frontend application
└── .gitignore
```

Update this tree to reflect the exact files, entry points, dependency
files, and modules in the current repository.

## Security Considerations

-   Keep API keys in environment variables or an untracked `.env` file.
-   Do not commit credentials, private documents, or sensitive user
    data.
-   Review generated answers against the cited source passages before
    relying on them.
-   Treat LLM-generated answers as potentially incorrect, even when
    relevant context is retrieved.

## Future Improvements

-   Evaluate retrieval and answer quality using a standardized RAG
    evaluation framework.
-   Add automated tests for document processing, retrieval, and answer
    generation.
-   Optimize pipeline latency through profiling and caching.
-   Compare embedding models and reranking strategies.
-   Improve handling of scanned PDFs, tables, and complex document
    layouts.
-   Add robust unsupported-answer detection and answer verification.

## Author

**Anurag Singh**

GitHub: [Anurag-singh-6f27](https://github.com/Anurag-singh-6f27)

## License

No license is specified in this README. Add a license after deciding how
you want others to use, modify, and distribute the project.
