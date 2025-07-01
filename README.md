# Marketing Agent

A FastAPI-based marketing research assistant that leverages Retrieval-Augmented Generation (RAG) from blog articles, web search, and Wikipedia to provide actionable marketing insights and suggestions.

## Features

- Synthesizes insights from ingested marketing blog articles.
- Provides actionable marketing strategies, ad copy, and campaign improvements.
- Uses web search and Wikipedia as additional tools.
- Exposes a REST API for integration.

## Project Structure

```
Marketing_Agent/
  ├── app.py                  # FastAPI app entrypoint
  ├── MarketingAgent.py       # Core agent logic (class: marketing_agent)
  ├── MarketingAgent_tools.py # Tool definitions and RAG setup
  ├── blog_corpus/            # Directory for blog .txt/.md files to ingest
  ├── chroma_db/              # Vector DB for RAG (auto-generated)
  └── reports/                # File management toolkit root
```

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Marketing_Agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare blog data

- Place your `.txt` or `.md` blog files inside the `blog_corpus/` directory.

### 4. Set environment variables 

- You may want a `.env` file for API keys (e.g., OpenAI, Serper, etc.) and to configure `FILE_TOOL_ROOT` if you want to change the file management root directory.

Example `.env`:
```
OPENAI_API_KEY=your-openai-key
SERPER_API_KEY=your-serper-key
FILE_TOOL_ROOT=reports
```

## Running the API

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## API Usage

### Endpoint: `/run-agent` (POST)

**Request Body:**
```json
{
  "message": "How can I improve my social media marketing in 2025?",
  "success_criteria": "Provide actionable, blog-based suggestions"
}
```

**Response:**
```json
{
  "result": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "assistant", "content": "Evaluator Feedback on this answer: ..."}
  ]
}
```

## Notes

- On startup, the agent ingests all blog files in `blog_corpus/` for RAG.
- The agent uses tools for file management, web search, and Wikipedia queries.
- The assistant will cite sources when answers are based on blog content or external tools.

## Extending

- Add more tools in `MarketingAgent_tools.py`.
- Add more blog content to `blog_corpus/` and restart the server to re-ingest. 