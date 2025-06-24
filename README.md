---
title: AgentFlow-Multi-Agent_Workflow_Orchestration_Assistant
app_file: app.py
sdk: gradio
sdk_version: 5.33.0
---
# AgentFlow

AgentFlow is a multi-agent workflow orchestration assistant built with LangChain, Gradio, and modern LLMs. It supports tool-augmented reasoning, Retrieval-Augmented Generation (RAG) from uploaded PDFs, and Google Calendar integration.

## Features

- **Multi-Agent Orchestration:** Modular agents for planning, research, coding, and evaluation.
- **Tool Augmentation:** Use web search, Wikipedia, file management, Python REPL, and more.
- **Retrieval-Augmented Generation (RAG):**
  - Upload PDF files and ask questions about their content.
  - Uses FAISS vector database for semantic search and context injection.
- **Google Calendar Integration:**
  - Create and list calendar events using natural language.
- **Detailed Reports & Summaries:**
  - Generate detailed reports, summaries, and answers based on user queries, PDF content, and research tools.
- **Gradio Web UI:**
  - Chat interface, PDF upload, and event management in your browser.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd AgentFlow
```

### 2. Install Dependencies
Make sure you have Python 3.10+ and pip installed.
```bash
pip install -r requirements.txt
```

### 3. (Optional) Set Up Google Calendar API
- Go to [Google Cloud Console](https://console.cloud.google.com/)
- Create a project, enable the Google Calendar API
- Create OAuth credentials (Desktop app), download `client_secret.json`
- Place `client_secret.json` in the project root
- The first time you use calendar features, you will be prompted to authenticate and a `token.json` will be created
- Set these in your `.env` (or use defaults):
  ```
  GOOGLE_TOKEN_PATH=token.json
  GOOGLE_CALENDAR_ID=primary
  ```

### 4. Run the App
```bash
python app.py
```
The app will open in your browser.

## .env Configuration

Create a `.env` file in the project root to configure environment variables. Example:

```
# Google Calendar API
GOOGLE_TOKEN_PATH=token.json         # Path to your Google OAuth token
GOOGLE_CALENDAR_ID=primary          # Calendar ID (usually 'primary')

# OpenAI API (required for LLMs)
OPENAI_API_KEY=your_openai_api_key

# Serper API (for web search tool)
SERPER_API_KEY=your_serper_api_key

# LangSmith (LangChain analytics and tracing, optional)
LANGCHAIN_API_KEY=your_langsmith_api_key

# File Management Tool
FILE_TOOL_ROOT=reports              # Root directory for file management tools

# Pushover Notifications (optional)
PUSHOVER_TOKEN=your_pushover_token
PUSHOVER_USER=your_pushover_user

# Other environment variables can be added as needed for custom tools
```

- If you use Google Calendar, set up the first two variables.
- For LLM and agent features, set your OpenAI API key.
- For web search, set your Serper API key.
- For LangSmith analytics/tracing, set your LangSmith API key.
- If you use file management tools, set `FILE_TOOL_ROOT`.
- For push notifications, set the Pushover variables.

## Usage

### PDF RAG
- Upload a PDF using the UI.
- Ask questions about its content (e.g., "What is my name?").
- The assistant will cite the PDF as the source when using its content.

### Detailed Reports & Summaries
- Ask the assistant to generate detailed reports or summaries on topics, documents, or research queries.
- The assistant leverages multi-agent reasoning, RAG, and research tools to produce comprehensive outputs.

### Google Calendar
- Ask to create or list calendar events (e.g., "Create a meeting tomorrow at 10am.", "List my next 3 events.")

### General Chat & Tools
- Use the chat for planning, research, coding, and more.
- The assistant can use web search, Wikipedia, Python REPL, and file management tools.

## Folder Structure
- `app.py` — Gradio UI and app entry point
- `agentflow.py` — Main agent orchestration logic
- `agentflow_tools.py` — Tool definitions (calendar, RAG, web, etc.)
