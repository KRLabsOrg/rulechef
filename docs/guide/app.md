# Web App

A web UI for RuleChef — upload data, learn rules via LLM, see highlighted entities, and correct mistakes to improve rules.

## Prerequisites

- Python 3.10+
- Node.js 20+
- An OpenAI-compatible API key (Groq, OpenAI, Together, etc.)

## Setup

### Backend

```bash
pip install -e ".[app]"
export OPENAI_API_KEY=gsk_...   # your API key
uvicorn api.main:app --reload --port 8000
```

The backend defaults to **Kimi K2** (`moonshotai/kimi-k2-instruct-0905`) via the Groq API. To use a different provider:

```bash
export OPENAI_BASE_URL=https://api.openai.com/v1/   # or any OpenAI-compatible API
export OPENAI_MODEL=gpt-4o-mini
```

If you're developing the `rulechef` library locally alongside the app:

```bash
pip install -e /path/to/rulechef
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). The dev server proxies `/api` requests to the backend on port 8000.

## Usage

1. **Data page** — Set entity labels, then add training examples by pasting text and highlighting spans with the annotation tool. Or bulk upload CSV/JSON.
2. **Learn page** — Click "Learn Rules" to trigger LLM-powered rule learning. Watch metrics and inspect learned rules.
3. **Extract page** — Enter text, run extraction, review highlighted entities. Select text to add entities, click highlights to change or remove them. Submit corrections to feed back into learning.

A default NER project is created automatically — no configuration step needed.

### Multi-user sessions

Each browser tab gets its own isolated session. Multiple people can use the app simultaneously without interfering with each other. Refreshing the page starts a fresh session. Sessions are automatically cleaned up after 1 hour of inactivity.

## Production Build

Build the frontend and serve everything from FastAPI:

```bash
cd frontend && npm run build && cd ..
uvicorn api.main:app --host 0.0.0.0 --port 8080
```

## Project Structure

```
api/                    FastAPI backend
  main.py               App entry, CORS, static files
  config.py             Settings from env vars
  state.py              Per-session RuleChef instances
  schemas.py            Pydantic request/response models
  tasks.py              Background learning runner
  routes/
    project.py          Configure task, get status, default project
    data.py             Upload, add examples/corrections
    learning.py         Trigger learning, poll status
    extraction.py       Run extraction
    rules.py            List/delete rules
frontend/               Vite + React + TypeScript + shadcn/ui
  src/
    api/                Fetch wrapper + React Query hooks
    pages/              DataPage, LearnPage, ExtractPage
    components/
      layout/           AppShell with top nav
      data/             FileUpload, ExampleTable
      learning/         LearnButton, RulesTable, MetricsCard
      extraction/       AnnotatedText, EntityLegend, LabelPicker, CorrectionToolbar
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/project/configure` | Create task + RuleChef instance |
| GET | `/api/project/status` | Current state + stats |
| POST | `/api/data/upload` | Upload CSV/JSON training data |
| POST | `/api/data/example` | Add single example |
| POST | `/api/data/correction` | Submit user correction |
| GET | `/api/data/examples` | List examples + corrections |
| POST | `/api/learn` | Trigger learning (background) |
| GET | `/api/learn/status` | Poll learning progress |
| POST | `/api/extract` | Run extraction on input |
| POST | `/api/extract/batch` | Batch extraction |
| GET | `/api/rules` | List learned rules |
| DELETE | `/api/rules/{id}` | Delete a rule |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | API key (Groq, OpenAI, or any compatible provider) |
| `OPENAI_BASE_URL` | `https://api.groq.com/openai/v1/` | LLM API base URL |
| `OPENAI_MODEL` | `moonshotai/kimi-k2-instruct-0905` | Model name |
| `RULECHEF_STORAGE_PATH` | `./rulechef_data` | Directory for persisted datasets |
| `SESSION_TTL_SECONDS` | `3600` | Session expiry (seconds of inactivity) |
| `SESSION_SECRET` | `dev-insecure-...` | Secret for signing session cookies (set in production!) |
