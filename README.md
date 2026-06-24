# harness

A bare-bones local LLM harness. CLI-driven agent loop with session persistence, a tool registry, and swappable model providers.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Config** — `config.yaml` is gitignored (it may contain API keys). Copy the example to get started:

```bash
cp config.example.yaml config.yaml
```

The default config uses a fake provider so you can run the harness without any model server.

## Usage

Single-shot:

```bash
python -m harness.main "Say hello"
```

Interactive chat:

```bash
python -m harness.main --chat
```

Named session:

```bash
python -m harness.main --session my-session "What did we talk about?"
```

Custom config path:

```bash
python -m harness.main --config path/to/config.yaml "Hello"
```

## Providers

Set `provider.type` in `config.yaml`:

| Type | Description |
|------|-------------|
| `fake` | Pre-programmed responses, no network (default) |
| `openai_compat` | OpenAI wire format — works with llama-server or the real OpenAI API |

For a local [llama.cpp](https://github.com/ggerganov/llama.cpp) model:

```yaml
provider:
  type: openai_compat
  base_url: http://localhost:8080/v1
  model: local
```

Start the server with:

```bash
llama-server -m path/to/model.gguf --port 8080
```

## Session data

Turns are persisted as JSONL under `data/sessions/<session-id>.jsonl`. The `data/` directory is gitignored.

## Tests

```bash
pytest
```

No network or API key required — the test suite uses a fake provider throughout.

Optional coverage report:

```bash
pytest --cov=harness --cov=providers --cov=tools --cov=storage
```
