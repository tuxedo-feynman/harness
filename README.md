# harness

A bare-bones local LLM harness built around a single abstraction: the **Action**.

User input, LLM calls, tool execution, and printing a response are all Actions.
The **Operator** (execution loop + policy controller) receives Action results,
builds context, and decides what runs next — the harness, not the model, is in
control. See `docs/architecture.txt` for the design.

- **Thinking Actions** — AI APIs that make decisions given context (OpenAI, llama-server, a fake for tests)
- **Effect Actions** — anything that touches the outside world (terminal today; telegram, calendar, timers later)
- **Null Action** — terminates the execution loop

Each turn is recorded as a chain of **Operands**: one per policy cycle, each
holding a batch of action requests and their paired results.

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

The default config uses the fake thinking action, so the harness runs without any model server.

## Usage

Single-shot:

```bash
python -m harness.main "Say hello"
```

Interactive chat (history carries across turns for the life of the process — nothing is persisted):

```bash
python -m harness.main --chat
```

Custom config path:

```bash
python -m harness.main --config path/to/config.yaml "Hello"
```

## Thinking actions

Set `thinking_action.type` in `config.yaml`:

| Type | Description |
|------|-------------|
| `fake` | Canned responses, no network (default) |
| `openai` | OpenAI wire format — works with llama-server or the real OpenAI API |

For a local [llama.cpp](https://github.com/ggerganov/llama.cpp) model:

```yaml
thinking_action:
  type: openai
  base_url: http://localhost:8080/v1
  model: local
```

Start the server with:

```bash
llama-server -m path/to/model.gguf --port 8080
```

Requires the optional dependency: `pip install -e ".[openai]"`.

## Logging

Structured logfmt lines (`key=value`) via Python logging. In dev, logs go to
`/tmp/harness.log`:

```bash
tail -f /tmp/harness.log
```

Policy decisions, action executions, and operand creation are all logged.

## Tests

```bash
pytest
```

No network or API key required — the suite uses the fake thinking action throughout.

Optional coverage report:

```bash
pytest --cov=harness --cov=actions
```
