# hyh

A bare-bones local LLM harness built around a single abstraction: the **Action**.

User input, LLM calls, tool execution, and printing a response are all Actions.
The **Operator** (execution loop + policy controller) receives Action results,
builds context, and decides what runs next — hyh, not the model, is in
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

The default config uses the fake thinking action, so hyh runs without any model server.

## Usage

Single-shot:

```bash
python -m hyh.main "Say hello"
```

Interactive chat (history carries across turns for the life of the process — nothing is persisted):

```bash
python -m hyh.main --chat
```

Custom config path:

```bash
python -m hyh.main --config path/to/config.yaml "Hello"
```

## Actions

Actions are registered from the `actions` section of `config.yaml`; each entry
gets its own config block. The null action is always present. Multiple thinking
actions can be registered at once — `default_thinking_action` names the one the
policy controller routes to.

| Action | Description |
|--------|-------------|
| `terminal` | Prints to / reads from the local terminal |
| `telegram` | Telegram bot channel — send and receive via the Bot API (no extra dependency) |
| `fake` | Canned thinking action, no network (default) |
| `openai` | OpenAI wire format — works with llama-server or the real OpenAI API |
| `claude` | Anthropic Claude Messages API (`pip install -e ".[claude]"`, `ANTHROPIC_API_KEY` in env) |

## Channels

In `--chat` mode every action with a listening method (`terminal.read`, `telegram.receive`)
gets a listener: a pending operand that resolves when a message arrives on that
channel. All channels share one history — a conversation started on the terminal
can be continued from Telegram and vice versa. `quit` closes the channel it was
typed on; the process exits when no channels are listening.

To enable Telegram, create a bot with [@BotFather](https://t.me/BotFather) and add
to `config.yaml` (gitignored — tokens are safe there):

```yaml
actions:
  terminal:
  fake:
  telegram:
    token: "123456:your-bot-token"
    # chat_id: 123456789   # optional: pin to one chat
```

For a local [llama.cpp](https://github.com/ggerganov/llama.cpp) model:

```yaml
actions:
  terminal:
  openai:
    base_url: http://localhost:8080/v1
    model: local

default_thinking_action: openai
```

Start the server with:

```bash
llama-server -m path/to/model.gguf --port 8080
```

Requires the optional dependency: `pip install -e ".[openai]"`.

## Logging

Structured logfmt lines (`key=value`) via Python logging. In dev, logs go to
`/tmp/hyh.log`:

```bash
tail -f /tmp/hyh.log
```

Policy decisions, action executions, and operand creation are all logged.

## Tests

```bash
pytest
```

No network or API key required — the suite uses the fake thinking action throughout.

Optional coverage report:

```bash
pytest --cov=hyh --cov=actions
```
