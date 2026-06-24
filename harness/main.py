import sys
from pathlib import Path

from harness.cli import parse_args
from harness.config import Config, load_config
from harness.context import ContextBuilder
from harness.loop import HarnessLoop
from harness.messages import AssistantTurn
from harness.sessions import SessionManager
from providers.fake_provider import FakeProvider
from storage.jsonl_store import JsonlSessionStore
from tools.echo_tool import EchoTool
from tools.registry import ToolRegistry


def _build_provider(config: Config):
    if config.provider.type == "fake":
        return FakeProvider([
            AssistantTurn(content="Hello! How can I help you?", tool_calls=[]),
        ])
    if config.provider.type == "openai_compat":
        from providers.openai_compat_provider import OpenAICompatProvider
        return OpenAICompatProvider(config.provider)
    raise ValueError(f"Unknown provider type: {config.provider.type!r}")


def _build_registry(config: Config) -> ToolRegistry:
    registry = ToolRegistry()
    tools_cfg = config.tools or {}
    if tools_cfg.get("echo", {}).get("enabled", True):
        registry.register(EchoTool())
    return registry


def main(argv=None) -> None:
    args = parse_args(argv)

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    session_id = args.session or config.default_session_id
    workspace_root = Path("workspace")
    workspace_root.mkdir(exist_ok=True)

    store = JsonlSessionStore(data_dir=Path(config.data_dir) / "sessions")
    session = SessionManager(store)
    context_builder = ContextBuilder()
    provider = _build_provider(config)
    registry = _build_registry(config)

    loop = HarnessLoop(
        config=config,
        session=session,
        context_builder=context_builder,
        provider=provider,
        tool_registry=registry,
        workspace_root=workspace_root,
    )

    if args.chat:
        print("Chat mode (type 'quit' to exit)")
        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue
            response = loop.run_turn(session_id, user_input)
            print(response)
    elif args.prompt:
        response = loop.run_turn(session_id, args.prompt)
        print(response)
    else:
        print('Usage: python -m harness.main "Your message"', file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
