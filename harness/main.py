import sys

from actions.fake_thinking_action import FakeThinkingAction
from actions.null_action import NullAction
from actions.terminal_action import TerminalAction
from harness.action import Action
from harness.action_directory import ActionDirectory
from harness.cli import parse_args
from harness.config import Config, load_config
from harness.context import ContextBuilder
from harness.logger import new_id, setup_logging
from harness.loop import ExecutionLoop
from harness.models import ActionDescription, ActionResult
from harness.policy import PolicyController


def _build_thinking_action(config: Config) -> Action:
    if config.thinking_action.type == "fake":
        return FakeThinkingAction()
    if config.thinking_action.type == "openai":
        from actions.openai_thinking_action import OpenAIThinkingAction

        return OpenAIThinkingAction(config.thinking_action)
    raise ValueError(f"Unknown thinking action type: {config.thinking_action.type!r}")


def _seed_user_input(context_builder: ContextBuilder, text: str) -> None:
    """Record user input as the result of a terminal.read action — input is
    just another Action's result, same shape as everything else."""
    context_builder.add_operand(
        action_requests=[
            ActionDescription(id=new_id(), action_name="terminal", method_name="read")
        ],
        action_results=[ActionResult(contents=text)],
    )


def main(argv=None) -> None:
    args = parse_args(argv)

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    setup_logging(config.logging)

    thinking = _build_thinking_action(config)
    directory = ActionDirectory([NullAction(), TerminalAction(), thinking])
    context_builder = ContextBuilder(config.system_prompt, directory)
    policy = PolicyController(directory, thinking_action_name=thinking.name)
    loop = ExecutionLoop(directory, policy, context_builder)

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
            _seed_user_input(context_builder, user_input)
            loop.run()
    elif args.prompt:
        _seed_user_input(context_builder, args.prompt)
        loop.run()
    else:
        print('Usage: python -m harness.main "Your message"', file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
