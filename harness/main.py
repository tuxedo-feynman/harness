import logging
import sys

from harness.action_directory import ActionDirectory
from harness.cli import parse_args
from harness.config import build_actions, load_config
from harness.context import ContextBuilder
from harness.dispatcher import Dispatcher
from harness.logger import new_id, setup_logging
from harness.loop import ExecutionLoop
from harness.models import ActionDescription, ActionResult
from harness.policy import PolicyController

# Explicit name: __name__ is "__main__" under python -m, which would fall
# outside the "harness" logger hierarchy.
log = logging.getLogger("harness.main")


def main(argv=None) -> None:
    args = parse_args(argv)

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    setup_logging(config.logging)

    actions = build_actions(config)
    directory = ActionDirectory(actions)
    context_builder = ContextBuilder(config.system_prompt, directory)

    default_thinking = directory.get(config.default_thinking_action)
    policy = PolicyController(
        directory, context_builder, thinking_action_name=default_thinking.name
    )
    loop = ExecutionLoop(directory, policy, context_builder)

    if args.chat:
        print("Chat mode (type 'quit' to exit)")
        Dispatcher(directory, context_builder, loop).serve()
    elif args.prompt:
        # Single-shot: no listener threads. Seed the prompt as an
        # already-resolved terminal.read stimulus and run one turn. The
        # listener Policy parks at turn end never resolves — the process exits.
        root = context_builder.add_root()
        stimulus = context_builder.add_operand(
            parent=root,
            action_requests=[
                ActionDescription(id=new_id(), action_name="terminal", method_name="read")
            ],
            action_results=[ActionResult(contents=args.prompt)],
        )
        try:
            loop.run(stimulus)
        except Exception as e:
            log.exception(f"system_failure error={e!r}")
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print('Usage: python -m harness.main "Your message"', file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
