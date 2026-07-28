import sys

from harness.action import LISTEN_METHOD
from harness.action_directory import ActionDirectory
from harness.cli import parse_args
from harness.config import build_actions, load_config
from harness.context import ContextBuilder
from harness.dispatcher import Dispatcher
from harness.logger import new_id, setup_logging
from harness.loop import ExecutionLoop
from harness.models import ActionDescription, ActionResult
from harness.policy import PolicyController


def main(argv=None) -> None:
    args = parse_args(argv)

    config = load_config(args.config)
    setup_logging(config.logging)

    actions = build_actions(config)
    directory = ActionDirectory(actions)
    context_builder = ContextBuilder(config.system_prompt, directory)

    default_thinking = directory.get(config.default_thinking_action)
    policy = PolicyController(
        directory, context_builder, thinking_action_name=default_thinking.name
    )
    loop = ExecutionLoop(directory, policy, context_builder)
    root = context_builder.add_root()

    if args.chat:
        print("Chat mode (type 'quit' to exit)")
        Dispatcher(directory, context_builder, loop).serve(root)
    elif args.prompt:
        # Single-shot: no listener threads. Seed the prompt as an
        # already-resolved terminal listen stimulus and run one turn. The
        # listener Policy parks at turn end never resolves — the process exits.
        stimulus = context_builder.add_operand(
            parent=root,
            action_requests=[
                ActionDescription(id=new_id(), action_name="terminal", method_name=LISTEN_METHOD)
            ],
            action_results=[ActionResult(contents=args.prompt)],
        )
        loop.run(stimulus)
    else:
        print('Usage: python -m harness.main "Your message"', file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
