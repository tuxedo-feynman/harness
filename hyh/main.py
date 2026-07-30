import sys

from hyh.action import INPUT_METHOD
from hyh.action_directory import ActionDirectory
from hyh.cli import parse_args
from hyh.config import build_actions, load_config
from hyh.context import ContextBuilder
from hyh.dispatcher import Dispatcher
from hyh.logger import new_id, setup_logging
from hyh.loop import ExecutionLoop
from hyh.models import ActionDescription, ActionResult
from hyh.policy import PolicyController


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
        # already-resolved terminal input stimulus and run one turn. The
        # input request Policy parks at turn end never resolves — the
        # process exits.
        stimulus = context_builder.add_operand(
            parents=[root],
            order=0,
            action_request=ActionDescription(
                id=new_id(), action_name="terminal", method_name=INPUT_METHOD
            ),
            action_result=ActionResult(contents=args.prompt),
        )
        loop.run(stimulus)
    else:
        print('Usage: python -m hyh.main "Your message"', file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
