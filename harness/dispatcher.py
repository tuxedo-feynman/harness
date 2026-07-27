import logging
import queue
import sys
import threading

from harness.action import Action
from harness.action_directory import ActionDirectory
from harness.context import ContextBuilder
from harness.logger import new_id
from harness.loop import ExecutionLoop
from harness.models import ActionDescription, ActionResult, Context

log = logging.getLogger(__name__)


class Dispatcher:
    """The stimulus boundary: one daemon thread per listening channel blocks on
    the channel and pushes results onto a queue. Threads know nothing about
    operands — attribution to a pending listener happens at dequeue time, on
    this (main) thread, which owns all operand mutation.
    """

    def __init__(
        self,
        action_directory: ActionDirectory,
        context_builder: ContextBuilder,
        loop: ExecutionLoop,
    ):
        self.action_directory = action_directory
        self.context_builder = context_builder
        self.loop = loop
        self._queue: queue.Queue[tuple[str, ActionResult | None]] = queue.Queue()
        self._closed: set[str] = set()  # channels whose transport has disconnected

    def _listen_methods(self) -> dict[str, str]:
        """channel action name -> its listen method name."""
        channels = {}
        for name, available in self.action_directory.method_index().items():
            for method in available.methods.values():
                if method.listen:
                    channels[name] = method.name
        return channels

    def arm_initial(self) -> None:
        """Create the root and one pending listener operand per channel."""
        root = self.context_builder.add_root()
        for channel, method_name in self._listen_methods().items():
            operand = self.context_builder.add_operand(
                parent=root,
                action_requests=[
                    ActionDescription(id=new_id(), action_name=channel, method_name=method_name)
                ],
            )
            self.context_builder.park_listener(channel, operand)

    def serve(self) -> None:
        self.arm_initial()
        for channel, method_name in self._listen_methods().items():
            threading.Thread(
                target=self._listen,
                args=(self.action_directory.get(channel), method_name),
                daemon=True,
            ).start()
        try:
            while self.context_builder.listener_channels() - self._closed:
                channel, result = self._queue.get()
                if result is None:
                    self._closed.add(channel)
                    log.info(f"channel_closed channel={channel}")
                    continue
                self.handle_stimulus(channel, result)
        except KeyboardInterrupt:
            pass

    def handle_stimulus(self, channel: str, result: ActionResult) -> None:
        """Resolve the channel's pending listener with the result and run the
        turn. Turn failures are system failures: reported to stderr, and the
        channel is re-armed by the dispatcher so serving continues."""
        operand = self.context_builder.pop_listener(channel)
        if operand is None:
            log.warning(f"stimulus_dropped channel={channel} reason=no_pending_listener")
            return
        operand.action_results.append(result)
        log.info(f"listener_resolved operand={operand.id} channel={channel}")
        try:
            self.loop.run(operand)
        except Exception as e:
            log.exception(f"system_failure error={e!r}")
            print(f"error: {e}", file=sys.stderr)
            rearmed = self.context_builder.add_operand(
                parent=operand,
                action_requests=[
                    ActionDescription(
                        id=new_id(),
                        action_name=channel,
                        method_name=self._listen_methods()[channel],
                    )
                ],
            )
            self.context_builder.park_listener(channel, rearmed)

    def _listen(self, action: Action, method_name: str) -> None:
        empty = Context(system_prompt="", history=[], available_actions={})
        while True:
            try:
                result = action.run(method_name, {}, empty)
            except (EOFError, KeyboardInterrupt):
                self._queue.put((action.name, None))
                return
            except Exception as e:
                log.exception(f"listener_failed channel={action.name} error={e!r}")
                self._queue.put((action.name, None))
                return
            self._queue.put((action.name, result))
