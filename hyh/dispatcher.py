import logging
import queue
import threading

from hyh.action import INPUT_METHOD, Action
from hyh.action_directory import ActionDirectory
from hyh.context import ContextBuilder
from hyh.logger import new_id, record
from hyh.loop import ExecutionLoop
from hyh.models import ActionDescription, ActionResult, Context, Operand

log = logging.getLogger(__name__)


class Dispatcher:
    """The stimulus boundary: one daemon thread per listening channel blocks on
    the channel and pushes results onto a queue. Threads know nothing about
    operands — attribution to a pending listener happens at dequeue time, on
    this (main) thread, which owns all operand mutation.

    Errors are system failures: an exception in a turn or a listener thread
    propagates out of serve() and crashes the process.
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
        self._queue: queue.Queue[tuple[str, ActionResult | Exception | None]] = queue.Queue()
        self._closed: set[str] = set()  # channels whose transport has disconnected

    def _channels(self) -> list[str]:
        return [
            name
            for name, available in self.action_directory.method_index().items()
            if INPUT_METHOD in available.methods
        ]

    def arm_initial(self, root: Operand) -> None:
        """One pending input request per channel, attached to the root."""
        for order, channel in enumerate(self._channels()):
            operand = self.context_builder.add_operand(
                parents=[root],
                order=order,
                action_request=ActionDescription(
                    id=new_id(), action_name=channel, method_name=INPUT_METHOD
                ),
            )
            self.context_builder.park_listener(channel, operand)

    def serve(self, root: Operand) -> None:
        self.arm_initial(root)
        for channel in self._channels():
            threading.Thread(
                target=self._listen,
                args=(self.action_directory.get(channel),),
                daemon=True,
            ).start()
        try:
            while self.context_builder.listener_channels() - self._closed:
                channel, item = self._queue.get()
                if item is None:
                    self._closed.add(channel)
                    log.info(f"channel_closed channel={channel}")
                    continue
                if isinstance(item, Exception):
                    raise item
                self.handle_stimulus(channel, item)
        except KeyboardInterrupt:
            pass

    def handle_stimulus(self, channel: str, result: ActionResult) -> None:
        """Resolve the channel's pending listener with the result and run the
        turn. Turn errors propagate — the process crashes."""
        operand = self.context_builder.pop_listener(channel)
        if operand is None:
            log.warning(f"stimulus_dropped channel={channel} reason=no_pending_listener")
            return
        operand.action_result = result
        record(operand, f"resolved channel={channel}")
        self.loop.run(operand)

    def _listen(self, action: Action) -> None:
        empty = Context(system_prompt="", history=[], available_actions={})
        while True:
            try:
                result = action.run(INPUT_METHOD, {}, empty)
            except (EOFError, KeyboardInterrupt):
                self._queue.put((action.name, None))
                return
            except Exception as e:
                # Not suppression: the exception crosses the thread boundary
                # and serve() re-raises it, crashing the process.
                self._queue.put((action.name, e))
                return
            self._queue.put((action.name, result))
