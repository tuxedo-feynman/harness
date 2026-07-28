import logging
from datetime import datetime, timezone

from harness.action_directory import ActionDirectory
from harness.logger import new_id
from harness.models import ActionDescription, ActionResult, Context, Operand

log = logging.getLogger(__name__)


class ContextBuilder:
    """Owns the operand tree for the life of the process. The tree (parent
    pointers) is the sole source of truth; the flat list is storage only.

    Invariants: children only attach beneath resolved operands, so unresolved
    operands are always leaves. Resolved operands are immutable history;
    unresolved operands are live state -- filled by Policy, moved here.
    """

    def __init__(self, system_prompt: str, action_directory: ActionDirectory):
        self.system_prompt = system_prompt
        self.action_directory = action_directory
        self._operands: list[Operand] = []
        self._by_id: dict[str, Operand] = {}
        self._listeners: dict[str, Operand] = {}  # channel action name -> pending operand

    def add_root(self) -> Operand:
        """The synthetic session-start operand. Initial listeners attach to it."""
        if self._operands:
            raise RuntimeError("Root operand already exists")
        return self._add(parent_id=None)

    def add_operand(
        self,
        parent: Operand,
        action_requests: list[ActionDescription] | None = None,
        action_results: list[ActionResult] | None = None,
    ) -> Operand:
        return self._add(parent.id, action_requests, action_results)

    def _add(
        self,
        parent_id: str | None,
        action_requests: list[ActionDescription] | None = None,
        action_results: list[ActionResult] | None = None,
    ) -> Operand:
        operand = Operand(
            id=new_id(),
            created_at=datetime.now(timezone.utc),
            parent=parent_id,
            action_requests=list(action_requests or []),
            action_results=list(action_results or []),
        )
        self._operands.append(operand)
        self._by_id[operand.id] = operand
        log.info(f"operand_created id={operand.id} parent={operand.parent}")
        return operand

    def move(self, operand: Operand, new_parent: Operand) -> None:
        """Reparent a pending operand. Only unresolved operands (live state,
        e.g. armed listeners) are ever moved; resolved operands are history."""
        log.info(f"operand_moved id={operand.id} from={operand.parent} to={new_parent.id}")
        operand.parent = new_parent.id

    def park_listener(self, channel: str, operand: Operand) -> None:
        self._listeners[channel] = operand

    def pop_listener(self, channel: str) -> Operand | None:
        return self._listeners.pop(channel, None)

    def listener_channels(self) -> set[str]:
        return set(self._listeners)

    def build(self, anchor: Operand) -> Context:
        """Context as seen from anchor: history is the ancestor path from the
        root to anchor. Pending listeners and dead branches are off-path and
        excluded automatically."""
        path: list[Operand] = []
        cursor: Operand | None = anchor
        while cursor is not None:
            path.append(cursor)
            cursor = self._by_id.get(cursor.parent) if cursor.parent else None
        path.reverse()
        return Context(
            system_prompt=self.system_prompt,
            history=path,
            available_actions=self.action_directory.method_index(),
            listeners=list(self._listeners.values()),
        )
