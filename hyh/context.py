import logging
from datetime import datetime, timezone

from hyh.action_directory import ActionDirectory
from hyh.logger import new_id
from hyh.models import ActionDescription, ActionResult, Context, Operand

log = logging.getLogger(__name__)


class ContextBuilder:
    """Owns the operand DAG for the life of the process. The graph (parent
    pointers) is the sole source of truth; the flat list is storage only.

    Invariants: children only attach beneath resolved operands, so unresolved
    operands are always leaves. Resolved operands are immutable history;
    unresolved operands are live state -- filled by the loop, moved here.
    """

    def __init__(self, system_prompt: str, action_directory: ActionDirectory):
        self.system_prompt = system_prompt
        self.action_directory = action_directory
        self._operands: list[Operand] = []
        self._by_id: dict[str, Operand] = {}
        self._listeners: dict[str, Operand] = {}  # channel action name -> pending operand

    def add_root(self) -> Operand:
        """The synthetic session-start operand. Initial input requests attach
        to it."""
        if self._operands:
            raise RuntimeError("Root operand already exists")
        return self._add(parents=[], order=0)

    def add_operand(
        self,
        parents: list[Operand],
        order: int,
        action_request: ActionDescription | None = None,
        action_result: ActionResult | None = None,
    ) -> Operand:
        return self._add([p.id for p in parents], order, action_request, action_result)

    def _add(
        self,
        parents: list[str],
        order: int,
        action_request: ActionDescription | None = None,
        action_result: ActionResult | None = None,
    ) -> Operand:
        operand = Operand(
            id=new_id(),
            created_at=datetime.now(timezone.utc),
            parents=list(parents),
            order=order,
            action_request=action_request,
            action_result=action_result,
        )
        self._operands.append(operand)
        self._by_id[operand.id] = operand
        log.info(f"operand_created id={operand.id} parents={','.join(parents) or None} order={order}")
        return operand

    def move(self, operand: Operand, new_parents: list[Operand]) -> None:
        """Reparent a pending operand. Only unresolved operands (live state,
        e.g. armed input requests) are ever moved; resolved operands are
        history."""
        log.info(
            f"operand_moved id={operand.id} from={','.join(operand.parents)}"
            f" to={','.join(p.id for p in new_parents)}"
        )
        operand.parents = [p.id for p in new_parents]

    def park_listener(self, channel: str, operand: Operand) -> None:
        self._listeners[channel] = operand

    def pop_listener(self, channel: str) -> Operand | None:
        return self._listeners.pop(channel, None)

    def listener_channels(self) -> set[str]:
        return set(self._listeners)

    def build(self, frontier: list[Operand]) -> Context:
        """Context as seen from the frontier: history is the ancestor closure
        of the frontier, linearized in topological order with sibling-order
        tie-breaks (deterministic). Pending listeners and dead branches are
        off the closure and excluded automatically."""
        closure: dict[str, Operand] = {}
        stack = list(frontier)
        while stack:
            cursor = stack.pop()
            if cursor.id in closure:
                continue
            closure[cursor.id] = cursor
            stack.extend(self._by_id[pid] for pid in cursor.parents)

        history: list[Operand] = []
        emitted: set[str] = set()
        ready = [op for op in closure.values() if not op.parents]
        while ready:
            ready.sort(key=lambda op: (op.order, op.created_at, op.id))
            operand = ready.pop(0)
            history.append(operand)
            emitted.add(operand.id)
            for candidate in closure.values():
                if (
                    candidate.id not in emitted
                    and candidate not in ready
                    and all(pid in emitted for pid in candidate.parents)
                ):
                    ready.append(candidate)
        return Context(
            system_prompt=self.system_prompt,
            history=history,
            available_actions=self.action_directory.method_index(),
            listeners=list(self._listeners.values()),
        )
