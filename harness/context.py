import logging
from datetime import datetime, timezone

from harness.action_directory import ActionDirectory
from harness.logger import new_id
from harness.models import ActionDescription, ActionResult, Context, Operand

log = logging.getLogger(__name__)


class ContextBuilder:
    """Owns the flat, ever-growing operand list for the life of the process.

    The first operand added is the root (parent=None). Every later operand
    attaches as a child of the current leaf, so history is a chain.
    """

    def __init__(self, system_prompt: str, action_directory: ActionDirectory):
        self.system_prompt = system_prompt
        self.action_directory = action_directory
        self._operands: list[Operand] = []

    def leaf(self) -> Operand | None:
        return self._operands[-1] if self._operands else None

    def add_operand(
        self,
        action_requests: list[ActionDescription] | None = None,
        action_results: list[ActionResult] | None = None,
    ) -> Operand:
        parent = self._operands[-1].id if self._operands else None
        operand = Operand(
            id=new_id(),
            created_at=datetime.now(timezone.utc),
            parent=parent,
            action_requests=list(action_requests or []),
            action_results=list(action_results or []),
        )
        self._operands.append(operand)
        log.info(f"operand_created id={operand.id} parent={operand.parent}")
        return operand

    def find(self, operand_id: str) -> Operand | None:
        return next((op for op in self._operands if op.id == operand_id), None)

    def build(self) -> Context:
        return Context(
            system_prompt=self.system_prompt,
            history=list(self._operands),
            available_actions=self.action_directory.method_index(),
        )
