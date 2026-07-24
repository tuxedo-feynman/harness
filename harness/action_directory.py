from harness.action import Action
from harness.models import AvailableAction


class ActionDirectory:
    def __init__(self, actions: list[Action]):
        self._actions: dict[str, Action] = {}
        for action in actions:
            if action.name in self._actions:
                raise ValueError(f"Duplicate action name: {action.name!r}")
            self._actions[action.name] = action

    def get(self, name: str) -> Action:
        if name not in self._actions:
            raise KeyError(f"Unknown action: {name!r}")
        return self._actions[name]

    def method_index(self) -> dict[str, AvailableAction]:
        """Read-only projection used to populate Context.available_actions."""
        return {
            name: AvailableAction(kind=action.kind, methods=dict(action.methods))
            for name, action in self._actions.items()
        }

    def descriptions(self) -> str:
        """Human-readable summary of every action. For logging/debugging."""
        lines = []
        for action in self._actions.values():
            lines.append(f"{action.name} ({action.kind}): {action.description}")
            for method in action.methods.values():
                lines.append(f"  .{method.name}: {method.description}")
        return "\n".join(lines)
