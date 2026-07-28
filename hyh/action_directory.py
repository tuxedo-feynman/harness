from hyh.action import Action
from hyh.models import AvailableAction


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
