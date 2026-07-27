import json
import logging
import urllib.request
from typing import Any

from harness.action import Action
from harness.config import TelegramActionConfig
from harness.models import ActionResult, Context

log = logging.getLogger("harness." + __name__)

POLL_SECONDS = 50


class TelegramAction(Action):
    """Telegram bot channel via the Bot API (stdlib HTTP, no dependency).
    receive long-polls getUpdates; send posts sendMessage. Stateless with
    respect to chats: the allow-list lives in config, the send target is a
    parameter (falling back to the sole numeric configured chat id).
    """

    name = "telegram"
    kind = "effect"
    description = "Telegram bot channel: message the user on Telegram."
    methods = {
        "send": Action.MethodDescription(
            name="send",
            description="Send a text message to a Telegram chat.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to send"},
                    "chat_id": {
                        "type": "string",
                        "description": "Numeric Telegram chat id. Optional if exactly one numeric chat id is configured.",
                    },
                },
                "required": ["text"],
            },
        ),
        "receive": Action.MethodDescription(
            name="receive",
            description="Wait for the next Telegram message from an allowed chat.",
            listen=True,
        ),
    }

    def __init__(self, config: TelegramActionConfig):
        self.config = config
        self._offset = 0

    def run(self, method_name: str, arguments: dict[str, Any], context: Context) -> ActionResult:
        if method_name == "send":
            text = arguments.get("text")
            if not isinstance(text, str):
                raise ValueError("'text' must be a string")
            self._api("sendMessage", {"chat_id": self._send_target(arguments), "text": text})
            return ActionResult(contents=text)
        if method_name == "receive":
            while True:
                response = self._api(
                    "getUpdates",
                    {"timeout": POLL_SECONDS, "offset": self._offset, "allowed_updates": ["message"]},
                )
                for update in response.get("result", []):
                    self._offset = update["update_id"] + 1
                    result = self._accept(update)
                    if result is not None:
                        return result
        raise ValueError(f"Unknown method: {method_name!r}")

    def _send_target(self, arguments: dict[str, Any]) -> str:
        explicit = arguments.get("chat_id")
        if explicit is not None:
            return str(explicit)
        numeric = [str(c) for c in self.config.chat_ids if str(c).lstrip("-").isdigit()]
        if len(numeric) == 1:
            return numeric[0]
        raise ValueError(
            "Cannot determine send target: pass 'chat_id' or configure exactly one"
            " numeric entry in telegram chat_ids (usernames cannot be send targets"
            " for private chats — the numeric id is logged when a message arrives)"
        )

    def _accept(self, update: dict[str, Any]) -> ActionResult | None:
        """Build the result for an update if its chat passes the allow-list
        (entries match on numeric chat id or @username). Sender identity is
        event data, so it travels in the result's metadata."""
        message = update.get("message") or {}
        text = message.get("text")
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if not isinstance(text, str) or chat_id is None:
            return None
        username = (chat.get("username") or "").lower()
        if self.config.chat_ids and not self._allowed(chat_id, username):
            log.warning(
                f"telegram_message_ignored chat_id={chat_id} username={username or '?'} reason=not_in_chat_ids"
            )
            return None
        log.info(f"telegram_message_accepted chat_id={chat_id} username={username or '?'}")
        return ActionResult(
            contents=text,
            metadata={"chat_id": str(chat_id), "username": username},
        )

    def _allowed(self, chat_id: Any, username: str) -> bool:
        for allowed in self.config.chat_ids:
            entry = str(allowed).lower()
            if entry == str(chat_id) or (username and entry.lstrip("@") == username):
                return True
        return False

    def _api(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"https://api.telegram.org/bot{self.config.token}/{method}"
        request = urllib.request.Request(
            url,
            data=json.dumps(params).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=POLL_SECONDS + 10) as response:
            payload = json.loads(response.read())
        if not payload.get("ok"):
            raise RuntimeError(f"Telegram API {method} failed: {payload}")
        return payload
