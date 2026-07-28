import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from harness.action import LISTEN_METHOD, SEND_METHOD, Action
from harness.config import TelegramActionConfig
from harness.models import ActionResult, Context

log = logging.getLogger("harness." + __name__)

POLL_SECONDS = 50
RECONNECT_SECONDS = 5
TRANSIENT_ERRORS = (urllib.error.URLError, ConnectionError, TimeoutError)
API_ATTEMPTS = 3


class TelegramAction(Action):
    """Telegram bot channel via the Bot API (stdlib HTTP, no dependency).
    listen long-polls getUpdates; send posts sendMessage. Stateless with
    respect to chats: the allow-list lives in config, the send target is a
    required parameter (Policy supplies it from the stimulus metadata).
    """

    name = "telegram"
    kind = "effect"
    description = "Telegram bot channel: message the user on Telegram."
    methods = {
        SEND_METHOD: Action.MethodDescription(
            name=SEND_METHOD,
            description="Send a text message to a Telegram chat.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to send"},
                    "chat_id": {"type": "string", "description": "Numeric Telegram chat id"},
                },
                "required": ["text", "chat_id"],
            },
        ),
        LISTEN_METHOD: Action.MethodDescription(
            name=LISTEN_METHOD,
            description="Wait for the next Telegram message from an allowed chat.",
        ),
    }

    def __init__(self, config: TelegramActionConfig):
        self.config = config
        self._offset = 0

    def run(self, method_name: str, arguments: dict[str, Any], context: Context) -> ActionResult:
        if method_name == SEND_METHOD:
            text = arguments.get("text")
            chat_id = arguments.get("chat_id")
            if not isinstance(text, str):
                raise ValueError("'text' must be a string")
            if chat_id is None:
                raise ValueError("'chat_id' is required")
            self._api_retrying("sendMessage", {"chat_id": str(chat_id), "text": text})
            return ActionResult(contents=text)
        if method_name == LISTEN_METHOD:
            while True:
                try:
                    response = self._api(
                        "getUpdates",
                        {"timeout": POLL_SECONDS, "offset": self._offset, "allowed_updates": ["message"]},
                    )
                except TRANSIENT_ERRORS as e:
                    # Transport lifecycle, not a system failure: idle long-poll
                    # connections get reset routinely (NAT timeouts, wifi
                    # transitions). Reconnect after a pause. Everything else —
                    # bad token, API errors, send failures — still crashes.
                    log.warning(f"telegram_poll_reconnect error={e!r}")
                    time.sleep(RECONNECT_SECONDS)
                    continue
                for update in response.get("result", []):
                    self._offset = update["update_id"] + 1
                    result = self._accept(update)
                    if result is not None:
                        # Ack immediately: getUpdates only confirms consumption
                        # via the next call's offset. Without this, the final
                        # message before shutdown (e.g. "quit") is redelivered
                        # to the next process.
                        self._api_retrying("getUpdates", {"offset": self._offset, "timeout": 0, "limit": 1})
                        return result
        raise ValueError(f"Unknown method: {method_name!r}")

    def _accept(self, update: dict[str, Any]) -> ActionResult | None:
        """Build the result for an update if its chat passes the allow-list
        (entries match on numeric chat id or @username). Sender identity is
        event data, so it travels in the result's metadata. Messages from an
        allowed chat with unsupported content (voice, photo, ...) become
        empty-content stimuli so Policy can reply honestly."""
        message = update.get("message") or {}
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if chat_id is None:
            log.warning(f"telegram_update_skipped reason=no_chat keys={sorted(update.keys())}")
            return None
        username = (chat.get("username") or "").lower()
        if self.config.chat_ids and not self._allowed(chat_id, username):
            log.warning(
                f"telegram_message_ignored chat_id={chat_id} username={username or '?'} reason=not_in_chat_ids"
            )
            return None
        # The full message object is the event — record all of it. Rendering
        # for the model happens in the thinking adapters; policy's schema
        # filter keeps it out of send parameters.
        metadata = {"chat_id": str(chat_id), "username": username, "message": message}
        text = message.get("text") or message.get("caption")
        if not isinstance(text, str):
            log.warning(
                f"telegram_message_unsupported chat_id={chat_id} keys={sorted(message.keys())}"
            )
            return ActionResult(contents="", metadata=metadata)
        log.info(f"telegram_message_accepted chat_id={chat_id} username={username or '?'}")
        return ActionResult(contents=text, metadata=metadata)

    def _allowed(self, chat_id: Any, username: str) -> bool:
        for allowed in self.config.chat_ids:
            entry = str(allowed).lower()
            if entry == str(chat_id) or (username and entry.lstrip("@") == username):
                return True
        return False

    def _api_retrying(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """One-shot API calls (send, ack) retried through transient connection
        errors — same transport-lifecycle class as the poll reconnect. Crashes
        after API_ATTEMPTS; API-level errors (ok: false) never retry."""
        for attempt in range(API_ATTEMPTS):
            try:
                return self._api(method, params)
            except TRANSIENT_ERRORS as e:
                if attempt == API_ATTEMPTS - 1:
                    raise
                log.warning(f"telegram_api_retry method={method} attempt={attempt + 1} error={e!r}")
                time.sleep(RECONNECT_SECONDS)
        raise RuntimeError("unreachable")  # for the type checker

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
