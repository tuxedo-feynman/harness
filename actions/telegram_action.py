import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any

from hyh.action import INPUT_METHOD, POLL_METHOD, REACT_METHOD, SEND_METHOD, TYPING_METHOD, Action
from hyh.config import TelegramActionConfig
from hyh.models import ActionResult, Context

log = logging.getLogger("hyh." + __name__)

POLL_SECONDS = 50
RECONNECT_SECONDS = 5
TRANSIENT_ERRORS = (urllib.error.URLError, ConnectionError, TimeoutError)
API_ATTEMPTS = 3
MESSAGE_LIMIT = 4096  # Telegram rejects longer sendMessage texts with a 400

# The Bot API's fixed set of reaction emoji; anything else is a 400.
ALLOWED_REACTIONS = (
    "👍", "👎", "❤", "🔥", "🥰", "👏", "😁", "🤔", "🤯", "😱", "🤬", "😢",
    "🎉", "🤩", "🤮", "💩", "🙏", "👌", "🕊", "🤡", "🥱", "🥴", "😍", "🐳",
    "❤‍🔥", "🌚", "🌭", "💯", "🤣", "⚡", "🍌", "🏆", "💔", "🤨", "😐", "🍓",
    "🍾", "💋", "🖕", "😈", "😴", "😭", "🤓", "👻", "👨‍💻", "👀", "🎃", "🙈",
    "😇", "😨", "🤝", "✍", "🤗", "🫡", "🎅", "🎄", "☃", "💅", "🤪", "🗿",
    "🆒", "💘", "🙉", "😎", "👾", "🤷‍♂", "🤷", "🤷‍♀", "😡",
)

# Split-point preference tiers: paragraph, line, sentence, word. Within a
# tier the rightmost match in the window wins. Sentence enders are a set
# because CJK punctuation takes no trailing space — and for spaceless
# scripts (Chinese) the sentence tier is the only thing between a
# paragraph and a hard mid-word cut.
SPLIT_BOUNDARIES = (
    ("\n\n",),
    ("\n",),
    (". ", "! ", "? ", "。", "！", "？", "۔ ", "। "),
    (" ",),
)


def _split_message(text: str, limit: int) -> list[str]:
    chunks = []
    while len(text) > limit:
        window = text[:limit]
        cut = limit
        for tier in SPLIT_BOUNDARIES:
            hits = [window.rfind(sep) + len(sep) for sep in tier if window.rfind(sep) > 0]
            if hits:
                cut = max(hits)
                break
        chunks.append(text[:cut])
        text = text[cut:]
    chunks.append(text)
    return chunks


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
            description=(
                "Send a text message to a Telegram chat. Text longer than "
                "4096 characters is split into consecutive messages."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to send"},
                    "chat_id": {"type": "string", "description": "Numeric Telegram chat id"},
                    "reply_to_message_id": {
                        "type": "string",
                        "description": (
                            "Optional: message_id of a message to reply to, "
                            "threading this response under it"
                        ),
                    },
                },
                "required": ["text", "chat_id"],
            },
        ),
        REACT_METHOD: Action.MethodDescription(
            name=REACT_METHOD,
            description=(
                "React to a Telegram message with a single emoji — a "
                "lightweight acknowledgment instead of a text reply."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "chat_id": {"type": "string", "description": "Numeric Telegram chat id"},
                    "message_id": {
                        "type": "string",
                        "description": "message_id of the message to react to",
                    },
                    "emoji": {"type": "string", "enum": list(ALLOWED_REACTIONS)},
                },
                "required": ["chat_id", "message_id", "emoji"],
            },
        ),
        POLL_METHOD: Action.MethodDescription(
            name=POLL_METHOD,
            description=(
                "Send a native multiple-choice poll to a Telegram chat. Polls "
                "are non-anonymous; each vote arrives back as a message giving "
                "the chosen option indexes into your options list."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "chat_id": {"type": "string", "description": "Numeric Telegram chat id"},
                    "question": {"type": "string", "description": "The question to ask"},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 10,
                        "description": "The answer choices",
                    },
                },
                "required": ["chat_id", "question", "options"],
            },
        ),
        TYPING_METHOD: Action.MethodDescription(
            name=TYPING_METHOD,
            description=(
                "Show the 'typing...' indicator in a Telegram chat for a few "
                "seconds. Fired automatically when thinking starts."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "chat_id": {"type": "string", "description": "Numeric Telegram chat id"},
                },
                "required": ["chat_id"],
            },
        ),
        INPUT_METHOD: Action.MethodDescription(
            name=INPUT_METHOD,
            description=(
                "The next input event from an allowed Telegram chat: a "
                "message, a poll vote, ..."
            ),
        ),
    }

    def __init__(self, config: TelegramActionConfig):
        self.config = config
        self._token = config.token or os.environ.get("TELEGRAM_BOT_TOKEN")
        if not self._token:
            raise ValueError(
                "telegram token missing: set token in config or TELEGRAM_BOT_TOKEN in the environment"
            )
        self._offset = 0

    def run(self, method_name: str, arguments: dict[str, Any], context: Context) -> ActionResult:
        if method_name == SEND_METHOD:
            text = arguments.get("text")
            chat_id = arguments.get("chat_id")
            if not isinstance(text, str):
                return ActionResult(contents="", error="'text' must be a string")
            if chat_id is None:
                return ActionResult(contents="", error="'chat_id' is required")
            reply_to = arguments.get("reply_to_message_id")
            if reply_to is not None:
                try:
                    reply_to = int(reply_to)
                except (TypeError, ValueError):
                    return ActionResult(
                        contents="",
                        error=f"'reply_to_message_id' must be numeric, got {reply_to!r}",
                    )
            for i, chunk in enumerate(_split_message(text, MESSAGE_LIMIT)):
                params: dict[str, Any] = {"chat_id": str(chat_id), "text": chunk}
                if reply_to is not None and i == 0:
                    # Only the first chunk threads; the rest follow it.
                    params["reply_parameters"] = {"message_id": reply_to}
                self._api_retrying("sendMessage", params)
            return ActionResult(contents=text)
        if method_name == TYPING_METHOD:
            chat_id = arguments.get("chat_id")
            if chat_id is None:
                raise ValueError("'chat_id' is required")
            self._api_retrying("sendChatAction", {"chat_id": str(chat_id), "action": "typing"})
            return ActionResult(contents="")
        if method_name == REACT_METHOD:
            chat_id = arguments.get("chat_id")
            message_id = arguments.get("message_id")
            emoji = arguments.get("emoji")
            if chat_id is None:
                return ActionResult(contents="", error="'chat_id' is required")
            try:
                message_id = int(message_id)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return ActionResult(
                    contents="", error=f"'message_id' must be numeric, got {message_id!r}"
                )
            if emoji not in ALLOWED_REACTIONS:
                return ActionResult(
                    contents="",
                    error=(
                        f"{emoji!r} is not an allowed telegram reaction emoji; "
                        "pick one from the method schema's enum"
                    ),
                )
            self._api_retrying(
                "setMessageReaction",
                {
                    "chat_id": str(chat_id),
                    "message_id": message_id,
                    "reaction": [{"type": "emoji", "emoji": emoji}],
                },
            )
            return ActionResult(contents=emoji)
        if method_name == POLL_METHOD:
            chat_id = arguments.get("chat_id")
            question = arguments.get("question")
            options = arguments.get("options")
            if chat_id is None:
                return ActionResult(contents="", error="'chat_id' is required")
            if not isinstance(question, str):
                return ActionResult(contents="", error="'question' must be a string")
            if not isinstance(options, list) or not 2 <= len(options) <= 10:
                return ActionResult(contents="", error="'options' must be a list of 2 to 10 strings")
            self._api_retrying(
                "sendPoll",
                {
                    "chat_id": str(chat_id),
                    "question": question,
                    "options": [{"text": str(option)} for option in options],
                    # Anonymous polls emit no poll_answer updates; votes only
                    # come back to listen() from non-anonymous polls.
                    "is_anonymous": False,
                },
            )
            return ActionResult(contents=question)
        if method_name == INPUT_METHOD:
            while True:
                try:
                    response = self._api(
                        "getUpdates",
                        {"timeout": POLL_SECONDS, "offset": self._offset, "allowed_updates": ["message", "poll_answer"]},
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
        empty-content stimuli so Policy can reply honestly. Poll votes arrive
        as poll_answer updates; in a private chat the voter's user id is the
        chat id, so the allow-list applies unchanged."""
        poll_answer = update.get("poll_answer")
        if poll_answer is not None:
            user = poll_answer.get("user") or {}
            chat_id = user.get("id")
            if chat_id is None:
                log.warning(f"telegram_update_skipped reason=no_voter keys={sorted(update.keys())}")
                return None
            username = (user.get("username") or "").lower()
            if self.config.chat_ids and not self._allowed(chat_id, username):
                log.warning(
                    f"telegram_vote_ignored chat_id={chat_id} username={username or '?'} reason=not_in_chat_ids"
                )
                return None
            log.info(f"telegram_vote_accepted chat_id={chat_id} username={username or '?'}")
            return ActionResult(
                contents=f"[poll vote: option_ids={poll_answer.get('option_ids', [])}]",
                metadata={"chat_id": str(chat_id), "username": username, "poll_answer": poll_answer},
            )
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
        url = f"https://api.telegram.org/bot{self._token}/{method}"
        request = urllib.request.Request(
            url,
            data=json.dumps(params).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=POLL_SECONDS + 10) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code >= 500:
                # Server-side hiccup: HTTPError is a URLError, so re-raising
                # keeps it in the transient class (poll reconnect, send retry).
                log.warning(f"telegram_http_error method={method} code={e.code} body={body}")
                raise
            raise RuntimeError(f"Telegram API {method} HTTP {e.code}: {body}") from e
        if not payload.get("ok"):
            raise RuntimeError(f"Telegram API {method} failed: {payload}")
        return payload
