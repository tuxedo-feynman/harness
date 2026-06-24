from harness.messages import Message


class ContextBuilder:
    def build(
        self,
        system_prompt: str,
        history: list[Message],
        latest_user_message: str,
    ) -> list[Message]:
        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.extend(history)
        messages.append(Message(role="user", content=latest_user_message))
        return messages
