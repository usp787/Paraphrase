from src.run_inference import build_prompt


class FakeTokenizer:
    def __init__(self):
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return "PROMPT"


def test_qwen_prompt_uses_thinking_switch():
    tokenizer = FakeTokenizer()
    assert build_prompt(tokenizer, "system", "question", "thinking") == "PROMPT"
    assert tokenizer.messages[0]["role"] == "system"
    assert tokenizer.kwargs["enable_thinking"] is True


def test_deepseek_prompt_has_no_system_role_and_forces_think_prefix():
    tokenizer = FakeTokenizer()
    rendered = build_prompt(tokenizer, "unused system", "question", "reasoning")
    assert rendered == "PROMPT<think>\n"
    assert [message["role"] for message in tokenizer.messages] == ["user"]
    assert "\\boxed{}" in tokenizer.messages[0]["content"]
