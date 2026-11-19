"""
OpenAI SDK runtime — dùng cho:

  Blue Team → OpenRouter liquid/lfm-3-1.7b:free (create_blue_pair)
  Red Team  → OpenAI gpt-4o-mini (create_openai_pair) khi RED_TEAM_PROVIDER=openai

Gemini Red Team dùng Google ADK trong agents/*.py — không đi qua file này.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from core.config import (
    get_red_model,
    get_red_provider,
    get_blue_model,
    get_blue_provider,
    blue_client_kwargs,
    red_openai_client_kwargs,
)


class EmptyModelResponseError(RuntimeError):
    """No usable model answer; never report this as an allowed empty reply."""


@dataclass
class OpenAIAgent:
    name: str
    instruction: str
    provider: str = "openai"


@dataclass
class _MockInvocationContext:
    user_id: str = "student"


@dataclass
class OpenAIRunner:
    """Optional ADK-style plugins + Chat Completions."""

    app_name: str
    model: str
    plugins: list = field(default_factory=list)
    provider: str = "openai"
    temperature: float = 0.4
    max_output_tokens: int | None = None
    max_model_retries: int = 0
    model_retries: int = field(default=0, init=False)
    client_kwargs: dict = field(default_factory=dict)
    input_hooks: list[Callable[[str], str | None]] = field(default_factory=list)
    output_hooks: list[Callable[[str], str]] = field(default_factory=list)

    def _client(self):
        from openai import OpenAI

        return OpenAI(**(self.client_kwargs or {}))

    async def _create_completion(self, client, **kwargs):
        """Retry only the provider call, without charging input quota again."""
        for attempt in range(self.max_model_retries + 1):
            try:
                completion = client.chat.completions.create(**kwargs)
                message = completion.choices[0].message
                if not (message.content or "").strip() and not getattr(message, "refusal", None):
                    # Some providers spend a short output budget before emitting
                    # any visible text. One bounded budget increase stays on the
                    # same provider/model and reuses the same admission decision.
                    if getattr(completion.choices[0], "finish_reason", None) == "length" and kwargs.get("max_tokens", 0):
                        kwargs["max_tokens"] = min(1024, kwargs["max_tokens"] * 2)
                    raise EmptyModelResponseError("Provider returned no visible answer")
                return completion
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                empty = isinstance(exc, EmptyModelResponseError)
                if (status not in (429, 500, 502, 503, 504) and not empty) or attempt == self.max_model_retries:
                    raise
                headers = getattr(getattr(exc, "response", None), "headers", {})
                try:
                    delay = max(1.0, float(headers.get("retry-after", 2 ** (attempt + 1))))
                except (TypeError, ValueError):
                    delay = 2 ** (attempt + 1)
                if delay > 30:
                    raise
                self.model_retries += 1
                label = "empty response" if empty else f"HTTP {status}"
                print(f"Provider {label}; retry {attempt + 1}/{self.max_model_retries} after {delay:g}s.", flush=True)
                await asyncio.sleep(delay)

    async def chat(self, agent: OpenAIAgent, user_message: str, *, user_id: str = "student") -> str:
        for hook in self.input_hooks:
            blocked = hook(user_message)
            if blocked:
                return blocked

        block_msg = await self._run_input_plugins(user_message, user_id=user_id)
        if block_msg is not None:
            return block_msg

        client = self._client()
        completion = await self._create_completion(
            client,
            model=self.model,
            messages=[
                {"role": "system", "content": agent.instruction},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
            **({"max_tokens": self.max_output_tokens} if self.max_output_tokens is not None else {}),
        )
        message = completion.choices[0].message
        text = (message.content or getattr(message, "refusal", None) or "").strip()

        for hook in self.output_hooks:
            text = hook(text)

        text = await self._run_output_plugins(text, user_message=user_message)
        return text

    async def _run_input_plugins(self, user_message: str, *, user_id: str = "student") -> str | None:
        if not self.plugins:
            return None
        try:
            from google.genai import types
        except ImportError:
            return None

        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )
        ctx = _MockInvocationContext(user_id=user_id)
        for plugin in self.plugins:
            cb = getattr(plugin, "on_user_message_callback", None)
            if cb is None:
                continue
            try:
                result = await cb(
                    invocation_context=ctx, user_message=user_content
                )
            except TypeError:
                result = cb(invocation_context=ctx, user_message=user_content)
            if result is None:
                continue
            return _content_to_text(result)
        return None

    async def _run_output_plugins(self, text: str, *, user_message: str | None = None) -> str:
        if not self.plugins or not text:
            return text
        try:
            from google.genai import types
        except ImportError:
            return text

        content = types.Content(
            role="model", parts=[types.Part.from_text(text=text)]
        )

        class _Resp:
            pass

        llm_response = _Resp()
        llm_response.content = content

        class _Ctx:
            pass

        ctx = _Ctx()
        ctx.user_content = types.Content(role="user", parts=[types.Part.from_text(text=user_message or "")])

        for plugin in self.plugins:
            cb = getattr(plugin, "after_model_callback", None)
            if cb is None:
                continue
            try:
                out = await cb(callback_context=ctx, llm_response=llm_response)
            except TypeError:
                out = cb(callback_context=ctx, llm_response=llm_response)
            if out is not None and getattr(out, "content", None) is not None:
                llm_response = out
        return _content_to_text(llm_response.content) or text


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = getattr(content, "parts", None) or []
    chunks = []
    for part in parts:
        t = getattr(part, "text", None)
        if t:
            chunks.append(t)
    return "".join(chunks)


def _make_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    model: str,
    provider: str,
    client_kwargs: dict,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
    max_output_tokens: int | None = None,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    agent = OpenAIAgent(name=name, instruction=instruction, provider=provider)
    runner = OpenAIRunner(
        app_name=app_name,
        model=model,
        provider=provider,
        client_kwargs=client_kwargs,
        plugins=list(plugins or []),
        input_hooks=list(input_hooks or []),
        output_hooks=list(output_hooks or []),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return agent, runner


def create_blue_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    """Blue Team — always OpenRouter liquid/lfm-3-1.7b:free."""
    return _make_pair(
        name=name,
        instruction=instruction,
        app_name=app_name,
        model=get_blue_model(),
        provider=get_blue_provider(),
        client_kwargs=blue_client_kwargs(),
        max_output_tokens=512,
        plugins=plugins,
        input_hooks=input_hooks,
        output_hooks=output_hooks,
        temperature=temperature,
    )


def create_openai_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
    model: str | None = None,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    """Red Team OpenAI path (default = soft model; advance may pass harder)."""
    return _make_pair(
        name=name,
        instruction=instruction,
        app_name=app_name,
        model=model or get_red_model(),
        provider=get_red_provider(),
        client_kwargs=red_openai_client_kwargs(),
        plugins=plugins,
        input_hooks=input_hooks,
        output_hooks=output_hooks,
        temperature=temperature,
    )
