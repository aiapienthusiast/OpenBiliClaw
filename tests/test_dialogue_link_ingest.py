"""Tests for link ingestion wiring inside SocraticDialogue (issue #83)."""

from __future__ import annotations

from typing import Any

from openbiliclaw.agent.loop import AgentLoop
from openbiliclaw.agent.tools import ToolRegistry
from openbiliclaw.llm.base import LLMResponse
from openbiliclaw.soul.dialogue import DialogueLearningMode, SocraticDialogue
from openbiliclaw.sources.link_ingest import IngestedLink, LinkIngestResult


class FakeIngestor:
    """LinkIngestor-shaped double returning a canned result."""

    def __init__(self, result: LinkIngestResult | None = None, *, fail: bool = False) -> None:
        self.result = result or LinkIngestResult()
        self.fail = fail
        self.messages: list[str] = []

    async def ingest(self, message: str) -> LinkIngestResult:
        self.messages.append(message)
        if self.fail:
            raise RuntimeError("ingestor boom")
        return self.result


def _link_result() -> LinkIngestResult:
    link = IngestedLink(
        original_url="https://www.bilibili.com/video/BV1xx411c7mD",
        resolved_url="https://www.bilibili.com/video/BV1xx411c7mD",
        platform="bilibili",
        content_id="BV1xx411c7mD",
        title="讲透历史叙事",
        author="历史实验室",
        status="ok",
    )
    return LinkIngestResult(
        links=[link],
        prompt_block="【用户分享的链接】\n1. [B站] 《讲透历史叙事》",
        relation_hint="[分享了链接《讲透历史叙事》]",
        recorded_events=1,
    )


class FakeDialogueService:
    """LLMService-shaped double capturing the socratic prompt."""

    def __init__(self) -> None:
        self.user_message = ""

    async def complete_socratic_dialogue(self, **kwargs: Any) -> Any:
        self.user_message = str(kwargs.get("user_message", ""))
        return type("Response", (), {"content": "收到，视频看起来不错"})()


def _respond_dialogue(service: Any, ingestor: Any = None) -> SocraticDialogue:
    return SocraticDialogue(  # type: ignore[arg-type]
        llm=None,
        soul_engine=object(),
        llm_service=service,
        session="popup",
        learning_mode=DialogueLearningMode.REPLY_ONLY_TEST,
        link_ingestor=ingestor,
    )


async def test_respond_injects_link_block_into_prompt_only() -> None:
    service = FakeDialogueService()
    ingestor = FakeIngestor(_link_result())
    dialogue = _respond_dialogue(service, ingestor)

    reply = await dialogue.respond("我就喜欢这个 https://www.bilibili.com/video/BV1xx411c7mD")

    assert reply == "收到，视频看起来不错"
    # 链接摘要块注入当轮 prompt,且在时间戳后缀之前。
    assert "【用户分享的链接】" in service.user_message
    assert service.user_message.index("【用户分享的链接】") < service.user_message.index(
        "当前时间:"
    )
    # 历史与审计保持用户原文;relation_prefix 只影响 LLM 历史渲染。
    user_turn = dialogue.history[-2]
    assert user_turn.content == "我就喜欢这个 https://www.bilibili.com/video/BV1xx411c7mD"
    assert user_turn.relation_prefix == "[分享了链接《讲透历史叙事》]"
    assert "[分享了链接《讲透历史叙事》]" in dialogue._history_to_messages()[-1]["content"]


async def test_respond_without_links_keeps_baseline_prompt() -> None:
    service = FakeDialogueService()
    ingestor = FakeIngestor(LinkIngestResult())
    dialogue = _respond_dialogue(service, ingestor)

    # "://" 命中快速路径,但 extract_urls 找不到合法 URL,ingest 返回空结果。
    await dialogue.respond("随便说说 :// 这不是链接")

    assert ingestor.messages == ["随便说说 :// 这不是链接"]
    assert "【用户分享的链接】" not in service.user_message
    assert dialogue.history[-2].relation_prefix == ""


async def test_respond_ingestor_failure_never_blocks_reply() -> None:
    service = FakeDialogueService()
    dialogue = _respond_dialogue(service, FakeIngestor(fail=True))

    reply = await dialogue.respond("看看 https://www.zhihu.com/question/1")

    assert reply == "收到，视频看起来不错"
    assert "【用户分享的链接】" not in service.user_message
    assert dialogue.history[-2].relation_prefix == ""


async def test_respond_fast_path_skips_ingestor_without_url() -> None:
    service = FakeDialogueService()
    ingestor = FakeIngestor(_link_result())
    dialogue = _respond_dialogue(service, ingestor)

    await dialogue.respond("纯文本消息")

    assert ingestor.messages == []


class FakeAgentLLM:
    """Service-shaped double returning queued LLMResponses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete_with_native_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        caller: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
        bypass_semaphore: bool = False,
    ) -> LLMResponse:
        self.calls.append({"messages": [dict(message) for message in messages]})
        return self._responses.pop(0)


async def test_stream_agent_reply_injects_link_block() -> None:
    llm = FakeAgentLLM([LLMResponse(content="这条视频很棒")])
    loop = AgentLoop(llm, ToolRegistry([]))
    ingestor = FakeIngestor(_link_result())
    dialogue = _respond_dialogue(object(), ingestor)

    events = [
        event
        async for event in dialogue.stream_agent_reply(loop, "我就喜欢这个 https://b23.tv/abc")
    ]  # noqa: E501

    assert [event.type for event in events][-1] == "final"
    user_messages = [
        message for call in llm.calls for message in call["messages"] if message["role"] == "user"
    ]
    assert any("【用户分享的链接】" in str(message.get("content")) for message in user_messages)
    assert dialogue.history[-2].relation_prefix == "[分享了链接《讲透历史叙事》]"
