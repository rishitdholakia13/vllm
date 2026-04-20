# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E tests for thinking_token_budget with reasoning models.

Configuration is read only from environment variables (no ``conftest`` hooks):

- ``VLLM_TEST_THINKING_TOKEN_BUDGET_SPEC_MODE``: ``none`` (default), ``mtp``, or
  ``eagle`` when ``VLLM_TEST_THINKING_TOKEN_BUDGET_SPECULATIVE_CONFIG`` is unset.
- ``VLLM_TEST_THINKING_TOKEN_BUDGET_TARGET_MODEL``: served HF model id (optional).
- ``VLLM_TEST_THINKING_TOKEN_BUDGET_DRAFT_MODEL``: draft ``model`` when JSON omits it
  (not merged for MTP-like methods).
- ``VLLM_TEST_THINKING_TOKEN_BUDGET_SPECULATIVE_CONFIG``: JSON object for
  ``--speculative-config`` (optional; overrides preset speculative dicts).
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal, TypedDict

import openai
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

MESSAGES = [{"role": "user", "content": "What is 1+1? Be concise."}]
THINK_BUDGET = 5

REASONING_CONFIG_JSON = (
    '{"reasoning_start_str": "<think>", '
    '"reasoning_end_str": "</think>"}'
)

SpecMode = Literal["none", "mtp", "eagle"]

# Preset draft (see tests/v1/e2e/spec_decode/test_spec_decode.py).
_PRESET_EAGLE3_DRAFT = "AngelSlim/Qwen3-8B_eagle3"

_ENV_SPEC_MODE = "VLLM_TEST_THINKING_TOKEN_BUDGET_SPEC_MODE"
_ENV_TARGET_MODEL = "VLLM_TEST_THINKING_TOKEN_BUDGET_TARGET_MODEL"
_ENV_DRAFT_MODEL = "VLLM_TEST_THINKING_TOKEN_BUDGET_DRAFT_MODEL"
_ENV_SPECULATIVE_CONFIG = "VLLM_TEST_THINKING_TOKEN_BUDGET_SPECULATIVE_CONFIG"


class _ThinkingBudgetProfile(TypedDict):
    served_model: str
    server_tail: list[str]
    gpu_memory_utilization: float
    max_wait_seconds: float | None


def _strip_opt(value: str) -> str | None:
    s = (value or "").strip()
    return s if s else None


def _spec_mode() -> SpecMode:
    raw = (os.environ.get(_ENV_SPEC_MODE, "none") or "none").strip().lower()
    if raw == "none":
        return "none"
    if raw == "mtp":
        return "mtp"
    if raw == "eagle":
        return "eagle"
    pytest.fail(f"{_ENV_SPEC_MODE} must be one of none, mtp, eagle; got {raw!r}")


def _method_hint(spec: dict[str, Any] | None) -> str:
    if not spec:
        return ""
    m = spec.get("method")
    return str(m).lower() if m is not None else ""


def _mtp_like_spec(spec: dict[str, Any]) -> bool:
    m = _method_hint(spec)
    return m == "mtp" or m.endswith("_mtp")


def _build_profile() -> _ThinkingBudgetProfile:
    mode = _spec_mode()
    target_override = _strip_opt(os.environ.get(_ENV_TARGET_MODEL, ""))
    draft_override = _strip_opt(os.environ.get(_ENV_DRAFT_MODEL, ""))
    spec_raw = _strip_opt(os.environ.get(_ENV_SPECULATIVE_CONFIG, ""))

    spec_dict: dict[str, Any] | None = None
    if spec_raw is not None:
        try:
            parsed = json.loads(spec_raw)
        except json.JSONDecodeError as e:
            pytest.fail(
                f"Invalid JSON for {_ENV_SPECULATIVE_CONFIG} "
                f"(environment variable): {e}"
            )
        if not isinstance(parsed, dict):
            pytest.fail("speculative-config JSON must be an object")
        spec_dict = parsed

    if spec_dict is None:
        if mode == "mtp":
            spec_dict = {
                "method": "mtp",
                "num_speculative_tokens": 1,
                "max_model_len": 2048,
            }
        elif mode == "eagle":
            spec_dict = {
                "method": "eagle3",
                "model": draft_override or _PRESET_EAGLE3_DRAFT,
                "num_speculative_tokens": 2,
                "max_model_len": 2048,
            }
        elif draft_override:
            pytest.fail(
                f"{_ENV_DRAFT_MODEL} is set but speculative decoding is off "
                f"({_ENV_SPEC_MODE}=none and {_ENV_SPECULATIVE_CONFIG} unset)."
            )

    if (
        spec_dict is not None
        and "model" not in spec_dict
        and draft_override
        and not _mtp_like_spec(spec_dict)
    ):
        spec_dict = {**spec_dict, "model": draft_override}

    has_spec = spec_dict is not None

    if has_spec:
        spec_cli = json.dumps(spec_dict, separators=(",", ":"))
        server_tail = ["--speculative-config", spec_cli]
        method = _method_hint(spec_dict)
        if "eagle" in method:
            gpu_mem, max_wait = 0.85, 480.0
        else:
            gpu_mem, max_wait = 0.55, None
    else:
        server_tail = ["--no-async-scheduling"]
        gpu_mem, max_wait = 0.4, None

    if target_override is not None:
        served = target_override
    elif spec_raw is not None and spec_dict is not None:
        if _mtp_like_spec(spec_dict):
            served = "Qwen/Qwen3.5-0.8B"
        elif "eagle" in _method_hint(spec_dict):
            served = "Qwen/Qwen3-8B"
        else:
            pytest.fail(
                f"Set {_ENV_TARGET_MODEL} for this speculative-config "
                "(unrecognized method for a default target)."
            )
    elif has_spec and mode == "mtp":
        served = "Qwen/Qwen3.5-0.8B"
    elif has_spec and mode == "eagle":
        served = "Qwen/Qwen3-8B"
    else:
        served = "Qwen/Qwen3-0.6B"

    return _ThinkingBudgetProfile(
        served_model=served,
        server_tail=server_tail,
        gpu_memory_utilization=gpu_mem,
        max_wait_seconds=max_wait,
    )


def _server_args(
    profile: _ThinkingBudgetProfile,
    reasoning_variant: Literal["default", "auto_config"],
) -> tuple[list[str], float | None]:
    args: list[str] = [
        "--reasoning-parser",
        "qwen3",
        "--max-model-len",
        "2048",
        "--enforce-eager",
        "--gpu-memory-utilization",
        str(profile["gpu_memory_utilization"]),
        *profile["server_tail"],
    ]
    if reasoning_variant == "default":
        args.extend(["--reasoning-config", REASONING_CONFIG_JSON])
    return args, profile["max_wait_seconds"]


@pytest.fixture(scope="module")
def thinking_token_budget_profile() -> _ThinkingBudgetProfile:
    return _build_profile()


@pytest.fixture(scope="module")
def served_model_name(thinking_token_budget_profile: _ThinkingBudgetProfile) -> str:
    return thinking_token_budget_profile["served_model"]


@pytest.fixture(scope="module")
def server(thinking_token_budget_profile):
    args, max_wait = _server_args(thinking_token_budget_profile, "default")
    model = thinking_token_budget_profile["served_model"]
    kwargs = {}
    if max_wait is not None:
        kwargs["max_wait_seconds"] = max_wait
    with RemoteOpenAIServer(model, args, **kwargs) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def server_with_auto_reasoning_config(thinking_token_budget_profile):
    args, max_wait = _server_args(thinking_token_budget_profile, "auto_config")
    model = thinking_token_budget_profile["served_model"]
    kwargs = {}
    if max_wait is not None:
        kwargs["max_wait_seconds"] = max_wait
    with RemoteOpenAIServer(model, args, **kwargs) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(request, server, server_with_auto_reasoning_config):
    server_map = {
        "default": server,
        "auto_config": server_with_auto_reasoning_config,
    }
    target_server = server_map[request.param]
    async with target_server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize("client", ["default", "auto_config"], indirect=True)
async def test_thinking_token_budget_mixed_requests(
    client: openai.AsyncOpenAI, served_model_name: str
):
    """Test that mixed requests (some with thinking_token_budget, some without)
    complete successfully without errors."""

    response_with_budget = await client.chat.completions.create(
        model=served_model_name,
        messages=MESSAGES,
        max_tokens=100,
        extra_body={"thinking_token_budget": THINK_BUDGET},
    )
    response_without_budget = await client.chat.completions.create(
        model=served_model_name,
        messages=MESSAGES,
        max_tokens=100,
    )

    msg_with = response_with_budget.choices[0].message
    msg_without = response_without_budget.choices[0].message

    assert msg_with.content or getattr(msg_with, "reasoning", None)
    assert msg_without.content or getattr(msg_without, "reasoning", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("client", ["default", "auto_config"], indirect=True)
async def test_thinking_token_budget_limits_reasoning(
    client: openai.AsyncOpenAI, served_model_name: str
):
    """Test that thinking_token_budget limits the number of reasoning tokens.

    In streaming mode each reasoning delta corresponds to one token, so
    counting non-empty reasoning_content chunks gives the exact token count.
    """

    reasoning_token_count = 0
    stream = await client.chat.completions.create(
        model=served_model_name,
        messages=MESSAGES,
        max_tokens=100,
        stream=True,
        extra_body={"thinking_token_budget": THINK_BUDGET},
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if getattr(delta, "reasoning", None):
            reasoning_token_count += 1

    assert reasoning_token_count == THINK_BUDGET, (
        f"reasoning tokens ({reasoning_token_count}) exceeded "
        f"thinking_token_budget ({THINK_BUDGET})"
    )
