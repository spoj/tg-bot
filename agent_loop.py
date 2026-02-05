"""Generic async agent loop with pluggable model calls, tool execution, and hooks.

This module provides a reusable agent loop abstraction that supports both:
- Batch/non-interactive use (like fs_checking)
- Interactive use with message injection (like tg-bot)

The loop is fully DI-based: you provide callables for model completion and tool
execution. Optional hooks enable interactive patterns like pending message draining.

Example (batch mode):

    from agent_loop import run_agent_loop

    async def call_model(messages, tools):
        response = await my_api.complete(messages=messages, tools=tools)
        return response.message, response.usage

    async def execute_tool(name, args):
        if name == "search":
            return json.dumps(search(args["query"]))
        return f"Unknown tool: {name}"

    result = await run_agent_loop(
        call_model=call_model,
        tool_executor=execute_tool,
        initial_messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What's 2+2?"},
        ],
    )

    if result.success:
        print(result.final_message)

Example (interactive mode with pending messages):

    # Define hooks for interactive chat
    async def before_call(messages):
        if session.pending_messages:
            for text in session.pending_messages:
                messages.append({"role": "user", "content": text})
            session.pending_messages.clear()
            return messages, True  # modified
        return messages, False

    async def before_tools(assistant_msg):
        if session.pending_messages:
            return False  # discard tool calls, retry
        return True  # proceed with tools

    result = await run_agent_loop(
        call_model=call_model,
        tool_executor=execute_tool,
        initial_messages=messages,
        hooks=LoopHooks(
            before_model_call=before_call,
            before_tool_execution=before_tools,
        ),
    )
"""

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Any, Protocol, cast


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class AgentResult:
    """Result of an agent loop run."""

    success: bool
    final_message: str
    iterations: int
    messages: list[dict]
    usage: dict = field(
        default_factory=lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read": 0,
            "cache_write": 0,
        }
    )
    tool_calls_count: int = 0


# =============================================================================
# Type Aliases
# =============================================================================

# Model callable: (messages, tools) -> (assistant_message_dict, usage_dict)
ModelCallable = Callable[
    [list[dict], list[dict] | None],
    Awaitable[tuple[dict, dict]],
]

# Tool executor: (name, args) -> result_string
SyncToolExecutor = Callable[[str, dict], str]
AsyncToolExecutor = Callable[[str, dict], Awaitable[str]]
ToolExecutor = SyncToolExecutor | AsyncToolExecutor


# =============================================================================
# Hook Protocols
# =============================================================================


class BeforeModelCallHook(Protocol):
    """Called before each model API call.

    Use for:
    - Injecting pending messages
    - Modifying message list
    - Logging/metrics

    Returns:
        Tuple of (possibly modified messages, was_modified flag).
        If was_modified is True, the loop may use this for decisions.
    """

    async def __call__(self, messages: list[dict]) -> tuple[list[dict], bool]: ...


class AfterModelCallHook(Protocol):
    """Called after each model API call.

    Use for:
    - Logging usage/metrics
    - Streaming partial responses
    - Custom processing
    """

    async def __call__(
        self,
        iteration: int,
        assistant_message: dict,
        usage: dict,
    ) -> None: ...


class BeforeToolExecutionHook(Protocol):
    """Called before executing tool calls.

    Use for:
    - Checking if pending messages arrived (return False to discard tools)
    - Logging tool calls
    - Validation

    Returns:
        True to proceed with tool execution, False to discard and retry model call.
    """

    async def __call__(self, assistant_message: dict) -> bool: ...


class AfterToolExecutionHook(Protocol):
    """Called after tool execution completes.

    Use for:
    - Logging tool results
    - Streaming tool outputs
    - Custom processing
    """

    async def __call__(self, tool_results: list[dict]) -> None: ...


class BeforeFinalResponseHook(Protocol):
    """Called before returning a final (non-tool) assistant response.

    Use for:
    - Checking for pending messages
    - Discarding stale responses (return False to retry)

    Returns:
        True to finalize response, False to discard and retry model call.
    """

    async def __call__(self, assistant_message: dict) -> bool: ...


class OnFinalHook(Protocol):
    """Called when loop completes (success or max iterations).

    Use for:
    - Final logging
    - Cleanup
    - Metrics
    """

    async def __call__(
        self,
        success: bool,
        iterations: int,
        final_message: str,
        total_usage: dict,
    ) -> None: ...


@dataclass
class LoopHooks:
    """Container for optional loop hooks.

    All hooks are optional. Provide only the ones you need.
    """

    before_model_call: BeforeModelCallHook | None = None
    after_model_call: AfterModelCallHook | None = None
    before_tool_execution: BeforeToolExecutionHook | None = None
    after_tool_execution: AfterToolExecutionHook | None = None
    before_final_response: BeforeFinalResponseHook | None = None
    on_final: OnFinalHook | None = None


# =============================================================================
# Helpers
# =============================================================================


def accumulate_usage(total: dict, usage: dict) -> None:
    """Add usage from a response to the running total.

    Handles different key formats from various providers.
    """
    # Standard keys (our normalized format)
    total["input_tokens"] += usage.get("input_tokens", 0)
    total["output_tokens"] += usage.get("output_tokens", 0)
    total["cache_read"] += usage.get("cache_read", 0)
    total["cache_write"] += usage.get("cache_write", 0)

    # OpenAI/LiteLLM style keys (fallback)
    if "prompt_tokens" in usage:
        total["input_tokens"] += usage.get("prompt_tokens", 0)
    if "completion_tokens" in usage:
        total["output_tokens"] += usage.get("completion_tokens", 0)


async def _execute_single_tool(
    tc: dict,
    tool_executor: ToolExecutor,
    is_async: bool,
) -> dict:
    """Execute a single tool call, returning tool message dict.

    Catches exceptions and converts them to error result strings.
    """
    func_name = tc.get("function", {}).get("name", "unknown")
    tool_call_id = tc.get("id", "unknown")

    try:
        args_str = tc.get("function", {}).get("arguments", "{}")
        if isinstance(args_str, str):
            import json

            args = json.loads(args_str)
        else:
            args = args_str  # Already a dict
    except Exception as e:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": f"Error parsing arguments: {e}",
        }

    try:
        if is_async:
            result = await cast(Any, tool_executor)(func_name, args)
        else:
            result = cast(Any, tool_executor)(func_name, args)
    except Exception as e:
        result = f"Tool error: {e}"

    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": result,
    }


async def process_tool_calls(
    tool_calls: list[dict],
    tool_executor: ToolExecutor,
    parallel: bool = True,
) -> list[dict]:
    """Execute tool calls and return tool response messages.

    Args:
        tool_calls: List of tool call dicts from assistant message.
        tool_executor: Function (name, args) -> result_string.
        parallel: If True and executor is async, run tools in parallel.

    Returns:
        List of tool response message dicts.
    """
    if not tool_calls:
        return []

    is_async = inspect.iscoroutinefunction(tool_executor)

    if is_async and parallel and len(tool_calls) > 1:
        # Run all tool calls in parallel
        results = await asyncio.gather(
            *[
                _execute_single_tool(tc, tool_executor, is_async=True)
                for tc in tool_calls
            ],
            return_exceptions=False,  # Exceptions already caught inside
        )
        return list(results)
    else:
        # Run sequentially
        results = []
        for tc in tool_calls:
            result = await _execute_single_tool(tc, tool_executor, is_async)
            results.append(result)
        return results


# =============================================================================
# Main Agent Loop
# =============================================================================


async def run_agent_loop(
    call_model: ModelCallable,
    tool_executor: ToolExecutor,
    initial_messages: list[dict],
    tools: list[dict] | None = None,
    max_iterations: int = 50,
    hooks: LoopHooks | None = None,
    parallel_tools: bool = True,
) -> AgentResult:
    """Run an agent loop until completion or max iterations.

    The loop continues until:
    1. Agent stops calling tools (returns final message)
    2. max_iterations reached

    Hooks enable interactive patterns:
    - before_model_call: Inject pending messages
    - before_tool_execution: Discard tool calls if new messages arrived
    - before_final_response: Discard stale final responses if new messages arrived
    - after_* hooks: Logging, streaming, metrics

    Args:
        call_model: Async function (messages, tools) -> (assistant_msg, usage).
        tool_executor: Function (name, args) -> result_string (sync or async).
        initial_messages: Starting messages (system + user).
        tools: Tool definitions (OpenAI format).
        max_iterations: Maximum loop iterations.
        hooks: Optional LoopHooks for interactive patterns.
        parallel_tools: Run async tool calls in parallel.

    Returns:
        AgentResult with success status, final message, usage stats.
    """
    hooks = hooks or LoopHooks()
    messages = list(initial_messages)
    total_usage: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read": 0,
        "cache_write": 0,
    }
    tool_calls_count = 0
    iteration = 0
    final_message = ""
    last_assistant_content = ""

    while iteration < max_iterations:
        iteration += 1

        # --- Before model call hook ---
        if hooks.before_model_call:
            messages, _ = await hooks.before_model_call(messages)

        # --- Call the model ---
        assistant_message, usage = await call_model(messages, tools)
        accumulate_usage(total_usage, usage)

        # --- After model call hook ---
        if hooks.after_model_call:
            await hooks.after_model_call(iteration, assistant_message, usage)

        # Add assistant message to history
        messages.append(assistant_message)
        if assistant_message.get("content"):
            last_assistant_content = assistant_message.get("content", "") or ""

        # Check for tool calls
        tool_calls = assistant_message.get("tool_calls", [])

        if tool_calls:
            # --- Before tool execution hook ---
            if hooks.before_tool_execution:
                proceed = await hooks.before_tool_execution(assistant_message)
                if not proceed:
                    # Discard this assistant message (tool calls not executed)
                    messages.pop()
                    continue  # Retry model call

            # Execute tools
            tool_results = await process_tool_calls(
                tool_calls, tool_executor, parallel=parallel_tools
            )
            tool_calls_count += len(tool_calls)
            messages.extend(tool_results)

            # --- After tool execution hook ---
            if hooks.after_tool_execution:
                await hooks.after_tool_execution(tool_results)
        else:
            # No tool calls - agent is done (unless hook discards)
            if hooks.before_final_response:
                proceed = await hooks.before_final_response(assistant_message)
                if not proceed:
                    messages.pop()
                    continue

            final_message = assistant_message.get("content", "") or ""

            # --- On final hook ---
            if hooks.on_final:
                await hooks.on_final(True, iteration, final_message, total_usage)

            return AgentResult(
                success=True,
                final_message=final_message,
                iterations=iteration,
                messages=messages,
                usage=total_usage,
                tool_calls_count=tool_calls_count,
            )

    # Max iterations reached
    final_message = last_assistant_content or "(max iterations reached)"

    if hooks.on_final:
        await hooks.on_final(False, iteration, final_message, total_usage)

    return AgentResult(
        success=False,
        final_message=final_message,
        iterations=iteration,
        messages=messages,
        usage=total_usage,
        tool_calls_count=tool_calls_count,
    )
