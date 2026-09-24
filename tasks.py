# tasks.py -- Real background task execution.
#
# For requests the user wants to kick off and check back on later, instead
# of waiting live in the chat stream. Runs as an in-process asyncio task
# (realistic for a single Render web service -- no separate worker dyno
# needed), persisted to storage.py so status survives across requests and
# is visible even if the user navigates away and comes back.
#
# Each plan step is executed by giving the model real tool access and
# letting it decide the actual arguments -- a plan step's description
# ("find EV sales data") is not itself valid tool arguments, so this
# reuses the same real tool-calling machinery as the live agentic loop
# rather than mechanically mapping step types to hardcoded calls.
import asyncio, uuid, json, time

import storage
from tools import execute_tool, TOOL_SCHEMAS, gemini_tool_declarations, ToolSession
from providers import CALL_FUNCS

MAX_STEPS = 8  # hard ceiling regardless of what the planner returns


async def create_task(uid: str, request: str, mode: str = "heavy") -> str:
    """Creates a task row and kicks off real background execution.
    Returns the task id immediately -- the caller polls for status."""
    task_id = uuid.uuid4().hex[:12]
    await storage.create_task_row(task_id, uid, mode, request)
    asyncio.create_task(_execute_task(task_id))
    return task_id


async def get_task_for_user(task_id: str, uid: str) -> dict | None:
    """Ownership-checked read -- returns None if the task doesn't exist
    OR belongs to someone else, so callers can't distinguish 'not found'
    from 'not yours' (no information leak about other users' task ids)."""
    task = await storage.get_task(task_id)
    if not task or task.get("uid") != uid:
        return None
    return _serialize(task)


async def list_tasks_for_user(uid: str) -> list:
    rows = await storage.list_tasks(uid)
    return [_serialize(t) for t in rows]


async def cancel_task(task_id: str, uid: str) -> bool:
    return await storage.delete_task(task_id, uid)


def _serialize(task: dict) -> dict:
    try:
        plan = json.loads(task.get("plan") or "[]")
    except Exception:
        plan = []
    return {
        "id": task["id"], "mode": task["mode"], "request": task["request"],
        "status": task["status"], "plan": plan,
        "progress_current": task["progress_current"], "progress_total": task["progress_total"],
        "result": task.get("result") or "", "error": task.get("error") or "",
        "created": task["created"], "updated": task["updated"],
    }


async def _drain_tool(name: str, args: dict, uid: str, session: "ToolSession") -> dict:
    """execute_tool is an async generator now (it can yield live progress
    events as a tool runs) -- background tasks aren't on an open SSE
    connection to forward those to, so this just drains it down to the
    final result event, same as before."""
    result = None
    async for ev in execute_tool(name, args, uid=uid, session=session):
        if ev["type"] == "result":
            result = ev
    return result or {"ok": False, "result_text": "Tool produced no result.", "summary": "empty"}


async def _decide_and_run_step(request: str, step: dict, prior: list, mode: str, uid: str,
                                session: "ToolSession") -> dict:
    """Runs ONE step by giving a real tool-capable model the step
    description plus accumulated context, and letting it decide the
    actual tool call (or just answer directly if no tool fits).

    `session` is shared across every step of the task (see _execute_task) --
    so if one step writes a file in the sandbox, a later step's
    sandbox_run_command can still see it, the same iterative-project
    behavior the live agentic loop gets within a turn."""
    from router import _agentic_chain_for  # local import: avoids a circular import at module load time

    context = f"Original request: {request}\n\nExecute this specific step now: [{step.get('type','STEP')}] {step.get('description','')}"
    if prior:
        recent = "\n".join(f"- {p.get('description','')}: {(p.get('result_text') or '')[:300]}" for p in prior[-3:])
        context += f"\n\nResults from earlier steps:\n{recent}"

    messages = [
        {"role": "system", "content": (
            "You are executing one step of a larger plan. Call the single most appropriate "
            "tool to complete THIS step, with real, complete arguments (full code/content, "
            "never placeholders). If no tool genuinely fits, just answer with the result text directly. "
            "IMPORTANT: this task runs unattended and its sandbox is destroyed the moment the task "
            "ends -- there is no follow-up turn and no one will read raw sandbox files afterward. "
            "sandbox_write_file/sandbox_run_command are for verifying work DURING this task (e.g. "
            "running a script to confirm it works) -- they are NOT how you deliver anything. Any file, "
            "script, or document the user should actually receive MUST be produced with "
            "create_code_file, create_document, create_zip, or generate_image -- never leave the "
            "final deliverable sitting only in the sandbox."
        )},
        {"role": "user", "content": context},
    ]

    for prov, model in _agentic_chain_for(mode):
        fn = CALL_FUNCS.get(prov)
        if not fn:
            continue
        try:
            tools_payload = gemini_tool_declarations() if prov == "gemini" else TOOL_SCHEMAS
            res = await fn(model, messages, tools=tools_payload)
        except Exception:
            continue
        tool_calls = res.get("tool_calls") or []
        if tool_calls:
            tc = tool_calls[0]
            name = tc.get("function", {}).get("name", "")
            try:
                args = json.loads(tc.get("function", {}).get("arguments") or "{}")
            except Exception:
                args = {}
            outcome = await _drain_tool(name, args, uid, session)
            return {"description": step.get("description", ""), "tool": name,
                    "result_text": outcome.get("result_text", ""), "artifact": outcome.get("artifact")}
        return {"description": step.get("description", ""), "tool": None,
                "result_text": res.get("content") or "", "artifact": None}
    return {"description": step.get("description", ""), "tool": None,
            "result_text": f"Could not complete this step (all providers unavailable): {step.get('description','')}",
            "artifact": None}


async def _compile_output(request: str, results: list) -> str:
    """Synthesizes a coherent final answer from all completed steps --
    not a raw concatenation of step outputs."""
    from router import _agentic_chain_for

    summary = "\n\n".join(
        f"Step: {r.get('description','')}\nResult: {(r.get('result_text') or '')[:800]}"
        for r in results
    )
    messages = [
        {"role": "system", "content": (
            "Compile a clear, complete final answer for the user from these completed work "
            "steps. Reference specific findings and figures. Synthesize into a coherent answer "
            "-- do not just repeat the raw step results verbatim."
        )},
        {"role": "user", "content": f"Original request: {request}\n\nCompleted steps:\n{summary}"},
    ]
    for prov, model in _agentic_chain_for("heavy"):
        fn = CALL_FUNCS.get(prov)
        if not fn:
            continue
        try:
            res = await fn(model, messages)
            content = res.get("content")
            if content:
                return content
        except Exception:
            continue
    return summary or "Task completed, but no results were produced."


async def _execute_task(task_id: str):
    task = await storage.get_task(task_id)
    if not task:
        return
    uid, request, mode = task["uid"], task["request"], task["mode"]
    session = ToolSession()  # task-scoped only (no conversation_id) -- one sandbox for the whole task, shared across all steps
    await session.load()  # no-op without a conversation_id, kept for consistency with router.py's usage
    try:
        await storage.update_task(task_id, status="planning")
        from router import create_plan
        plan = await create_plan(request)
        steps = (plan.get("steps") or [])[:MAX_STEPS]
        if not steps:
            steps = [{"type": "ANSWER", "description": request}]
        await storage.update_task(task_id, plan=json.dumps(steps), progress_total=len(steps))

        await storage.update_task(task_id, status="executing")
        results = []
        for i, step in enumerate(steps):
            outcome = await _decide_and_run_step(request, step, results, mode, uid, session)
            results.append(outcome)
            await storage.update_task(task_id, progress_current=i + 1)

        await storage.update_task(task_id, status="verifying")
        final_text = await _compile_output(request, results)

        # Collect any real artifacts (documents/images/etc) produced along
        # the way so the frontend can render them, not just the text.
        artifacts = [r["artifact"] for r in results if r.get("artifact")]
        await storage.update_task(task_id, status="completed",
            result=json.dumps({"text": final_text, "artifacts": artifacts}))
    except Exception as e:
        await storage.update_task(task_id, status="failed", error=str(e)[:500])
    finally:
        await session.close()


# ── STALE-TASK WATCHDOG ─────────────────────────────────────────────────
# storage.reap_orphaned_tasks() (called once at startup, see main.py) only
# catches tasks orphaned by a CLEAN restart -- it runs once, at boot. A
# process that dies mid-task without a clean restart (OOM kill, crash)
# leaves its task stuck with no startup event to reap it, possibly for a
# long time until the next deploy. This periodic loop catches that case
# too: anything non-terminal with no update in a while gets marked failed.
# It also carries the usage_log purge (see storage.get_usage) -- no reason
# to run that one every 5 minutes since rows are only stale after a full
# week, so it's gated to once every 12 ticks (~hourly).
_WATCHDOG_INTERVAL = 300     # check every 5 minutes
_STALE_THRESHOLD = 900       # 15 minutes with no progress update = assume it's dead
_USAGE_PURGE_EVERY_N_TICKS = 12

async def _watchdog_loop():
    tick = 0
    while True:
        try:
            stale_ids = await storage.list_stale_tasks(_STALE_THRESHOLD)
            for task_id in stale_ids:
                await storage.update_task(task_id, status="failed",
                    error="No progress for over 15 minutes -- likely interrupted. Please retry.")
        except Exception:
            pass  # best-effort -- a missed sweep just means the next one catches it
        tick += 1
        if tick % _USAGE_PURGE_EVERY_N_TICKS == 0:
            try:
                await storage.purge_old_usage()
            except Exception:
                pass
        await asyncio.sleep(_WATCHDOG_INTERVAL)


def start_watchdog():
    """Called once from main.py's startup event. Returns the asyncio.Task
    so main.py can hold a reference (otherwise nothing keeps it alive --
    an unreferenced background task can get garbage-collected)."""
    return asyncio.create_task(_watchdog_loop())
