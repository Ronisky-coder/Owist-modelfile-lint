# storage.py -- Persistent storage layer, Postgres-only (Neon).
#
# This app no longer supports the ephemeral local-SQLite fallback that used
# to run when DATABASE_URL wasn't set. That path meant every Render
# redeploy silently wiped artifacts, tasks, memory, and subscriptions --
# indistinguishable from data just vanishing. DATABASE_URL is now required;
# the app refuses to start without it rather than quietly degrading to
# storage that doesn't survive a restart.
#
# Every caller elsewhere in this codebase (main.py, tools.py, tasks.py)
# only ever calls the async functions below -- nothing outside this file
# touches asyncpg directly. That boundary is what makes swapping the
# backend (or adding read replicas, etc.) safe later.
import os, time, json, asyncio
import asyncpg

DATABASE_URL = os.environ.get("DATABASE_URL", "")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is required -- this app runs on Postgres (Neon) only now, "
        "the ephemeral SQLite fallback has been removed. Set DATABASE_URL to your "
        "Neon connection string (the DIRECT/unpooled one, not the PgBouncer pooler "
        "endpoint -- this app already pools client-side via asyncpg, and Render runs "
        "it as one persistent process rather than many short-lived serverless ones, "
        "so stacking Neon's own pooler underneath just adds a network hop for nothing)."
    )

_ARTIFACT_TTL = 3600       # 1 hour, unpublished artifacts only
_SESSION_TTL = 7200        # 2 hours idle -- roughly matches e2b's own sandbox ceiling,
                            # no point persisting a session pointer to a sandbox that
                            # e2b will have already reaped on its own

# Supabase/Neon's connection pooler (PgBouncer in "transaction" mode) does NOT
# support server-side prepared statements the way asyncpg uses by default --
# every query gets prepared once and reused, but transaction pooling can hand
# your next query to a different backend connection that's never seen that
# prepared statement, causing real "prepared statement does not exist" errors
# in production. statement_cache_size=0 disables that caching. Harmless (if
# unnecessary) on a direct connection; required if you ever do point this at
# a pooled endpoint instead.
_PG_DSN = DATABASE_URL.split("?")[0]  # strip query params, pass ssl explicitly below
_pool = None
_pool_lock = asyncio.Lock()


async def _get_pool():
    global _pool
    if _pool is not None:
        return _pool
    async with _pool_lock:
        if _pool is None:  # re-check inside the lock -- another coroutine may have won the race
            _pool = await asyncpg.create_pool(
                _PG_DSN, ssl="require", statement_cache_size=0,
                min_size=1, max_size=5, command_timeout=15,
            )
            await _init_schema(_pool)
    return _pool


async def _init_schema(pool):
    async with pool.acquire() as c:
        await c.execute("""CREATE TABLE IF NOT EXISTS artifacts (
            id TEXT PRIMARY KEY,
            media_type TEXT NOT NULL,
            filename TEXT NOT NULL,
            data BYTEA NOT NULL,
            source_markdown TEXT,
            owner_uid TEXT NOT NULL DEFAULT '',
            published BOOLEAN NOT NULL DEFAULT FALSE,
            created DOUBLE PRECISION NOT NULL
        )""")
        # source_markdown lets create_document's ORIGINAL markdown survive
        # alongside the rendered PDF/DOCX bytes, so update_document can
        # revise the real source instead of trying to decompile a PDF.
        # ALTER-guarded so this rolls forward on an existing DB too.
        await c.execute("""DO $$ BEGIN
            ALTER TABLE artifacts ADD COLUMN source_markdown TEXT;
        EXCEPTION WHEN duplicate_column THEN NULL; END $$""")

        await c.execute("""CREATE TABLE IF NOT EXISTS user_memory (
            uid TEXT PRIMARY KEY,
            memory TEXT NOT NULL DEFAULT '',
            updated DOUBLE PRECISION NOT NULL
        )""")
        await c.execute("""CREATE TABLE IF NOT EXISTS subscriptions (
            uid TEXT PRIMARY KEY,
            plan TEXT NOT NULL DEFAULT 'free',
            status TEXT NOT NULL DEFAULT 'inactive',
            phone TEXT NOT NULL DEFAULT '',
            paystack_reference TEXT NOT NULL DEFAULT '',
            current_period_end DOUBLE PRECISION,
            updated DOUBLE PRECISION NOT NULL
        )""")
        await c.execute("""CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            uid TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'heavy',
            request TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            plan TEXT NOT NULL DEFAULT '[]',
            progress_current INTEGER NOT NULL DEFAULT 0,
            progress_total INTEGER NOT NULL DEFAULT 0,
            result TEXT NOT NULL DEFAULT '',
            error TEXT NOT NULL DEFAULT '',
            created DOUBLE PRECISION NOT NULL,
            updated DOUBLE PRECISION NOT NULL
        )""")
        await c.execute("CREATE INDEX IF NOT EXISTS idx_tasks_uid ON tasks(uid)")
        await c.execute("""CREATE TABLE IF NOT EXISTS user_facts (
            id SERIAL PRIMARY KEY,
            uid TEXT NOT NULL,
            fact TEXT NOT NULL,
            created DOUBLE PRECISION NOT NULL
        )""")
        await c.execute("CREATE INDEX IF NOT EXISTS idx_facts_uid ON user_facts(uid)")

        # Cross-turn tool session state -- one row per conversation, keyed by
        # a conversation_id the frontend will send once it's updated to.
        # Lets the sandbox (and "which artifact did we just make") survive
        # into the NEXT message instead of resetting every turn.
        await c.execute("""CREATE TABLE IF NOT EXISTS sessions (
            conversation_id TEXT PRIMARY KEY,
            uid TEXT NOT NULL DEFAULT '',
            sandbox_id TEXT,
            last_image_artifact TEXT,
            last_document_artifact TEXT,
            last_code_artifact TEXT,
            updated DOUBLE PRECISION NOT NULL
        )""")

        # Real usage metering, replacing a client-side-only localStorage
        # counter that had zero server-side enforcement (clearing storage
        # or opening a private window fully reset it). One row per actual
        # request; daily/weekly usage is computed on read as a rolling sum
        # over this log, not a separately-maintained counter that could
        # drift out of sync with reality.
        await c.execute("""CREATE TABLE IF NOT EXISTS usage_log (
            id SERIAL PRIMARY KEY,
            key TEXT NOT NULL,
            units INTEGER NOT NULL DEFAULT 1,
            ts DOUBLE PRECISION NOT NULL
        )""")
        await c.execute("CREATE INDEX IF NOT EXISTS idx_usage_key_ts ON usage_log(key, ts)")


async def store_artifact(aid, data, media_type, filename, owner_uid="", source_markdown=None):
    pool = await _get_pool()
    async with pool.acquire() as c:
        await c.execute(
            "INSERT INTO artifacts (id, media_type, filename, data, source_markdown, owner_uid, published, created) "
            "VALUES ($1,$2,$3,$4,$5,$6,FALSE,$7) "
            "ON CONFLICT (id) DO UPDATE SET media_type=EXCLUDED.media_type, "
            "filename=EXCLUDED.filename, data=EXCLUDED.data, source_markdown=EXCLUDED.source_markdown, "
            "owner_uid=EXCLUDED.owner_uid, created=EXCLUDED.created",
            aid, media_type, filename, data, source_markdown, owner_uid or "", time.time())
    asyncio.create_task(_cleanup_artifacts())  # fire-and-forget, don't block the write


async def get_artifact(aid):
    pool = await _get_pool()
    async with pool.acquire() as c:
        row = await c.fetchrow(
            "SELECT media_type, filename, data, source_markdown, owner_uid, published, created "
            "FROM artifacts WHERE id=$1", aid)
    if not row:
        return None
    # TTL only ever applies to anonymous/guest artifacts (no owner_uid) --
    # anything created by a signed-in user is permanent from the moment it's
    # made, not contingent on them remembering to hit "Publish" within an
    # hour. "Published" now only means "world-shareable link", it no longer
    # doubles as "don't delete this".
    is_owned = bool(row["owner_uid"])
    if not row["published"] and not is_owned and time.time() - row["created"] > _ARTIFACT_TTL:
        async with pool.acquire() as c:
            await c.execute("DELETE FROM artifacts WHERE id=$1", aid)
        return None
    return {"media_type": row["media_type"], "filename": row["filename"], "bytes": bytes(row["data"]),
            "source_markdown": row["source_markdown"], "owner_uid": row["owner_uid"] or "",
            "published": row["published"], "created": row["created"]}


async def list_artifacts_for_user(uid: str, limit: int = 200):
    """Every artifact this account owns, newest first -- this is what makes
    the Artifacts panel a real permanent, cross-device library instead of a
    per-tab JS object. No content bytes here (this can be a long list),
    just enough to render cards and let the client fetch bytes on open."""
    if not uid:
        return []
    pool = await _get_pool()
    async with pool.acquire() as c:
        rows = await c.fetch(
            "SELECT id, media_type, filename, published, created, "
            "(source_markdown IS NOT NULL) AS is_document "
            "FROM artifacts WHERE owner_uid=$1 ORDER BY created DESC LIMIT $2", uid, limit)
    return [dict(r) for r in rows]


async def delete_artifact(aid: str, uid: str) -> bool:
    """Owner-only delete. Returns False (no-op, not an error) if the
    artifact doesn't exist or belongs to someone else -- same
    can't-distinguish-not-found-from-not-yours posture as tasks.py."""
    pool = await _get_pool()
    async with pool.acquire() as c:
        result = await c.execute(
            "DELETE FROM artifacts WHERE id=$1 AND owner_uid=$2 AND owner_uid <> ''", aid, uid)
    return result != "DELETE 0"


async def publish_artifact(aid, requester_uid):
    entry = await get_artifact(aid)
    if not entry:
        return False
    if entry["owner_uid"] and entry["owner_uid"] != requester_uid:
        return False
    pool = await _get_pool()
    async with pool.acquire() as c:
        await c.execute("UPDATE artifacts SET published=TRUE WHERE id=$1", aid)
    return True


async def _cleanup_artifacts():
    try:
        pool = await _get_pool()
        cutoff = time.time() - _ARTIFACT_TTL
        async with pool.acquire() as c:
            # owner_uid='' catches BOTH true guests and the old rows created
            # before ownership was threaded through everywhere -- anything
            # with a real owner is permanent and this never touches it.
            await c.execute(
                "DELETE FROM artifacts WHERE created < $1 AND published=FALSE AND owner_uid=''", cutoff)
    except Exception:
        pass  # best-effort background cleanup -- never let this break a write


async def get_user_memory(uid):
    if not uid:
        return ""
    pool = await _get_pool()
    async with pool.acquire() as c:
        row = await c.fetchrow("SELECT memory FROM user_memory WHERE uid=$1", uid)
    return row["memory"] if row else ""


async def set_user_memory(uid, memory):
    if not uid:
        return
    pool = await _get_pool()
    async with pool.acquire() as c:
        await c.execute(
            "INSERT INTO user_memory (uid, memory, updated) VALUES ($1,$2,$3) "
            "ON CONFLICT (uid) DO UPDATE SET memory=EXCLUDED.memory, updated=EXCLUDED.updated",
            uid, memory[:8000], time.time())


async def add_user_fact(uid, fact):
    if not uid or not fact:
        return
    pool = await _get_pool()
    async with pool.acquire() as c:
        await c.execute("INSERT INTO user_facts (uid, fact, created) VALUES ($1,$2,$3)",
                         uid, fact[:500], time.time())


async def get_user_facts(uid, limit=40):
    if not uid:
        return []
    pool = await _get_pool()
    async with pool.acquire() as c:
        rows = await c.fetch("SELECT fact FROM user_facts WHERE uid=$1 ORDER BY created DESC LIMIT $2",
                              uid, limit)
    return [r["fact"] for r in rows]


async def get_subscription(uid):
    if not uid:
        return {"plan": "free", "status": "inactive", "current_period_end": None}
    pool = await _get_pool()
    async with pool.acquire() as c:
        row = await c.fetchrow("SELECT plan, status, phone, current_period_end FROM subscriptions WHERE uid=$1", uid)
    if not row:
        return {"plan": "free", "status": "inactive", "current_period_end": None}
    return {"plan": row["plan"], "status": row["status"], "phone": row["phone"],
            "current_period_end": row["current_period_end"]}


async def set_subscription(uid, plan, status, current_period_end=None, phone="", paystack_reference=""):
    pool = await _get_pool()
    async with pool.acquire() as c:
        await c.execute(
            "INSERT INTO subscriptions (uid, plan, status, phone, paystack_reference, current_period_end, updated) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7) ON CONFLICT (uid) DO UPDATE SET "
            "plan=EXCLUDED.plan, status=EXCLUDED.status, phone=EXCLUDED.phone, "
            "paystack_reference=EXCLUDED.paystack_reference, current_period_end=EXCLUDED.current_period_end, updated=EXCLUDED.updated",
            uid, plan, status, phone, paystack_reference, current_period_end, time.time())


# ── TASKS ────────────────────────────────────────────────────────────
# Non-terminal statuses a task can be "stuck" in if the process that was
# running it dies. Anything in this set that never reaches completed/failed
# on its own is a candidate for reaping.
_NONTERMINAL_TASK_STATUSES = ("queued", "planning", "executing", "verifying")

async def create_task_row(task_id, uid, mode, request):
    pool = await _get_pool()
    now = time.time()
    async with pool.acquire() as c:
        await c.execute(
            "INSERT INTO tasks (id, uid, mode, request, status, created, updated) VALUES ($1,$2,$3,$4,'queued',$5,$5)",
            task_id, uid, mode, request, now)


async def get_task(task_id):
    pool = await _get_pool()
    async with pool.acquire() as c:
        row = await c.fetchrow("SELECT * FROM tasks WHERE id=$1", task_id)
    return dict(row) if row else None


async def list_tasks(uid, limit=20):
    pool = await _get_pool()
    async with pool.acquire() as c:
        rows = await c.fetch("SELECT * FROM tasks WHERE uid=$1 ORDER BY created DESC LIMIT $2", uid, limit)
    return [dict(r) for r in rows]


async def update_task(task_id, **fields):
    if not fields:
        return
    fields["updated"] = time.time()
    pool = await _get_pool()
    cols = list(fields.keys())
    set_clause = ", ".join(f"{col}=${i+2}" for i, col in enumerate(cols))
    async with pool.acquire() as c:
        await c.execute(f"UPDATE tasks SET {set_clause} WHERE id=$1", task_id, *[fields[c2] for c2 in cols])


async def delete_task(task_id, uid):
    pool = await _get_pool()
    async with pool.acquire() as c:
        result = await c.execute("DELETE FROM tasks WHERE id=$1 AND uid=$2", task_id, uid)
    return result != "DELETE 0"


async def reap_orphaned_tasks() -> int:
    """Called once at app startup. Any task still in a non-terminal status
    was, by definition, being run by a PREVIOUS process instance that no
    longer exists -- this process just booted, so nothing is actually
    working on it. Left alone these sit stuck forever, indistinguishable
    from having silently vanished. Mark them failed with a clear, honest
    reason instead, so a poller sees a real terminal state and the user can
    just retry. Returns how many were reaped (worth logging)."""
    pool = await _get_pool()
    placeholders = ",".join(f"'{s}'" for s in _NONTERMINAL_TASK_STATUSES)  # fixed internal set, not user input
    async with pool.acquire() as c:
        result = await c.execute(
            f"UPDATE tasks SET status='failed', "
            f"error='Interrupted by a server restart before this finished -- please retry.', "
            f"updated=$1 WHERE status IN ({placeholders})", time.time())
    try:
        return int(result.split()[-1])
    except Exception:
        return 0


async def list_stale_tasks(older_than_seconds: float = 900):
    """Tasks that are non-terminal AND haven't been updated recently --
    catches a process that crashed without a clean restart (so the startup
    reaper never got a chance to run for it). Used by a periodic watchdog
    in tasks.py, not by the startup reaper above."""
    pool = await _get_pool()
    cutoff = time.time() - older_than_seconds
    placeholders = ",".join(f"'{s}'" for s in _NONTERMINAL_TASK_STATUSES)
    async with pool.acquire() as c:
        rows = await c.fetch(
            f"SELECT id FROM tasks WHERE status IN ({placeholders}) AND updated < $1", cutoff)
    return [r["id"] for r in rows]


# ── CROSS-TURN TOOL SESSIONS ───────────────────────────────────────────
async def get_session(conversation_id: str):
    if not conversation_id:
        return None
    pool = await _get_pool()
    async with pool.acquire() as c:
        row = await c.fetchrow("SELECT * FROM sessions WHERE conversation_id=$1", conversation_id)
    if not row:
        return None
    if time.time() - row["updated"] > _SESSION_TTL:
        return None  # stale -- treat as if it never existed, a fresh one gets written on save
    return dict(row)


async def save_session(conversation_id: str, uid: str = "", **fields):
    if not conversation_id:
        return
    fields["updated"] = time.time()
    fields["uid"] = uid
    pool = await _get_pool()
    cols = list(fields.keys())
    col_list = ", ".join(cols)
    placeholders = ", ".join(f"${i+2}" for i in range(len(cols)))
    update_clause = ", ".join(f"{c2}=EXCLUDED.{c2}" for c2 in cols)
    async with pool.acquire() as c:
        await c.execute(
            f"INSERT INTO sessions (conversation_id, {col_list}) VALUES ($1, {placeholders}) "
            f"ON CONFLICT (conversation_id) DO UPDATE SET {update_clause}",
            conversation_id, *[fields[c2] for c2 in cols])


async def delete_stale_sessions(older_than_seconds: float = _SESSION_TTL) -> int:
    pool = await _get_pool()
    cutoff = time.time() - older_than_seconds
    async with pool.acquire() as c:
        result = await c.execute("DELETE FROM sessions WHERE updated < $1", cutoff)
    try:
        return int(result.split()[-1])
    except Exception:
        return 0


# ── USAGE METERING ──────────────────────────────────────────────────
# `key` is uid for signed-in users, or an IP-derived key for guests (see
# main.py's usage_key()) -- this module doesn't care which, it just sums
# whatever key it's given over a rolling window.
_WEEK_SECONDS = 7 * 86400

async def log_usage(key: str, units: int = 1):
    pool = await _get_pool()
    async with pool.acquire() as c:
        await c.execute("INSERT INTO usage_log (key, units, ts) VALUES ($1,$2,$3)",
                         key, units, time.time())


async def get_usage(key: str) -> dict:
    """Real rolling sums, computed fresh from the log every time -- not a
    separately-maintained counter that could drift from what actually
    happened. Returns raw used-units for both windows; main.py turns
    these into percentages against whatever the current caps are, so
    caps can change without a data migration."""
    pool = await _get_pool()
    now = time.time()
    async with pool.acquire() as c:
        daily = await c.fetchval(
            "SELECT COALESCE(SUM(units),0) FROM usage_log WHERE key=$1 AND ts > $2",
            key, now - 86400)
        weekly = await c.fetchval(
            "SELECT COALESCE(SUM(units),0) FROM usage_log WHERE key=$1 AND ts > $2",
            key, now - _WEEK_SECONDS)
    return {"daily_used": int(daily or 0), "weekly_used": int(weekly or 0)}


async def purge_old_usage(older_than_seconds: float = _WEEK_SECONDS + 3600) -> int:
    """Nothing needs rows older than the weekly window once it's passed --
    called periodically (see tasks.py's watchdog loop) so this table
    doesn't grow forever."""
    pool = await _get_pool()
    cutoff = time.time() - older_than_seconds
    async with pool.acquire() as c:
        result = await c.execute("DELETE FROM usage_log WHERE ts < $1", cutoff)
    try:
        return int(result.split()[-1])
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════════
# SHARED
# ═══════════════════════════════════════════════════════════════════════
async def get_billing_status(uid: str) -> dict:
    """The real 'count and alert' logic: computed on read (not a scheduled
    job that could silently fail) so it's always accurate the moment
    someone checks. Returns enough for the frontend to show a banner when
    a paid plan is about to lapse."""
    sub = await get_subscription(uid)
    end = sub.get("current_period_end")
    days_remaining = None
    expiring_soon = False
    if end:
        days_remaining = max(0, round((end - time.time()) / 86400, 1))
        expiring_soon = sub.get("status") == "active" and days_remaining <= 3
        if days_remaining <= 0 and sub.get("status") == "active":
            sub["status"] = "expired"
    return {**sub, "days_remaining": days_remaining, "expiring_soon": expiring_soon}


async def compose_memory(uid: str) -> str:
    """What actually gets injected into the system prompt: explicit
    memory blob + the running list of learned facts, newest first."""
    parts = []
    mem = await get_user_memory(uid)
    if mem:
        parts.append(mem.strip())
    facts = await get_user_facts(uid)
    if facts:
        parts.append("Known facts about this user:\n- " + "\n- ".join(facts))
    return "\n\n".join(parts)
