import os, json, base64, time, uuid
from typing import List, Literal, Optional
from fastapi import FastAPI, Request, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from monitoring import get_stats
import storage
import billing
import tasks
from auth import get_uid_optional, require_uid
from router import route_chat_stream, generate_title, generate_image
from providers import groq_tts, groq_stt, run_code_e2b, ProviderError, generate_pdf, generate_docx, generate_zip

ALLOWED = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://openwist.kesug.com,https://openwist.com,http://localhost:5500,http://127.0.0.1:5500"
).split(",")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")

app = FastAPI(title="Openwist AI API v3")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED,
    allow_methods=["GET","POST","OPTIONS"], allow_headers=["*"])

_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://www.gstatic.com "
    "https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data: blob: https://image.pollinations.ai https://fal.media "
    "https://v3.fal.media https://*.fal.media https://lh3.googleusercontent.com "
    "https://www.gstatic.com; "
    "connect-src 'self' https://openwist-poc.onrender.com "
    "https://identitytoolkit.googleapis.com https://securetoken.googleapis.com "
    "https://firestore.googleapis.com https://api.allorigins.win "
    "wss://*.firebaseio.com; "
    "frame-src 'self' blob:; "
    "object-src 'none'; "
    "base-uri 'self';"
)

@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = _CSP
    return response

@app.on_event("startup")
async def _on_startup():
    # Any task still non-terminal at boot was being run by a PREVIOUS
    # process instance that no longer exists (redeploy, crash, restart) --
    # this process just started, nothing is actually working on it. Reap
    # those into a clean 'failed' state instead of leaving them stuck
    # forever looking like they silently vanished. The periodic watchdog
    # in tasks.py catches the same situation for a mid-task crash that
    # never triggers a clean restart (this startup hook only ever runs once).
    try:
        reaped = await storage.reap_orphaned_tasks()
        if reaped:
            print(f"[startup] reaped {reaped} task(s) orphaned by a previous process")
    except Exception as e:
        print(f"[startup] task reap failed (non-fatal): {e}")
    app.state.watchdog_task = tasks.start_watchdog()

# ── MODELS ────────────────────────────────────────────────────────────
class Msg(BaseModel):
    role: Literal["user","assistant"]
    content: str
    image_data: Optional[str] = None
    image_mime: Optional[str] = None

class ChatReq(BaseModel):
    message: str
    mode: Literal["auto","calm","code","heavy","fun"] = "calm"
    effort: Literal["medium","high","extra"] = "medium"  # <--- NEW: real reasoning_effort on gpt-oss models + chain-priority boost, see router.py
    history: List[Msg] = []
    search: bool = False
    image_data: Optional[str] = None
    image_mime: Optional[str] = None
    user_name: Optional[str] = ""
    user_memory: Optional[str] = ""
    force_tools: bool = False  # <--- NEW: allows bridge to force tool usage
    conversation_id: Optional[str] = None  # <--- NEW: enables cross-turn sandbox/artifact persistence when the frontend sends a stable id per thread; omitted = old per-turn-only behavior, fully backward compatible

class TitleReq(BaseModel):
    message: str

class TTSReq(BaseModel):
    text: str
    voice: str = "autumn"

class STTReq(BaseModel):
    audio_b64: str
    filename: str = "audio.webm"

class CodeReq(BaseModel):
    code: str
    language: str = "python"

class ImageReq(BaseModel):
    prompt: str

class SearchReq(BaseModel):
    query: str

class DocReq(BaseModel):
    title: str = "Document"
    content: str
    format: Literal["pdf", "docx"] = "pdf"

class ZipFile(BaseModel):
    name: str
    content: str

class ZipReq(BaseModel):
    files: List[ZipFile]
    archive_name: str = "openwist-files"

class MemoryReq(BaseModel):
    memory: str = ""

class CheckoutReq(BaseModel):
    plan: str = "pro_monthly"
    callback_url: str = "https://openwist.kesug.com/beta/"

class MobileChargeReq(BaseModel):
    plan: str = "pro_monthly"
    phone: str
    provider: str

class TaskReq(BaseModel):
    request: str
    mode: str = "heavy"

# ── USAGE LIMITS ─────────────────────────────────────────────────────
# Real, server-enforced daily + weekly caps -- replaces a client-side-only
# localStorage counter (200 credits/8h) that had zero actual enforcement;
# clearing storage or opening a private window fully reset it. Same
# per-mode weighting that system used (heavier modes cost more), just
# measured and blocked here instead of trusted to the browser. Numbers are
# starting defaults carried over from the old flat-200 baseline -- tune via
# env vars once there's real cost data, not fixed forever.
MODE_UNIT_COST = {"calm": 1, "code": 3, "heavy": 8, "fun": 1}
TASK_UNIT_COST = 8  # a background task plans + runs multiple steps -- charged like heavy mode, not metered per internal step
DAILY_LIMIT_UID = int(os.environ.get("DAILY_LIMIT_UID", "200"))
WEEKLY_LIMIT_UID = int(os.environ.get("WEEKLY_LIMIT_UID", "1000"))
DAILY_LIMIT_GUEST = int(os.environ.get("DAILY_LIMIT_GUEST", "20"))
# A trusted bridge service (currently the WhatsApp/gowa bridge) can identify
# a specific end user instead of every one of its users sharing one
# anonymous-guest bucket keyed off the bridge server's own outbound IP --
# that was silently exhausted by ordinary single-person use and, once hit,
# indistinguishable from a real limit (the actual cause of the WhatsApp bot
# defaulting to "Sorry, I couldn't process your message" on every turn: a
# limit_reached event with nothing left to show for it). Real per-identity
# limits, distinct from the anonymous-web-guest tier since these ARE
# individually identified, just not signed in via Firebase.
DAILY_LIMIT_WHATSAPP = int(os.environ.get("DAILY_LIMIT_WHATSAPP", "60"))
WEEKLY_LIMIT_WHATSAPP = int(os.environ.get("WEEKLY_LIMIT_WHATSAPP", "300"))
BRIDGE_AUTH_SECRET = os.environ.get("BRIDGE_AUTH_SECRET", "")

def usage_key(request: Request, uid: str) -> str:
    """uid for signed-in web users. A trusted bridge can claim an identity
    via X-Client-Key (e.g. 'wa:2547...'), honored only when X-Bridge-Secret
    matches BRIDGE_AUTH_SECRET -- unset by default, so this does nothing
    until deliberately configured, and an unproven claim never bypasses the
    IP-based fallback. Any other unauthenticated caller falls back to IP --
    imperfect (a shared/NAT'd IP shares a limit) but real, unlike having no
    server-side guest limit at all. X-Forwarded-For is checked first since
    Render (like most PaaS) proxies requests -- request.client.host alone
    would report the proxy's own address, not the real client."""
    if uid:
        return f"u:{uid}"
    client_key = request.headers.get("x-client-key", "")
    bridge_secret = request.headers.get("x-bridge-secret", "")
    if client_key and BRIDGE_AUTH_SECRET and bridge_secret == BRIDGE_AUTH_SECRET:
        return client_key
    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")
    return f"g:{ip}"

async def usage_status(request: Request, uid: str) -> dict:
    key = usage_key(request, uid)
    used = await storage.get_usage(key)
    if uid:
        daily_limit, weekly_limit = DAILY_LIMIT_UID, WEEKLY_LIMIT_UID
    elif key.startswith("wa:"):
        daily_limit, weekly_limit = DAILY_LIMIT_WHATSAPP, WEEKLY_LIMIT_WHATSAPP
    else:
        daily_limit, weekly_limit = DAILY_LIMIT_GUEST, None  # anonymous IP-based guests aren't tracked weekly -- too noisy (shared/rotating IPs) to mean anything
    daily_pct = min(100, round(100 * used["daily_used"] / daily_limit)) if daily_limit else 0
    weekly_pct = min(100, round(100 * used["weekly_used"] / weekly_limit)) if weekly_limit else 0
    blocked = used["daily_used"] >= daily_limit or (weekly_limit and used["weekly_used"] >= weekly_limit)
    return {
        "daily_used": used["daily_used"], "daily_limit": daily_limit, "daily_pct": daily_pct,
        "weekly_used": used["weekly_used"], "weekly_limit": weekly_limit, "weekly_pct": weekly_pct,
        "blocked": bool(blocked),
    }

@app.get("/api/usage")
async def usage_endpoint(request: Request, uid: str = Depends(get_uid_optional)):
    return await usage_status(request, uid)

# ── HEALTH ────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status":"ok","version":"3.0"}

CURRENT_APP_VERSION = "2.7.0"
CURRENT_APP_MESSAGE = "New Auto mode picks the right persona for you automatically. Real Effort control (Medium/High/Extra) actually changes model reasoning depth, not just a label. Real server-enforced daily/weekly usage limits replace the old client-side credit counter, which had no actual enforcement."
CURRENT_APP_REQUIRED = False

@app.get("/api/version")
async def version():
    return {
        "version": CURRENT_APP_VERSION,
        "message": CURRENT_APP_MESSAGE,
        "required": CURRENT_APP_REQUIRED,
    }

@app.get("/api/stats")
async def stats():
    return get_stats()

# ── CHAT STREAM ────────────────────────────────────────────────────────
@app.post("/api/chat/stream")
@limiter.limit("30/minute")
async def chat_stream(request: Request, req: ChatReq, uid: str = Depends(get_uid_optional)):
    hist = [m.dict() for m in req.history]
    server_mem = await storage.compose_memory(uid) if uid else ""
    combined_memory = "\n\n".join(p for p in [server_mem, req.user_memory or ""] if p).strip()

    # Real limit check -- BEFORE any model call, so a blocked user never
    # costs anything and nothing else in the app is affected, just this.
    status = await usage_status(request, uid)
    if status["blocked"]:
        async def blocked_gen():
            yield f"data: {json.dumps({'type': 'limit_reached', **status})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(blocked_gen(), media_type="text/event-stream",
            headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no","Connection":"keep-alive"})

    async def generate():
        try:
            logged = False
            async for chunk in route_chat_stream(
                message=req.message, mode=req.mode, history=hist,
                use_search=req.search,
                image_data=req.image_data, image_mime=req.image_mime,
                user_name=req.user_name or "", user_memory=combined_memory, uid=uid,
                force_tools=req.force_tools, conversation_id=req.conversation_id or "",
                effort=req.effort or "medium"):
                if not logged and chunk.get("type") == "meta":
                    # Charged on the RESOLVED mode (chunk['mode'], set by
                    # router.py) -- for "auto" requests that's whatever the
                    # classifier actually picked, not a guess made here
                    # that could disagree with it. Only charged once we
                    # know the request actually produced a real response
                    # (a meta event exists) -- a total failure isn't billed.
                    logged = True
                    resolved_mode = chunk.get("mode") or req.mode
                    cost = MODE_UNIT_COST.get(resolved_mode, 1)
                    try:
                        await storage.log_usage(usage_key(request, uid), cost)
                    except Exception:
                        pass  # never let usage logging break the actual response
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','content':str(e)})}\n\n"
            yield "data: [DONE]\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream",
        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no","Connection":"keep-alive"})

# ── MEMORY ────────────────────────────────────────────────────────────
def _check_self(url_uid: str, verified_uid: str):
    if url_uid != verified_uid:
        raise HTTPException(403, "Token does not match this account")

@app.get("/api/memory/{uid}")
async def get_memory(uid: str, verified_uid: str = Depends(require_uid)):
    _check_self(uid, verified_uid)
    return {"memory": await storage.get_user_memory(uid), "facts": await storage.get_user_facts(uid)}

@app.post("/api/memory/{uid}")
async def set_memory(uid: str, req: MemoryReq, verified_uid: str = Depends(require_uid)):
    _check_self(uid, verified_uid)
    await storage.set_user_memory(uid, req.memory)
    return {"ok": True}

@app.delete("/api/memory/{uid}")
async def clear_memory(uid: str, verified_uid: str = Depends(require_uid)):
    _check_self(uid, verified_uid)
    await storage.set_user_memory(uid, "")
    return {"ok": True}

# ── TITLE ─────────────────────────────────────────────────────────────
@app.post("/api/title")
async def title(req: TitleReq):
    return {"title": await generate_title(req.message)}

# ── TTS ──────────────────────────────────────────────────────────────
@app.post("/api/tts")
@limiter.limit("20/minute")
async def tts(request: Request, req: TTSReq):
    try:
        audio_bytes = await groq_tts(req.text, req.voice)
        return Response(content=audio_bytes, media_type="audio/wav",
            headers={"Content-Disposition":"inline","Cache-Control":"no-cache"})
    except ProviderError as e:
        raise HTTPException(502, str(e))

# ── STT ──────────────────────────────────────────────────────────────
@app.post("/api/stt")
@limiter.limit("20/minute")
async def stt(request: Request, req: STTReq):
    try:
        audio_bytes = base64.b64decode(req.audio_b64)
        text = await groq_stt(audio_bytes, req.filename)
        return {"text": text}
    except ProviderError as e:
        raise HTTPException(502, str(e))

# ── IMAGE GENERATION ──────────────────────────────────────────────────
@app.post("/api/generate-image")
@limiter.limit("10/minute")
async def gen_image(request: Request, req: ImageReq, uid: str = Depends(get_uid_optional)):
    from tools import generate_and_store_image
    try:
        result = await generate_and_store_image(req.prompt, uid=uid)
        return result
    except ProviderError as e:
        raise HTTPException(502, str(e))

# ── CODE SANDBOX ──────────────────────────────────────────────────────
@app.post("/api/run-code")
@limiter.limit("15/minute")
async def run_code(request: Request, req: CodeReq):
    try:
        result = await run_code_e2b(req.code, req.language)
        return result
    except ProviderError as e:
        raise HTTPException(502, str(e))

# ── SEARCH ─────────────────────────────────────────────────────────────
@app.post("/api/search")
@limiter.limit("20/minute")
async def search(request: Request, req: SearchReq):
    from providers import ddg_search
    results = await ddg_search(req.query)
    return {"results": results, "query": req.query}

# ── DOCUMENT GENERATION ──────────────────────────────────────────────
@app.post("/api/create-doc")
@limiter.limit("15/minute")
async def create_doc(request: Request, req: DocReq, uid: str = Depends(get_uid_optional)):
    try:
        if req.format == "pdf":
            data = generate_pdf(req.title, req.content)
            media_type = "application/pdf"
            ext = "pdf"
        else:
            data = generate_docx(req.title, req.content)
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ext = "docx"
        safe_name = "".join(c for c in req.title if c.isalnum() or c in " -_").strip() or "document"
        # This endpoint used to just stream bytes back and forget them --
        # every doc made this way (outside the agentic tool loop, e.g. from
        # Smart Study) was invisible to the Artifacts library forever, even
        # for signed-in users. Persist it the same way tools.py does.
        aid = uuid.uuid4().hex[:16]
        await storage.store_artifact(aid, data, media_type, f"{safe_name}.{ext}", uid, req.content)
        return Response(content=data, media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{safe_name}.{ext}"',
                     "X-Artifact-Id": aid})
    except Exception as e:
        raise HTTPException(500, f"Document generation failed: {e}")

@app.post("/api/create-zip")
@limiter.limit("15/minute")
async def create_zip(request: Request, req: ZipReq, uid: str = Depends(get_uid_optional)):
    try:
        files = [{"name": f.name, "content": f.content} for f in req.files]
        data = generate_zip(files)
        safe_name = "".join(c for c in req.archive_name if c.isalnum() or c in " -_").strip() or "files"
        aid = uuid.uuid4().hex[:16]
        await storage.store_artifact(aid, data, "application/zip", f"{safe_name}.zip", uid)
        return Response(content=data, media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{safe_name}.zip"',
                     "X-Artifact-Id": aid})
    except Exception as e:
        raise HTTPException(500, f"ZIP generation failed: {e}")

# ── ARTIFACTS ──────────────────────────────────────────────────────────
@app.get("/api/artifacts")
async def list_artifacts(uid: str = Depends(require_uid)):
    """The user's permanent artifact library -- every doc/image/code
    file/zip this account has ever produced, across every device, since
    it's just a read against their owner_uid in Postgres. Signed-in only:
    a guest has no account-level identity to attach a library to."""
    return {"artifacts": await storage.list_artifacts_for_user(uid)}

@app.delete("/api/artifact/{artifact_id}")
async def delete_artifact_endpoint(artifact_id: str, uid: str = Depends(require_uid)):
    ok = await storage.delete_artifact(artifact_id, uid)
    if not ok:
        raise HTTPException(404, "Artifact not found or you don't own it")
    return {"ok": True}

@app.get("/api/artifact/{artifact_id}")
async def get_artifact(artifact_id: str, uid: str = Depends(get_uid_optional)):
    from tools import get_artifact as _get
    entry = await _get(artifact_id)
    if not entry:
        raise HTTPException(404, "Artifact not found or expired (artifacts expire after 1 hour unless published)")
    owner = entry.get("owner_uid") or ""
    if owner and owner != uid and not entry.get("published"):
        raise HTTPException(403, "This file belongs to a different account")
    # Content-Disposition:attachment was being sent unconditionally -- correct
    # for a .zip/.docx/.py the user wants to save, but it tells the browser
    # "this is a download, not something to render," which is exactly why an
    # <img src=...> pointed at this endpoint could sit on a loading skeleton
    # forever: the request succeeds, but the browser won't paint it inline
    # because the response says not to. Images (and PDFs, which people expect
    # to preview) get 'inline' instead; real downloadable files keep 'attachment'.
    is_inline = entry["media_type"].startswith("image/") or entry["media_type"] == "application/pdf"
    disposition = "inline" if is_inline else "attachment"
    return Response(content=entry["bytes"], media_type=entry["media_type"],
        headers={"Content-Disposition": f'{disposition}; filename="{entry["filename"]}"'})

@app.post("/api/artifact/{artifact_id}/publish")
async def publish_artifact_endpoint(artifact_id: str, uid: str = Depends(get_uid_optional)):
    ok = await storage.publish_artifact(artifact_id, uid)
    if not ok:
        raise HTTPException(403, "Only the owner can publish this file")
    return {"ok": True, "url": f"/api/artifact/{artifact_id}"}

# ── ADMIN ──────────────────────────────────────────────────────────────
def _check_admin(token: str = ""):
    if not ADMIN_TOKEN:
        raise HTTPException(503, "Admin endpoints are disabled (ADMIN_TOKEN not set)")
    if not token or token != ADMIN_TOKEN:
        raise HTTPException(401, "Unauthorized")

@app.get("/api/admin/stats")
async def admin_stats(x_admin_token: str = Header(default="")):
    _check_admin(x_admin_token)
    return get_stats()

@app.post("/api/admin/suspend")
async def admin_suspend(request: Request, x_admin_token: str = Header(default="")):
    _check_admin(x_admin_token)
    data = await request.json()
    uid = data.get("uid")
    duration = data.get("duration","24h")
    reason = data.get("reason","Terms of Service violation")
    if not uid:
        raise HTTPException(400, "uid required")
    import time
    until = None
    if duration == "24h":
        until = int(time.time()*1000) + 86400000
    return {"success":True,"uid":uid,"suspended":True,"until":until,"reason":reason}

@app.post("/api/admin/unsuspend")
async def admin_unsuspend(request: Request, x_admin_token: str = Header(default="")):
    _check_admin(x_admin_token)
    data = await request.json()
    uid = data.get("uid")
    if not uid:
        raise HTTPException(400, "uid required")
    return {"success":True,"uid":uid,"suspended":False}

# ── BILLING ────────────────────────────────────────────────────────────
@app.get("/api/billing/status")
async def billing_status(uid: str = Depends(require_uid)):
    return await storage.get_billing_status(uid)

@app.post("/api/billing/checkout")
@limiter.limit("10/minute")
async def billing_checkout(request: Request, req: CheckoutReq, uid: str = Depends(require_uid)):
    try:
        data = await billing.init_card_checkout(email=f"{uid}@users.openwist.kesug.com",
                                                  plan=req.plan, callback_url=req.callback_url)
        return {"ok": True, **data}
    except billing.BillingError as e:
        raise HTTPException(503, str(e))

@app.post("/api/billing/mobile-charge")
@limiter.limit("10/minute")
async def billing_mobile_charge(request: Request, req: MobileChargeReq, uid: str = Depends(require_uid)):
    try:
        data = await billing.init_mobile_money_charge(email=f"{uid}@users.openwist.kesug.com",
                                                        plan=req.plan, phone=req.phone, provider=req.provider)
        return {"ok": True, **data}
    except billing.BillingError as e:
        raise HTTPException(503, str(e))

@app.post("/api/billing/webhook")
@limiter.limit("60/minute")
async def billing_webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("x-paystack-signature", "")
    if not billing.verify_webhook_signature(body, signature):
        raise HTTPException(403, "Invalid signature")
    event = json.loads(body)
    if event.get("event") == "charge.success":
        data = event.get("data", {})
        plan = (data.get("metadata") or {}).get("plan", "pro_monthly")
        email = data.get("customer", {}).get("email", "")
        uid = email.split("@")[0] if email.endswith("@users.openwist.kesug.com") else ""
        if uid:
            interval_days = billing.PLANS.get(plan, {}).get("interval_days", 30)
            await storage.set_subscription(uid, plan, "active",
                current_period_end=time.time() + interval_days * 86400,
                paystack_reference=data.get("reference", ""))
    return {"ok": True}

# ── TASKS ──────────────────────────────────────────────────────────────
@app.post("/api/tasks")
@limiter.limit("10/minute")
async def create_task_endpoint(request: Request, req: TaskReq, uid: str = Depends(require_uid)):
    if not req.request.strip():
        raise HTTPException(400, "Empty request")
    status = await usage_status(request, uid)
    if status["blocked"]:
        raise HTTPException(429, "Usage limit reached -- see /api/usage for reset timing.")
    task_id = await tasks.create_task(uid, req.request.strip(), req.mode)
    try:
        await storage.log_usage(usage_key(request, uid), TASK_UNIT_COST)
    except Exception:
        pass
    return {"id": task_id, "status": "queued"}

@app.get("/api/tasks/{task_id}")
async def get_task_endpoint(task_id: str, uid: str = Depends(require_uid)):
    task = await tasks.get_task_for_user(task_id, uid)
    if not task:
        raise HTTPException(404, "Task not found")
    return task

@app.get("/api/tasks")
async def list_tasks_endpoint(uid: str = Depends(require_uid)):
    return {"tasks": await tasks.list_tasks_for_user(uid)}

@app.delete("/api/tasks/{task_id}")
async def cancel_task_endpoint(task_id: str, uid: str = Depends(require_uid)):
    ok = await tasks.cancel_task(task_id, uid)
    if not ok:
        raise HTTPException(404, "Task not found")
    return {"ok": True}

# ── WHATSAPP WEBHOOK STUB (kept for legacy) ──────────────────────────
@app.get("/api/webhook/whatsapp")
async def whatsapp_verify(request: Request):
    params = dict(request.query_params)
    verify_token = os.environ.get("WHATSAPP_VERIFY_TOKEN", "")
    if not verify_token:
        raise HTTPException(503, "WhatsApp webhook not configured")
    if (params.get("hub.mode") == "subscribe" and
            params.get("hub.verify_token") == verify_token):
        return Response(content=params.get("hub.challenge",""), media_type="text/plain")
    raise HTTPException(403, "Verification failed")

@app.post("/api/webhook/whatsapp")
@limiter.limit("60/minute")
async def whatsapp_message(request: Request):
    return {"status":"ok","note":"Use new webhook at /webhook on bridge service"}
