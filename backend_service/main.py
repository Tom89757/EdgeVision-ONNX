"""EdgeVision-ONNX 阶段 4：轻量端云联动 FastAPI 服务。"""
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

try:
    from .schemas import (
        DailyReportEnvelope,
        DailyReportOut,
        DailyReportRequest,
        ErrorEnvelope,
        PetEventEnvelope,
        PetEventIn,
        PetEventOut,
        PetEventPage,
        PetEventPageEnvelope,
    )
except ImportError:  # 支持在 backend_service 目录内执行 uvicorn main:app
    from schemas import (  # type: ignore
        DailyReportEnvelope,
        DailyReportOut,
        DailyReportRequest,
        ErrorEnvelope,
        PetEventEnvelope,
        PetEventIn,
        PetEventOut,
        PetEventPage,
        PetEventPageEnvelope,
    )

LOGGER = logging.getLogger("edgevision.backend")
DB_PATH = Path(os.getenv("EDGEVISION_DB_PATH", Path(__file__).with_name("events.db")))
MAX_REQUEST_BYTES = int(os.getenv("MAX_REQUEST_BYTES", "524288"))


def _utc_iso(value: datetime) -> str:
    """统一为 UTC ISO-8601，保证 SQLite 文本时间可直接比较。"""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _row_to_event(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "thumbnail": row["thumbnail"],
        "category": row["category"],
        "confidence": row["confidence"],
        "timestamp": _parse_utc(row["event_time"]),
        "device_id": row["device_id"],
        "created_at": _parse_utc(row["created_at"]),
    }


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    """每次请求独立连接，规避 FastAPI 线程池与 SQLite 的跨线程限制。"""
    connection = sqlite3.connect(str(DB_PATH), timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with db() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS pet_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                idempotency_key TEXT UNIQUE,
                device_id TEXT NOT NULL,
                category TEXT NOT NULL,
                confidence REAL NOT NULL,
                thumbnail TEXT NOT NULL,
                event_time TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_pet_events_time ON pet_events(event_time);
            CREATE INDEX IF NOT EXISTS idx_pet_events_category ON pet_events(category);
            """
        )


async def enforce_body_limit(request: Request) -> None:
    """依赖级请求体上限；先看 Content-Length，再对无长度请求做兜底。"""
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            length = int(declared)
        except ValueError as exc:
            raise HTTPException(400, detail={"code": "INVALID_CONTENT_LENGTH", "message": "Content-Length 非法"}) from exc
        if length > MAX_REQUEST_BYTES:
            raise HTTPException(
                413,
                detail={"code": "PAYLOAD_TOO_LARGE", "message": "请求体超过上限", "data": {"max_bytes": MAX_REQUEST_BYTES}},
            )
    body = await request.body()
    if len(body) > MAX_REQUEST_BYTES:
        raise HTTPException(
            413,
            detail={"code": "PAYLOAD_TOO_LARGE", "message": "请求体超过上限", "data": {"max_bytes": MAX_REQUEST_BYTES}},
        )


app = FastAPI(title="EdgeVision ONNX 端云联动服务", version="1.0.0", dependencies=[Depends(enforce_body_limit)])


def _assert_same_event(row: sqlite3.Row, payload: PetEventIn, event_time: str) -> None:
    if (
        row["device_id"] != payload.device_id
        or row["category"] != payload.category
        or row["confidence"] != payload.confidence
        or row["event_time"] != event_time
        or row["thumbnail"] != payload.thumbnail
    ):
        raise HTTPException(
            409,
            detail={"code": "IDEMPOTENCY_CONFLICT", "message": "同一 Idempotency-Key 对应了不同事件内容"},
        )


@app.post("/api/v1/events/pet", response_model=PetEventEnvelope, status_code=201)
def create_pet_event(
    payload: PetEventIn,
    response: Response,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
) -> PetEventEnvelope:
    """接收宠物抓拍事件；同 key 重放返回原记录，不重复落库。"""
    key = (idempotency_key or "").strip() or None
    if key and len(key) > 200:
        raise HTTPException(400, detail={"code": "INVALID_IDEMPOTENCY_KEY", "message": "Idempotency-Key 不能超过 200 字符"})
    event_time = _utc_iso(payload.timestamp)
    with db() as connection:
        if key:
            row = connection.execute("SELECT * FROM pet_events WHERE idempotency_key = ?", (key,)).fetchone()
            if row is not None:
                _assert_same_event(row, payload, event_time)
                response.status_code = 200
                return PetEventEnvelope(message="幂等重放", data=PetEventOut(**_row_to_event(row)))
        try:
            cursor = connection.execute(
                """INSERT INTO pet_events
                   (idempotency_key, device_id, category, confidence, thumbnail, event_time, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (key, payload.device_id, payload.category, payload.confidence, payload.thumbnail,
                 event_time, _utc_iso(datetime.now(timezone.utc))),
            )
        except sqlite3.IntegrityError:
            row = connection.execute("SELECT * FROM pet_events WHERE idempotency_key = ?", (key,)).fetchone()
            if row is None:
                raise
            _assert_same_event(row, payload, event_time)
            response.status_code = 200
            return PetEventEnvelope(message="幂等重放", data=PetEventOut(**_row_to_event(row)))
        row = connection.execute("SELECT * FROM pet_events WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return PetEventEnvelope(message="事件已接收", data=PetEventOut(**_row_to_event(row)))


@app.get("/api/v1/events", response_model=PetEventPageEnvelope)
def list_events(
    start_time: Optional[datetime] = Query(default=None),
    end_time: Optional[datetime] = Query(default=None),
    pet_type: Optional[str] = Query(default=None, min_length=1, max_length=80),
    category: Optional[str] = Query(default=None, min_length=1, max_length=80),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> PetEventPageEnvelope:
    """按时间范围与宠物类别分页查询事件。"""
    if start_time and end_time and _utc_iso(start_time) > _utc_iso(end_time):
        raise HTTPException(400, detail={"code": "INVALID_TIME_RANGE", "message": "start_time 不能晚于 end_time"})
    clauses, args = [], []
    if start_time:
        clauses.append("event_time >= ?")
        args.append(_utc_iso(start_time))
    if end_time:
        clauses.append("event_time <= ?")
        args.append(_utc_iso(end_time))
    selected_category = pet_type or category
    if selected_category:
        clauses.append("category = ?")
        args.append(selected_category)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with db() as connection:
        total = connection.execute("SELECT COUNT(*) FROM pet_events" + where, args).fetchone()[0]
        rows = connection.execute(
            "SELECT * FROM pet_events" + where + " ORDER BY event_time DESC, id DESC LIMIT ? OFFSET ?",
            [*args, page_size, (page - 1) * page_size],
        ).fetchall()
    data = PetEventPage(items=[PetEventOut(**_row_to_event(row)) for row in rows],
                        total=total, page=page, page_size=page_size)
    return PetEventPageEnvelope(data=data)


def _local_summary(day: date, total: int, counts: Dict[str, int], devices: int) -> str:
    if total == 0:
        return f"{day.isoformat()} 未检测到宠物事件。"
    detail = "、".join(f"{name} {count} 次" for name, count in counts.items())
    return f"{day.isoformat()} 共检测到 {total} 次宠物事件（{detail}），来自 {devices} 台设备。"


async def _summary(day: date, total: int, counts: Dict[str, int], devices: int, fallback: str) -> Tuple[str, str]:
    """有密钥时调用 OpenAI 兼容接口；任何失败都降级为本地统计文本。"""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return fallback, "local"
    try:
        import httpx
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        prompt = f"根据统计写一句简洁中文日报：日期 {day.isoformat()}，事件 {total} 次，类别 {counts}，设备 {devices} 台。"
        async with httpx.AsyncClient(timeout=10.0) as client:
            result = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 120,
                },
            )
            result.raise_for_status()
            text = result.json()["choices"][0]["message"]["content"].strip()
            if text:
                return " ".join(text.split()), "llm"
    except Exception as exc:  # noqa: BLE001 - LLM 故障不能影响日报主流程
        LOGGER.warning("LLM 摘要失败，降级为本地统计：%s", exc)
    return fallback, "local"


@app.post("/api/v1/reports/daily", response_model=DailyReportEnvelope)
async def daily_report(
    payload: Optional[DailyReportRequest] = Body(default=None),
    requested_date: Optional[date] = Query(default=None, alias="date"),
) -> DailyReportEnvelope:
    """汇总指定日期（默认当天）的宠物事件并生成一句话摘要。"""
    day = (payload.report_date if payload else None) or requested_date or date.today()
    local_start = datetime.combine(day, time.min)
    start = local_start.astimezone(timezone.utc)
    end = (local_start + timedelta(days=1)).astimezone(timezone.utc)
    with db() as connection:
        rows = connection.execute(
            "SELECT category, COUNT(*) AS count FROM pet_events WHERE event_time >= ? AND event_time < ? "
            "GROUP BY category ORDER BY count DESC, category",
            (_utc_iso(start), _utc_iso(end)),
        ).fetchall()
        devices = connection.execute(
            "SELECT COUNT(DISTINCT device_id) FROM pet_events WHERE event_time >= ? AND event_time < ?",
            (_utc_iso(start), _utc_iso(end)),
        ).fetchone()[0]
    counts = {row["category"]: row["count"] for row in rows}
    total = sum(counts.values())
    summary, source = await _summary(day, total, counts, devices,
                                     _local_summary(day, total, counts, devices))
    report = DailyReportOut(date=day, total_events=total, category_counts=counts,
                            device_count=devices, summary=summary, summary_source=source,
                            generated_at=datetime.now(timezone.utc))
    return DailyReportEnvelope(message="日报已生成", data=report)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [{"loc": list(item.get("loc", [])), "msg": item.get("msg", ""), "type": item.get("type", "")}
              for item in exc.errors()]
    return JSONResponse(status_code=422, content=ErrorEnvelope(
        code="VALIDATION_ERROR", message="请求参数校验失败", data=errors).model_dump(mode="json"))


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, dict) else {}
    defaults = {400: "BAD_REQUEST", 404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED",
                409: "CONFLICT", 413: "PAYLOAD_TOO_LARGE"}
    envelope = ErrorEnvelope(
        code=detail.get("code", defaults.get(exc.status_code, f"HTTP_{exc.status_code}")),
        message=str(detail.get("message", exc.detail)),
        data=detail.get("data"),
    )
    return JSONResponse(status_code=exc.status_code, content=envelope.model_dump(mode="json"))


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("后端未处理异常", exc_info=exc)
    return JSONResponse(status_code=500, content=ErrorEnvelope(
        code="INTERNAL_ERROR", message="服务内部错误").model_dump(mode="json"))


init_db()
