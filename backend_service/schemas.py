"""端云联动接口的 Pydantic 契约与统一响应信封。"""
from datetime import date as Date
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """输入模型默认拒绝未知字段，避免端侧字段漂移被静默忽略。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PetEventIn(StrictModel):
    """端侧宠物抓拍事件。thumbnail 为 Base64 或 data URL 字符串。"""

    thumbnail: str = Field(
        ...,
        min_length=1,
        max_length=300_000,
        validation_alias=AliasChoices("thumbnail", "thumbnail_base64", "image"),
    )
    category: str = Field(
        ...,
        min_length=1,
        max_length=80,
        validation_alias=AliasChoices("category", "pet_type"),
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime = Field(
        ...,
        validation_alias=AliasChoices("timestamp", "event_time", "captured_at"),
    )
    device_id: str = Field(..., min_length=1, max_length=128)


class PetEventOut(PetEventIn):
    """落库后的宠物事件。"""

    id: int
    created_at: datetime


class PetEventPage(BaseModel):
    """按时间、宠物类别过滤后的分页结果。"""

    items: List[PetEventOut]
    total: int
    page: int
    page_size: int


class DailyReportRequest(StrictModel):
    """日报请求；不传日期时由服务端使用当天日期。"""

    report_date: Optional[Date] = Field(
        default=None,
        validation_alias=AliasChoices("date", "report_date"),
    )


class DailyReportOut(BaseModel):
    """日报统计与一句话摘要。summary_source 为 llm 或 local。"""

    date: Date
    total_events: int
    category_counts: Dict[str, int]
    device_count: int
    summary: str
    summary_source: str
    generated_at: datetime


class ApiEnvelope(BaseModel):
    """统一成功信封。"""

    code: str = "OK"
    message: str = "success"
    data: Any = None


class PetEventEnvelope(ApiEnvelope):
    data: PetEventOut


class PetEventPageEnvelope(ApiEnvelope):
    data: PetEventPage


class DailyReportEnvelope(ApiEnvelope):
    data: DailyReportOut


class ErrorEnvelope(BaseModel):
    """统一错误信封：code 为稳定错误码，data 可携带校验或限额细节。"""

    code: str
    message: str
    data: Optional[Any] = None
