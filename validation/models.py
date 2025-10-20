from __future__ import annotations
from typing import List, Optional, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict

# Pydantic v2 style. If you have v1, tell me and I'll adapt.

class Publication(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    title: str = Field(..., alias="title")
    authors: List[str] = Field(default_factory=list, alias="authors")
    journal: Optional[str] = Field(None, alias="journal")        # sometimes "venue" in other datasets
    year: Optional[int] = Field(None, alias="year")
    article_link: Optional[str] = Field(None, alias="article_link")

    @field_validator("year", mode="before")
    @classmethod
    def coerce_year(cls, v: Any) -> Optional[int]:
        if v is None:
            return None
        # allow strings like "2019" or "2019 "
        if isinstance(v, str):
            s = v.strip()
            if s.isdigit():
                return int(s)
            return None  # if it's not clean, drop it instead of failing validation
        if isinstance(v, (int,)):
            return v
        return None

class Professor(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    # Map Persian keys with spaces to pythonic names via aliases:
    first_name: str = Field(..., alias="نام")
    last_name: str = Field(..., alias="نام خانوادگی")
    faculty: Optional[str] = Field(None, alias="دانشکده")
    department: Optional[str] = Field(None, alias="گروه آموزشی")
    employment_status: Optional[str] = Field(None, alias="وضعیت اشتغال")
    email: Optional[str] = Field(None, alias="پست الکترونیکی")
    homepage: Optional[str] = Field(None, alias="صفحه شخصی")
    publications: List[Publication] = Field(default_factory=list, alias="publications")