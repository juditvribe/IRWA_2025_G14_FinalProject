from pydantic import BaseModel, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import re


class Document(BaseModel):
    _id: str
    pid: str
    title: str
    description: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    sub_category: Optional[str] = None
    product_details: Optional[Dict[str, Any]] = None
    seller: Optional[str] = None
    out_of_stock: bool = False
    selling_price: Optional[float] = None
    discount: Optional[float] = None
    actual_price: Optional[float] = None
    average_rating: Optional[float] = None
    url: Optional[str] = None
    images: Optional[List[str]] = None

    def to_json(self):
        return self.model_dump_json()

    # --- Validators ---

    @field_validator("selling_price", "actual_price", mode="before")
    def parse_price(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip().replace(",", "")
            if v == "":
                return None
            try:
                return float(v)
            except ValueError:
                return None
        return v

    @field_validator("average_rating", mode="before")
    def parse_rating(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            if v == "":
                return None
            try:
                return float(v)
            except ValueError:
                return None
        return v

    @field_validator("discount", mode="before")
    def parse_discount(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            match = re.search(r"(\d+(?:\.\d+)?)", v.replace(",", ""))
            if match:
                return float(match.group(1))
            return None
        return v

    @field_validator("product_details", mode="before")
    def normalize_product_details(cls, v):
        if isinstance(v, list):
            merged = {}
            for item in v:
                if isinstance(item, dict):
                    merged.update(item)
            return merged
        return v

    def __str__(self) -> str:
        return self.model_dump_json(indent=2)


class StatsDocument(BaseModel):
    """
    Original corpus data as an object
    """
    pid: str
    title: str
    description: Optional[str] = None
    url: Optional[str] = None
    count: Optional[int] = None

    def __str__(self) -> str:
        return self.model_dump_json(indent=2)
    
    def to_json(self):
        return self.model_dump_json()


class ResultItem(BaseModel):
    pid: str
    title: str
    description: Optional[str] = None
    brand: Optional[str] = None
    selling_price: Optional[float] = None
    average_rating: Optional[float] = None
    url: Optional[str] = None
    ranking: Optional[float] = None

    def __str__(self) -> str:
        return self.model_dump_json(indent=2)
    
    def to_json(self):
        return self.model_dump_json()


# New class to show HTTP requests
class UserReq:
    def __init__(self, id, count):
        self.id = id
        self.count = count


# New class to show active sessions
class Session:
    def __init__(self, id, start, actual_time, clicks, browser, os):  
        self.id = id
        self.start = start.replace(microsecond=0)
        self.actual_time = actual_time
        start_time = start.hour * 60 + start.minute
        end_time = actual_time.hour * 60 + actual_time.minute
        self.time_elapsed = end_time-start_time
        self.clicks = clicks
        self.browser = browser
        self.os= os


# New class to show query counter
class Query:
    def __init__(self, query, count):
        self.query = query
        self.count = count 

    def to_json(self):
        return self.__dict__