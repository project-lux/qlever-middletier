"""Pydantic response models used to document the API."""

from __future__ import annotations

from pydantic import BaseModel


class SearchScopes(BaseModel):
    searchScopes: dict[str, int]


class StatisticsResponse(BaseModel):
    estimates: SearchScopes
