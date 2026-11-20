"""Strict internal observation schema; source quotes never form model answers."""
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,field_validator
from chest_xpert.structured_findings import CATALOG

class FindingScope(BaseModel):
    model_config=ConfigDict(extra='forbid')
    pattern:str|None=None
    size:str|None=None
    degree:str|None=None
    visibility:str|None=None
    temporal:str|None=None
    certainty_threshold:str|None=None

class FindingObservation(BaseModel):
    model_config=ConfigDict(extra='forbid')
    status:Literal['present','absent','uncertain','unassessable']
    assessment_scope:FindingScope
    laterality:Literal['left','right','bilateral']|None=None
    region:str|None=None
    degree:str|None=None
    supporting_view:Literal['PA','AP','frontal','lateral']|None=None

class FindingResponse(BaseModel):
    model_config=ConfigDict(extra='forbid')
    finding:str=Field(json_schema_extra={"enum":list(CATALOG)})
    observations:list[FindingObservation]=Field(min_length=1)
    @field_validator('finding')
    @classmethod
    def known_finding(cls,value):
        if value not in CATALOG:raise ValueError('Unknown finding')
        return value
