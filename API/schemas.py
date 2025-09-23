from pydantic import BaseModel, validator, Field
from datetime import date
from typing import Dict, Any, Optional, List

class DDLStatement(BaseModel):
    statement: str

class QueryInfo(BaseModel):
    queryid: str
    query: str
    runquantity: int

class TaskResponse(BaseModel):
    taskid: str

class StatusResponse(BaseModel):
    taskid: str

class OptimizationRequest(BaseModel):
    url: str
    ddl: List[DDLStatement]
    queries: List[QueryInfo]





# class Genre(BaseModel):
#     name: str

# class MetadataResponse(BaseModel):
#     data: Any
#     metadata: Dict[str, Any]
#
# class Book(BaseModel):
#     title: str
#     writer: str
#     duration: str
#     date: date
#     summary: str
#     genres: List[Genre] = []
#     pages: int
#
# class Bookout(BaseModel):
#     id: int
#
# class Author(BaseModel):
#     first_name: str = Field(..., max_length=5)
#     last_name: str
#     age: int = Field(..., gt=15, lt=90)

    # @validator('age')
    # def check_age(cls, v):
    #     if v < 15:
    #         raise ValueError('< 15')
    #     return v