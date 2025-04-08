from pydantic import BaseModel


class FilePath(BaseModel):
    path: str
