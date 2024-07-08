from pydantic import BaseModel


class PredictRequestForm(BaseModel):
    age: int
    gender: str