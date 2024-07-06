from pydantic import BaseModel


class AnalysisResponseForm(BaseModel):
    labels: list