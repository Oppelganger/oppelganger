from datetime import datetime

from pydantic import BaseModel


class Stage(BaseModel):
    stage: str = 'Ожидание первичного запроса'
    last_update: datetime = datetime.now()
