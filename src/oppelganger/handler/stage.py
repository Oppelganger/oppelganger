from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class Stage(BaseModel):
    stage: str = 'Ожидание первичного запроса'
    last_update: datetime = datetime.now()
    object_url: str = None

    def update_stage(self, stage_name: str, object_url: Optional[str] = None):
        self.stage = stage_name
        self.last_update = datetime.now()
        self.object_url = object_url

