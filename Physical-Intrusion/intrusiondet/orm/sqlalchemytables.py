"""Define imperatively database tables"""
from typing import Final

from sqlalchemy import (
    TIMESTAMP,
    Boolean,
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
)
from sqlalchemy.orm import registry

import intrusiondet.model as model

MAPPER_REGISTRY: Final[registry] = registry()
"""SQLAlchemy registry for the object relational (ORM) model"""


METADATA: Final[MetaData] = MAPPER_REGISTRY.metadata
"""SQLAlchemy ORM metadata for tables"""


DETECTED_OBJECTS_TABLE_NAME: Final[str] = "detections"
"""SQL table name for detected objects"""


DETECTED_OBJECTS_TABLE = Table(
    DETECTED_OBJECTS_TABLE_NAME,
    METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("class_id", Integer),
    Column("class_name", String(15)),
    Column("conf", Float),
    Column("bbox_left", Integer),
    Column("bbox_top", Integer),
    Column("bbox_width", Integer),
    Column("bbox_height", Integer),
    Column("datetime_utc", TIMESTAMP(timezone=True)),  # Not supported for many drivers
    Column("location", String(31)),
    Column("zones", String(255)),
    Column("intrusion", Boolean),
    Column("filename", String(255)),
    Column("notes", String(255))
)
"""SQL table for `intrusiondet.model.detectedobject.DetectedObject`"""


def start_mappers() -> None:
    """Initialize the object relation model"""
    MAPPER_REGISTRY.map_imperatively(
        model.detectedobject.DetectedObject,
        DETECTED_OBJECTS_TABLE,
    )
