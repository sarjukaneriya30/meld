"""Database interactions including the client sever, (server) database, and a database
query through SQLAlchemy and SQLite3.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from sqlalchemy import MetaData, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.engine.result import ChunkedIteratorResult
from sqlalchemy.orm import Session, clear_mappers, sessionmaker

from intrusiondet.model.detectedobject import DetectedObject, \
    sort_key_detected_object_by_datetime_utc
from intrusiondet.orm.repositorybase import RepositoryBase


class SQLAlchemyRepository(RepositoryBase):
    """SQL-based repository using SQLAlchemy"""

    def __init__(self, session: Session) -> None:
        """SQL-based repository

        :param session: SQLAlchemy initialized session
        """
        super().__init__()
        self.session = session

    @classmethod
    def create_session(
        cls,
        metadata: MetaData,
        start_mappers: Callable,
        driver: str,
        host: str,
        port: Optional[str] = None,
        uname: Optional[str] = None,
        passwd: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """Create a session with a database. The connection details follow SQLAlchemy
        engine syntax driver://user:pass@host/database

        :param metadata: Table metadata to define the tables
        :param start_mappers: Callable method to initialize ORM mappings
        :param driver: SQL driver/engine name like sqlite, postgresql
        :param host: Host address of the repository.
        :param port: Port for host address, if necessary
        :param uname: Username for the host, if necessary
        :param passwd: Password for the host, if necessary
        :param name: Name of the repository, if necessary
        :return: SQLAlchemy repository
        """
        logger = logging.getLogger()
        if uname is None:
            uname = ""
        if passwd is None:
            passwd = ""
        if port is None:
            port = ""
        if name is None:
            name = ""
        connection_string = f"{driver}://"
        # SQLite
        if "sqlite" in driver:
            connection_string += "/"
            if "memory" in host:
                connection_string += ":memory:"
            else:
                connection_string += host
        # non-sqlite syntax
        else:
            logger.warning(
                "Drivers other than SQLite are NOT tested. The input driver is %s",
                driver,
            )
        if "postgresql" in driver:
            if len(uname) > 0:
                connection_string += uname
            if len(passwd) > 0:
                connection_string += f":{passwd}"
            connection_string += f"@{host}"
            if len(port) > 0:
                connection_string += f":{port}"
            if len(name):
                connection_string += f"/{name}"
        if "http" in driver:
            connection_string += host
            if len(port) > 0:
                connection_string += f":{port}"
        logger.info("Starting SQLAlchemy mappers")
        start_mappers()
        logger.info(
            "Attempting to connect/create to SQL database with connection string %s",
            connection_string,
        )
        engine: Engine = create_engine(connection_string, echo=True)
        metadata.create_all(engine)
        session_maker = sessionmaker(engine)
        session = session_maker()
        return cls(session)

    def __del__(self):
        self._logger.info("Ending SQLAlchemy session")
        self.session.close()
        clear_mappers()

    def add(self, item: Any) -> None:
        """Add an item to the repository. Make sure to run a session.commit afterwards

        :param item: Table item
        """
        try:
            self._logger.info("Trying to add object %s", str(item))
            self.session.add(item)
        except Exception as err:
            self._logger.exception(
                "Unable to add object %s due to %s", str(item), str(err)
            )
            self.session.rollback()
            raise

    def keep_most_recent_detections(self, size: int) -> list[DetectedObject]:
        """Limit detections table to fixed row count by keeping most recent detections

        :param size: Number of rows to save
        :return: List of detected object (rows) to delete
        """
        try:
            self._logger.info(
                "Trying to reduce number of DetectedObjects in database to %d entries",
                size
            )
            deleted_records: list[DetectedObject] = []
            detection_records: list[DetectedObject] = self.list(DetectedObject)
            detection_records.sort(key=sort_key_detected_object_by_datetime_utc)
            while len(detection_records) > size:
                det_obj = detection_records.pop(0)
                self.session.delete(det_obj)
                deleted_records.append(det_obj)
            return deleted_records
        except Exception as err:
            self._logger.exception(
                "Unable to delete DetectedObject due to %s", str(err)
            )
            self.session.rollback()
            raise

    def list(self, orm_type: Any) -> list:
        """Run a query for the defined orm table

        :param orm_type: Type defined in a table
        :return: List of same type
        """
        statement = select(orm_type)
        executed: ChunkedIteratorResult = self.session.execute(statement)
        return list(executed.iterator)
