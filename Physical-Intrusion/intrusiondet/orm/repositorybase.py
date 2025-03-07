"""Base classes and methods for a database repository"""
from __future__ import annotations

import abc
import logging
from typing import Any


class RepositoryBase:
    """Database repository pattern"""

    def __init__(self):
        """Database repository pattern"""
        self._logger = logging.getLogger()

    @abc.abstractmethod
    def add(self, item: Any) -> None:
        """Add an item to the repository"""
        raise NotImplementedError

    @abc.abstractmethod
    def list(self, orm_type: Any) -> list:
        """List all available records of specified type in the repository

        :param orm_type: A mapped object type
        :return: List of objects found
        """
        raise NotImplementedError
