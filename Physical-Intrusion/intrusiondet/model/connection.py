"""Connection model with a REST-API"""
import logging
import os
from copy import copy
from json import loads
from typing import Final, Optional, Union

import requests
from dotenv import dotenv_values

from intrusiondet.core.parser import parse_dotenv_secret_key_list
from intrusiondet.core.types import PathLike
from intrusiondet.remote import session


def get_secret_dotenv_values(path_or_dict: Union[PathLike, dict]) -> dict:
    """Using a dotenv-formatted file, load in secret information. See the
    `intrusiondet.core.parser.parse_dotenv_secret_key_list` method for file schema

    :param path_or_dict: Path to a dotenv file or dictionary instance
    :return:
    """
    if isinstance(path_or_dict, dict):
        return path_or_dict
    if isinstance(path_or_dict, PathLike):
        return parse_dotenv_secret_key_list(os.fspath(path_or_dict))
    logging.getLogger().exception("Unsupported type: %s", type(path_or_dict))
    raise TypeError


def get_public_dotenv_values(path_or_dict: Union[PathLike, dict]) -> dict:
    """

    :param path_or_dict:
    :return:
    """
    ret_dict: dict
    if isinstance(path_or_dict, dict):
        ret_dict = path_or_dict.copy()
    elif isinstance(path_or_dict, PathLike):
        ret_dict = dotenv_values(os.fspath(path_or_dict))
    else:
        logging.getLogger().exception("Unsupported type: %s", type(path_or_dict))
        raise TypeError
    for loads_keyname in (
        "POST_KWARGS",
        "GET_KWARGS",
    ):
        loads_str: Optional[str] = ret_dict.get(loads_keyname)
        if loads_str is not None:
            ret_dict[loads_keyname] = loads(loads_str)
    return ret_dict


class Connection:
    """A remote connection with public, private, and session information"""

    def __init__(self, public: str, secret: str):
        """A remote connection with public, private, and session information

        :param public: Path to the public data dotenv file
        :param secret: Path to the secret data dotenv file
        """
        self._public: Final[dict] = get_public_dotenv_values(public)
        self._secret: Final[dict] = get_secret_dotenv_values(secret)
        self._sessionkey: str = ""

    @property
    def public(self) -> dict:
        """The public connection details

        :return: Public data
        """
        return self._public.copy()

    @property
    def sessionkey(self) -> str:
        """The session key for connection

        :return: The session key string
        """
        return copy(self._sessionkey)

    def activate_session(self):
        """Activate a remote session"""
        sessionkey = session.get_session_key(
            url=self._public.get("API_GET_SESSIONKEY"),
            data=self._secret,
            session_keyword=self._public.get("KEY_GET_SESSIONKEY"),
            post_kwargs=self._public.get("POST_KWARGS"),
        )
        if sessionkey is None:
            logging.getLogger().error(
                "Unable to get session key with API_GET_SESSIONKEY at %s",
                self._public.get("API_GET_SESSIONKEY"),
            )
            raise requests.RequestException
        self._sessionkey = sessionkey
