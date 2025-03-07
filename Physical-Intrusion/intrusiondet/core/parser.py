"""Parsing mechanism for secret information"""
import logging
import os
from base64 import b64encode
from typing import Any, Callable, Optional, Union

from dotenv import dotenv_values

from intrusiondet.core.types import PathLike

ENCODING_CALLABLES: dict[Optional[str], Callable]
"""A mapping for encodings"""


def identity_encoding(item: Any) -> Any:
    """Trivial encoding identity function

    :param item: A single parameter instance of anything
    :return: The same parameter back
    """
    return item


def none_encoding(item: Any) -> Any:
    """Duplicate of identity_encoding

    :param item: A single parameter instance of anything
    :return: The same parameter back
    """
    return identity_encoding(item)


def byte64_encoding(item: Any, fmt: Optional[str] = None) -> bytes:
    """Encode any input using byte64 encoding

    :param item: Input string
    :param fmt: encoding format (default=utf-8)
    :return: Encoded string
    """
    encode_format: str
    if fmt is not None:
        encode_format = fmt
    else:
        encode_format = "utf-8"
    return b64encode(bytes(item, encode_format))


def parse_dotenv_secret_key_list(config: Union[PathLike, dict]) -> dict:
    """Parse a dotenv secret file in dict-like formatting into a dictionary.
    Required schema is

    KEYS=KEY1,...
    KEY1_KEY=foo
    KEY1_VALUE=bar
    KEY1_ENCODING=byte64_encoding

    :param config: A dotenv file path or loaded dotenv dictionary
    :return: Parsed secret configuration
    """
    logger = logging.getLogger()
    if isinstance(config, PathLike):
        kwargs = dotenv_values(os.fspath(config))
    else:
        kwargs = config
    secret_keys: Optional[str] = kwargs.get("KEYS")
    if not isinstance(secret_keys, str):
        logger.error(
            "Unable to parse secret keys as unparsed value is NOT a string, but a %s",
            str(type(secret_keys)),
        )
        raise TypeError
    if len(secret_keys) == 0:
        logger.info(
            "No secrets to parse from %s",
            os.fspath(config) if isinstance(config, PathLike) else "",
        )
        return {}
    try:
        keys_to_parse: list[str] = secret_keys.split(",")
    except Exception as err:
        logger.error("Unable to parse secrets due to error %s", str(err))
        return {}
    out_secret = {
        kwargs.get(key + "_KEY"): ENCODING_CALLABLES[kwargs.get(key + "_ENCODING")](
            kwargs.get(key + "_VALUE")
        )
        for key in keys_to_parse
    }
    return out_secret


ENCODING_CALLABLES = {
    encoding_callable.__name__: encoding_callable
    for encoding_callable in [none_encoding, identity_encoding, byte64_encoding]
}
ENCODING_CALLABLES[None] = none_encoding
