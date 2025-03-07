"""Extracting session data like session key, list of images/videos, and remote file download

"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Final, Optional

import requests

from intrusiondet.core.types import PathLike

RESPONSE_POST_DEFAULT_KWARGS: Final[dict] = {"verify": False}
"""Default kwargs for the request.post callable"""

RESPONSE_GET_DEFAULT_KWARGS: Final[dict] = {"verify": False}
"""Default kwargs for the request.get callable"""


def run_callable_get_response(
    request_call: Callable, url: str, n_attempts: int, callable_kwargs: dict
) -> Optional[requests.Response]:
    """General purpose remote method call with multiple attempts

    :param request_call: The remote method
    :param url: The URL
    :param n_attempts: Number of attempts before returning nothing
    :param callable_kwargs: Any kwargs for the method call
    :return: A response if available
    """
    logger = logging.getLogger()
    response: Optional[requests.Response] = None
    attempt_count: int = 0
    while attempt_count < n_attempts:
        try:
            logger.debug(
                "Running function %s with url %s with kwargs %s",
                str(request_call),
                url,
                str(callable_kwargs),
            )
            response = request_call(url=url, **callable_kwargs)
        except requests.exceptions.Timeout:
            pass
        except requests.exceptions.TooManyRedirects:
            logger.error("TooManyRedirects exception was raised")
            return None
        except requests.exceptions.RequestException:
            logger.error("General RequestException exception was raised")
            return None
        if response is not None:
            break
        attempt_count += 1
        logger.info("Attempting retry %d of %d", attempt_count, n_attempts)
    if response is None:
        logger.error("Too many timeout errors occurred")
    return response


def run_post_get_response(
    url: str,
    n_attempts: int,
    post_kwargs: dict,
) -> Optional[requests.Response]:
    """Execute a remote.post method and return the response

    :param url: The URL
    :param n_attempts: Number of attempts
    :param post_kwargs: Any kwargs for the get method
    :return: A response if available
    """
    request_call: Callable = requests.post
    response: Optional[requests.Response] = run_callable_get_response(
        request_call=request_call,
        url=url,
        n_attempts=n_attempts,
        callable_kwargs=post_kwargs,
    )
    return response


def run_get_get_response(
    url: str, n_attempts: int, get_kwargs: dict
) -> Optional[requests.Response]:
    """Execute a remote.get method and return the response

    :param url: The URL
    :param n_attempts: Number of attempts
    :param get_kwargs: Any kwargs for the get method
    :return: A response if available
    """
    request_call: Callable = requests.get
    response: Optional[requests.Response] = run_callable_get_response(
        request_call=request_call,
        url=url,
        n_attempts=n_attempts,
        callable_kwargs=get_kwargs,
    )
    return response


def get_from_response(response: requests.Response, target: str) -> Optional[dict]:
    """Extract a target dictionary given a response

    :param response: A response to an API call
    :param target: Target string
    :return: A dictionary if the target is a member of the response keys
    """
    logger = logging.getLogger()
    response_body: dict = response.json()
    logger.debug("Response body: %s", response_body)
    if not response_body.get("success"):
        logger.error("Unsuccessful response with target %s", target)
        return None
    if target not in response_body:
        logger.error("Target %s not in response body", target)
        return None
    target_response: dict = response_body.get(target)
    logger.info("Got target %s", target)
    return target_response


def get_results(response: requests.Response) -> Optional[dict]:
    """Get the results of an API call

    :param response: The response instance
    :return: A dictionary if the "results" keyword is in the response
    """
    return get_from_response(response=response, target="results")


def get_session_key(
    url: str,
    data: dict,
    session_keyword: str,
    n_attempts: int = 3,
    post_kwargs: Optional[dict] = None,
) -> Optional[str]:
    """Get the session key using an API call

    :param url: URL path
    :param data: Required data for the session activation
    :param session_keyword: The unique keyname for the session key
    :param n_attempts: The number of attempts before returning
    :param post_kwargs: The kwargs for the remote.post method
    :return: A session key if available
    """
    logger = logging.getLogger()
    if post_kwargs is None:
        post_kwargs = RESPONSE_POST_DEFAULT_KWARGS.copy()
    post_kwargs.update({"json": None, "data": data})

    response: Optional[requests.Response] = run_post_get_response(
        url=url, n_attempts=n_attempts, post_kwargs=post_kwargs
    )
    error_msg = (
        "Unable to obtain session key at %s, with the data %s, session keyword %"
        ", and post kwargs %s"
    )
    error_args = url, str(data), str(session_keyword), str(post_kwargs)
    if response is None:
        logger.error(error_msg, *error_args)
        return ""
    results: dict = get_results(response)
    session_key: Optional[str] = results.get(session_keyword)
    if session_key:
        logger.info("Successfully obtained session key")
    else:
        logger.error(error_msg, *error_args)
    return session_key


def get_list_stored_images(
    url: str,
    data: dict,
    image_list_keyword: str,
    n_attempts: int = 3,
    post_kwargs: Optional[dict] = None,
) -> Optional[list[dict]]:
    """Get a list of stored images of the specified encoding using an API call

    :param url: URL path
    :param data: Required data for the call
    :param image_list_keyword: The unique keyname for the list of stored images
    :param n_attempts: The number of attempts before returning
    :param post_kwargs:
    :return: A message list of stored videos of `SavedVideo` signature
    """
    logger = logging.getLogger()
    if post_kwargs is None:
        post_kwargs = RESPONSE_POST_DEFAULT_KWARGS.copy()
    post_kwargs.update({"data": data})

    response: Optional[requests.Response] = run_post_get_response(
        url=url, n_attempts=n_attempts, post_kwargs=post_kwargs
    )

    error_msg = (
        "Unable to obtain stored images at %s, with the data %s, session keyword %"
        ", and post kwargs %s"
    )
    error_args = url, str(data), str(image_list_keyword), str(post_kwargs)
    if response is None:
        logger.error(error_msg, *error_args)
        return []

    results: dict = get_results(response)
    image_list: Optional[list[dict]] = results.get(image_list_keyword)
    if image_list:
        logger.info("Successfully obtained image list")
    else:
        logger.error(error_msg, *error_args)
    return image_list


def download_image(
    url: str,
    filename: PathLike,
    chuck_size_bytes: int = 1024**2,
    n_attempts: int = 3,
    overwrite: bool = True,
    get_kwargs: Optional[dict] = None,
) -> int:
    """Download a remote image to the specified filename using an API call

    :param url: URL path
    :param filename: The local filename
    :param chuck_size_bytes: Download chuck size in bytes (default = 1024**2)
    :param n_attempts: Number of attempts to download the image (default = 3)
    :param overwrite: Overwrite if already available (default = True)
    :param get_kwargs: Any kwargs for the remote.get method
    :return: The filesize in bytes
    """
    logger = logging.getLogger()
    logger.info("Downloading image from %s to %s", url, os.fspath(filename))
    if get_kwargs is None:
        get_kwargs = RESPONSE_POST_DEFAULT_KWARGS.copy()
    get_kwargs.update({"stream": True})

    download_filename = Path(filename)
    Path(download_filename.parent).mkdir(exist_ok=True, parents=True)
    if overwrite:
        download_filename.unlink(missing_ok=True)
    response: Optional[requests.Response] = run_get_get_response(
        url=url, n_attempts=n_attempts, get_kwargs=get_kwargs
    )
    error_msg = (
        "Unable to download video at %s, with the target filename %s, and get kwargs %s"
    )
    error_args = url, os.fspath(filename), str(get_kwargs)
    if response is None:
        logger.error(error_msg, *error_args)
        return 0

    try:
        with download_filename.open("wb") as downloaded_file_pointer:
            for chuck_index, chunk in enumerate(
                response.iter_content(chunk_size=chuck_size_bytes)
            ):
                if chunk:
                    logger.info("Writing chuck #%d", chuck_index)
                    downloaded_file_pointer.write(chunk)
    except requests.exceptions.RequestException:
        logger.error(
            "General RequestException exception was raised." + os.linesep + "%s",
            error_msg,
            *error_args,
        )
        download_filename.unlink(missing_ok=False)
        return 0
    download_size: int = download_filename.stat().st_size
    logger.info("Downloaded image with size in bytes: %d", download_size)
    return download_size
