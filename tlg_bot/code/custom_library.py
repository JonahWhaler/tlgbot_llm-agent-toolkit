import os
import re
import json
import logging
import chromadb
from telegram.ext import CallbackContext
from telegram.error import TelegramError


class ChromaDBFactory:
    instance: chromadb.ClientAPI | None = None

    @classmethod
    def get_instance(
        cls, persist: bool | None, persist_directory: str | None
    ) -> chromadb.ClientAPI:
        if cls.instance:
            return cls.instance
        if persist and persist_directory:
            cls.instance = chromadb.Client(
                settings=chromadb.Settings(
                    is_persistent=persist,
                    persist_directory=persist_directory,
                )
            )
        else:
            cls.instance = chromadb.Client()
        return cls.instance


def handle_triple_ticks(text: str, closed: bool):
    TRIPLE_TICKS = "```"
    t = text[:]
    if not closed:
        t = TRIPLE_TICKS + t
    close = len(re.findall(TRIPLE_TICKS, t)) % 2 == 0
    if not close:
        t += TRIPLE_TICKS
    return t, close


def escape_markdown(text) -> str:
    """
    Escape special characters for Telegram's HTMLV2.
    """
    # Characters that need to be escaped
    special_chars = [
        # "_",
        # "*",
        "[",
        "]",
        "(",
        ")",
        # "~",
        # "`",
        ">",
        # "#",
        "+",
        "-",
        "=",
        "|",
        "{",
        "}",
        ".",
        "!",
    ]

    # Escape backslash first to avoid double escaping
    # text = text.replace("\\", "\\\\")
    _text = text[:]
    # # Escape special characters
    for char in special_chars:
        _text = _text.replace(char, f"\\{char}")

    # _text = re.sub(r"#{1,} (.*?)\n", r"*\1*\n", _text)
    _text = re.sub(r"#{2,}", "", _text)
    _text = _text.replace("#", r"\#")

    # logger.info("Raw Markdown: %s", text)
    # logger.info("Escaped Markdown: %s", _text)
    return _text


def escape_html(text) -> str:
    """
    Escape special characters for Telegram's HTMLV2.
    """
    logger = logging.getLogger(__name__)

    _text = text[:]
    _text = _text.replace("&", "&amp;")
    _text = _text.replace("<", "&lt;")
    _text = _text.replace(">", "&gt;")
    # _text = _text.replace("-", r"\-")
    # _text = _text.replace("\n\n", "<br><br>")
    triple_ticks_pattern = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)
    _text = triple_ticks_pattern.sub(
        lambda m: f"<code>\n{m.group(1).strip()}\n</code>", _text
    )

    trap_pattern = re.compile(r"^#{2,} ([\w\d\s:./-]+)(?:\n|$)", re.MULTILINE)
    _text = trap_pattern.sub(lambda m: f"<b>{m.group(1)}</b>\n", _text)

    while True:
        if "**" in _text and len(_text.split("**")) > 2:
            _text = _text.replace("**", "<b>", 1)
            _text = _text.replace("**", "</b>", 1)
        else:
            break

    while True:
        if "*" in _text and len(_text.split("*")) > 2:
            _text = _text.replace("*", "<i>", 1)
            _text = _text.replace("*", "</i>", 1)
        else:
            break

    # print("Raw HTML: %s", text)
    logger.info("Escaped HTML: %s", _text)
    return _text


async def store_to_drive(
    file_id: str, temp_path: str, context: CallbackContext, overwrite: bool = False
):
    logger = logging.getLogger(__name__)

    if os.path.exists(temp_path) and not overwrite:
        return None

    try:
        _file = await context.bot.get_file(
            file_id,
            read_timeout=300,
            write_timeout=300,
            connect_timeout=300,
            pool_timeout=300,
        )
        await _file.download_to_drive(
            temp_path,
            read_timeout=3000,
            write_timeout=3000,
            connect_timeout=3000,
            pool_timeout=300,
        )
        return None
    except TelegramError as tg_err:
        logger.error("[store_to_drive]=TelegramError: %s", str(tg_err))
    except Exception as e:
        logger.error("[store_to_drive]=Exception: %s", str(e))
    raise RuntimeError(f"({file_id}, {temp_path}) => File download failed.")


def unpack_ii_content(data: str) -> str:

    logger = logging.getLogger(__name__)
    output_string = ""
    try:
        jresult = json.loads(data)
        # summary
        if "summary" in jresult:
            summary = jresult["summary"]
            output_string += f"**Summary**\n{summary}\n\n"

        # long_description
        if "long_description" in jresult:
            long_description = jresult["long_description"]
            output_string += f"**Description**\n{long_description}\n\n"

        # keywords
        if "keywords" in jresult:
            keywords = jresult["keywords"]
            if not isinstance(keywords, list):
                logger.warning(
                    "Encounter invalid response schema. Expect keywords to be a list of str, but get '%s'.",
                    type(keywords).__name__,
                )
            else:
                kws = ", ".join(keywords)
                output_string += f"**Keywords**\n[{kws}]"

        if output_string == "":
            output_string = jresult["text"]

        return output_string
    except json.JSONDecodeError as jde:
        logger.error(jde, exc_info=True, stack_info=True)
        raise
    except Exception as e:
        logger.error(e, exc_info=True, stack_info=True)
        raise


def format_identifier(identifier: int) -> str:
    return f"g{identifier}" if identifier < 0 else str(identifier)
