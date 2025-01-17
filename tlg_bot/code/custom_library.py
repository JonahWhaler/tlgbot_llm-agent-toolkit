import re
import chromadb


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
        "_",
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
    _text = text[:]
    _text = _text.replace("&", "&amp;")
    _text = _text.replace("<", "&lt;")
    _text = _text.replace(">", "&gt;")
    return _text
