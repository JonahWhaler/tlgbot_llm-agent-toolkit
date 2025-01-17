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
