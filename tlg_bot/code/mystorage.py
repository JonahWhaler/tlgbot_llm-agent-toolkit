import json
import os
import re
import sqlite3
import csv
import time
import threading
from typing import Any
from abc import ABC, abstractmethod
import chromadb


class Storage(ABC):
    """
    Abstract class for storage.

    Methods:
    --------
    get(key: str) -> Any
        Gets the value associated with the key.

    set(key: str, value: Any) -> None
        Sets the value associated with the key.
    """

    @abstractmethod
    def get(self, key: str) -> Any:
        """
        Gets the value associated with the key.

        Parameters
        ----------
        key : str
            The key to get the value for.

        Returns
        -------
        Any
            The value associated with the key.
        """

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """
        Sets the value associated with the key.

        Parameters
        ----------
        key : str
            The key to set the value for.
        value : Any
            The value to set.
        """

    @abstractmethod
    def drop(self, key: str):
        """
        Drops the value associated with the key.

        Parameters
        ----------
        key : str
            The key to drop the value for.
        """

    @abstractmethod
    def clear(self):
        """
        Clears all the values in the storage.
        """

    @abstractmethod
    def keys(self) -> list:
        """
        Returns all the keys in the storage.
        """


class SQLite3_Storage(Storage):
    """
    SQLite3_Storage is a subclass of the Storage abstract base class.
    This class provides a concrete implementation of the Storage interface using SQLite3 as the storage system.
    It stores key-value pairs in a SQLite3 database.

    Attributes:
    db_path (str): The path to the SQLite3 database.
    table_name (str): The name of the table in the SQLite3 database.

    Notes:
    Expect the keys to be string, or at least convertible to strings.
    """

    def __init__(
        self, db_path: str, table_name: str = "storage", overwrite: bool = False
    ):
        """
        Initializes a new instance of the SQLite3_Storage class.

        Args:
        db_path (str): The path to the SQLite3 database.
        table_name (str, optional): The name of the table in the SQLite3 database. Defaults to "storage".
        overwrite (bool, optional): If True, overwrites the existing database at db_path. Defaults to False.
        """
        self.db_path = db_path
        self.table_name = table_name
        self.init(db_path, table_name, overwrite)

    @classmethod
    def init(cls, db_path: str, table_name: str, overwrite: bool = False):
        """
        Initializes the SQLite3 database.

        Args:
        db_path (str): The path to the SQLite3 database.
        table_name (str): The name of the table in the SQLite3 database.
        overwrite (bool, optional): If True, overwrites the existing database at db_path. Defaults to False.
        """
        _ = [
            SQLite3_Storage.validate_db_path(db_path),
            SQLite3_Storage.validate_table_name(table_name),
        ]  # Exception will be raised if validation fails

        if overwrite:
            if os.path.exists(db_path):
                os.remove(db_path)
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS {table_name} (key TEXT PRIMARY KEY, value TEXT)"
            )
            conn.commit()
        except sqlite3.Error as e:
            raise e
        finally:
            conn.close()

    @classmethod
    def validate_db_path(cls, db_path: str):
        if not isinstance(db_path, str):
            raise ValueError(f"Invalid database path: {db_path}")

        if not db_path:
            raise ValueError("Database path cannot be empty")

        if not os.path.basename(db_path):
            raise ValueError(f"Invalid database path: {db_path}")

    @classmethod
    def validate_table_name(cls, table_name: str):
        if not isinstance(table_name, str):
            raise ValueError(f"Invalid table name: {table_name}")

        if not table_name:
            raise ValueError("Table name cannot be empty")

        if re.search(r"^\w+$", table_name) is None:
            raise ValueError(f"Invalid table name: {table_name}")

    def get(self, key: str):
        """
        Retrieves the value associated with the given key from the SQLite3 database.

        Args:
        key (str): The key to retrieve the value for.

        Returns:
        Any: The value associated with the given key, or None if the key does not exist.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT value FROM {self.table_name} WHERE key=?", (key,))
            result = cursor.fetchone()
            if result:
                return json.loads(result[0])
            return None
        except sqlite3.Error as e:
            raise e
        finally:
            conn.close()

    def set(self, key: str, value: Any):
        """
        Sets the value for the given key in the SQLite3 database.

        Args:
        key (str): The key to set the value for.
        value (Any): The value to set.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"INSERT OR REPLACE INTO {self.table_name} (key, value) VALUES (?, ?)",
                (key, json.dumps(value, ensure_ascii=False)),
            )
            # ensure_ascii = False to support non-ascii characters
            # sqlite3 support utf-8 by default without further configuration
            conn.commit()
        except sqlite3.Error as e:
            raise e
        finally:
            conn.close()

    def drop(self, key: str):
        """
        Deletes the key-value pair associated with the given key from the SQLite3 database.

        Args:
        key (str): The key to delete the value for.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {self.table_name} WHERE key=?", (key,))
            conn.commit()
        except sqlite3.Error as e:
            raise e
        finally:
            conn.close()

    def clear(self):
        """
        Deletes all key-value pairs from the SQLite3 database.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {self.table_name}")
            conn.commit()
        except sqlite3.Error as e:
            raise e
        finally:
            conn.close()

    def keys(self) -> list[str]:
        """
        Returns a list of all keys in the SQLite3 database.

        Returns:
        list[str]: A list of all keys in the SQLite3 database.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT key FROM {self.table_name}")
            return [row[0] for row in cursor.fetchall()]
        except sqlite3.Error as e:
            raise e
        finally:
            conn.close()

    def export_csv(self, filename: str) -> None:
        assert filename[-4:] == ".csv"
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT key, value FROM {self.table_name}")
            with open(filename, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile, delimiter=",")
                writer.writerow([i[0] for i in cursor.description])
                writer.writerows(cursor)
        except sqlite3.Error as e:
            raise e
        finally:
            conn.close()


class WebCache:
    def __init__(self, ttl: int = 300, maxsize: int = 128):
        self.mem: dict[str, tuple[float, Any]] = {}
        self.lock = threading.Lock()
        self.ttl = ttl
        self.maxsize = maxsize

    def get(self, url: str) -> None | Any:
        with self.lock:
            now_second = time.time()
            data_tuple: tuple[float, Any] | None = self.mem.get(url, None)

            if data_tuple is None:
                return None

            create_second, data_dict = data_tuple
            if now_second - create_second > self.ttl:
                return None

            self.mem[url] = (now_second, data_dict)
            self.__checkout()
            return data_dict

    def set(self, url: str, data_dict: Any):
        with self.lock:
            now_second = time.time()
            self.mem[url] = (now_second, data_dict)
            self.__checkout()

    def __checkout(self) -> None:
        now_second = time.time()
        expired_url: list[str] = []
        for url, dtuple in self.mem.items():
            create_second, _ = dtuple
            if now_second - create_second > self.ttl:
                expired_url.append(url)
        for url in expired_url:
            self.mem.pop(url)

        expecting_to_remove_num = len(list(self.mem.keys())) - self.maxsize
        if expecting_to_remove_num > 0:
            score_list: list[tuple[str, float]] = []
            for url, dtuple in self.mem.items():
                create_second, _ = dtuple
                score_list.append((url, now_second - create_second))

            while expecting_to_remove_num > 0:
                url, sec = max(score_list, key=lambda x: x[1])
                score_list.remove((url, sec))
                self.mem.pop(url)
                expecting_to_remove_num -= 1


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


# END
