import os
import yaml


with open("/config/provider.yaml", "r", encoding="utf-8") as model_file:
    PROVIDER: dict[str, dict] = yaml.safe_load(model_file)

DEFAULT_T2T_MODEL = ("deepseek", "deepseek-chat")
DEFAULT_I2T_MODEL = ("ollama", "llava:7b")

with open("/config/parameter.yaml", "r", encoding="utf-8") as parameter_file:
    PARAMETER: dict[str, dict] = yaml.safe_load(parameter_file)

with open("/config/character.yaml", "r", encoding="utf-8") as character_file:
    CHARACTER: dict[str, dict] = yaml.safe_load(character_file)

DEFAULT_CHARACTER = "general"

with open("/config/membership.yaml", "r", encoding="utf-8") as membership_file:
    MEMBERSHIP: dict[str, int] = yaml.safe_load(membership_file)

PREMIUM_MEMBERS = []
for member in MEMBERSHIP:
    PREMIUM_MEMBERS.append(MEMBERSHIP[member])

REPO_PATH = os.environ["REPO_PATH"]

TLG_TOKEN = os.environ["TLG_TOKEN"]
TLG_MAX_RETRY = int(os.environ["TLG_MAX_RETRY"])

OLLAMA_HOST = os.environ["OLLAMA_HOST"]

MEMORY_LEN = int(os.environ["MEMORY_LEN"])

DEBUG = os.environ["DEBUG"]
FREE = os.environ["FREE"]
