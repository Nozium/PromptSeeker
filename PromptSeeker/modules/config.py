from dotenv import load_dotenv

load_dotenv()

import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))

# laod dotenv file
DB_TYPE = os.getenv("DB_TYPE")
DB_ENDPOINT = os.getenv("DB_ENDPOINT")
DB_TOKEN = os.getenv("DB_TOKEN")
COMMON_TOKEN = os.getenv("COMMON_TOKEN")
WORLD_TOKEN = os.getenv("WORLD_TOKEN")
SPECIES_TOKEN = os.getenv("SPECIES_TOKEN")
CHARACTER_TOKEN = os.getenv("CHARACTER_TOKEN")
NOVERIST_TOKEN = os.getenv("NOVERIST_TOKEN")
OBSERVER_TOKEN = os.getenv("OBSERVER_TOKEN")

MODERATE_CATEGORY_SCORE = {
    "hate": 0.5,
    "hate/threatening": 0.5,
    "self-harm": 0.5,
    "sexual": 0.5,
    "sexual/minors": 0.5,
    "violence": 0.5,
    "violence/graphic": 0.5,
}
