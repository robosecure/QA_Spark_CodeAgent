import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.model_name = os.environ.get("MODEL_NAME", "gpt-5.2")
        self.pass_threshold = int(os.environ.get("PASS_THRESHOLD", "95"))

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Set it in your environment or .env file."
            )
