import os


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.environ.get("OPENAI_BASE_URL", "https://api.groq.com/openai/v1/")
    OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "moonshotai/kimi-k2-instruct-0905")
    STORAGE_PATH: str = os.environ.get("RULECHEF_STORAGE_PATH", "./rulechef_data")
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    SESSION_SECRET: str = os.environ.get("SESSION_SECRET", "dev-insecure-session-secret-change-me")
    SESSION_COOKIE_NAME: str = os.environ.get("SESSION_COOKIE_NAME", "rulechef_session")
    SESSION_TTL_SECONDS: int = int(os.environ.get("SESSION_TTL_SECONDS", "3600"))
    SESSION_COOKIE_SECURE: bool = _env_bool("SESSION_COOKIE_SECURE", default=False)
    SESSION_COOKIE_SAMESITE: str = os.environ.get("SESSION_COOKIE_SAMESITE", "lax")
    SESSION_COOKIE_PATH: str = os.environ.get("SESSION_COOKIE_PATH", "/")


settings = Settings()
