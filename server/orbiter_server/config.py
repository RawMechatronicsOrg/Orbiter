from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    storage_dir: Path = Path("./data")
    # Thumbnail tiers (sizes/qualities) live in camera_adapter.THUMB_TIERS;
    # config no longer carries them.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174"
    port: int = 8000
    # Default camera_adapter preset when the client doesn't specify one.
    # Override via ORBITER_DEFAULT_CAMERA_PRESET=sm22 in .env when running
    # against a Galaxy S22 (SM-S921B) etc. See storage-api/camera_adapter.py.
    default_camera_preset: str = "native"

    # ESP32 firmware address. The storage-api is the sole proxy to the
    # firmware (Viser-pattern migration) — set ORBITER_ESP_IP in .env.
    # Live state is streamed from the firmware's ws://<esp_ip>/ws/log.
    esp_ip: str = "192.168.1.50"

    # Camera still-image URL (e.g. http://<phone-ip>:8080/photoaf.jpg). The
    # server-side scan loop GETs this for each capture. Empty → a placeholder
    # image is stored instead (pose data is still recorded).
    camera_url: str = ""

    model_config = SettingsConfigDict(
        env_prefix="ORBITER_",
        env_file=".env",
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def scans_dir(self) -> Path:
        return self.storage_dir / "scans"

    @property
    def captures_dir(self) -> Path:
        return self.storage_dir / "captures"


settings = Settings()
