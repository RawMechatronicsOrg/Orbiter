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
    # against a Galaxy S22 (SM-S921B) etc. See camera_adapter.py.
    default_camera_preset: str = "native"

    # ESP32 firmware address. The server is the sole proxy to the
    # firmware (Viser pattern) — set ORBITER_ESP_IP in .env.
    # Live state is streamed from the firmware's ws://<esp_ip>/ws/log.
    esp_ip: str = "192.168.1.50"

    # Camera still-image URL (e.g. http://<phone-ip>:8080/photoaf.jpg). The
    # server-side scan loop GETs this for each capture. Empty → a placeholder
    # image is stored instead (pose data is still recorded).
    camera_url: str = ""

    # AZ-encoder harmonic correction — calibration "stage B" (1st+2nd harmonic
    # of the AS5600 reading). DISABLED by default: on the reference rig the
    # solved coefficients pinned to their ±bound (over-fit, absorbing unmodeled
    # error), and a ~10° correction is really a magnet-mounting problem, not an
    # encoder-model one. With it off, calibration solves no harmonic and any
    # stored one is ignored everywhere (viz, capture poses, accuracy test).
    # Re-enable via ORBITER_AZ_HARMONIC_ENABLED=true after the AS5600 magnet is
    # reseated and a fresh calibration is validated.
    az_harmonic_enabled: bool = False

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
