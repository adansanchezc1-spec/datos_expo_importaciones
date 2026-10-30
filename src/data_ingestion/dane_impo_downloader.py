"""Módulo de descarga resiliente de microdatos de Importaciones (IMPO) desde DANE ANDA.

Descarga los paquetes oficiales de los años de estudio (2013, 2014, 2018, 2019,
2020, 2024 y 2025) desde los catálogos 473 (histórico 2012-2024) y 856 (2025-2026)
de microdatos.dane.gov.co, aplicando principios SOLID, Clean Code y PEP 8.
"""

from __future__ import annotations

import concurrent.futures
import logging
import ssl
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DaneImpoResource:
    """Metadatos de un archivo oficial de microdatos de importaciones del DANE."""

    year: int
    filename: str
    catalog_id: int
    resource_id: int

    @property
    def download_url(self) -> str:
        """Construye el endpoint directo de descarga en ANDA DANE."""
        return (
            f"https://microdatos.dane.gov.co/index.php/catalog/"
            f"{self.catalog_id}/download/{self.resource_id}"
        )


# Catálogo oficial de recursos IMPO correspondientes a las mismas fechas de Expo_*
TARGET_IMPO_RESOURCES: Tuple[DaneImpoResource, ...] = (
    DaneImpoResource(year=2013, filename="Impo_2013.zip", catalog_id=473, resource_id=9029),
    DaneImpoResource(year=2014, filename="Impo_2014.zip", catalog_id=473, resource_id=9030),
    DaneImpoResource(year=2018, filename="Impo_2018.zip", catalog_id=473, resource_id=10347),
    DaneImpoResource(year=2019, filename="Impo_2019.zip", catalog_id=473, resource_id=12602),
    DaneImpoResource(year=2020, filename="Impo_2020.zip", catalog_id=473, resource_id=20042),
    DaneImpoResource(year=2024, filename="Impo_2024.zip", catalog_id=473, resource_id=24390),
    DaneImpoResource(year=2025, filename="Impo_2025_1.zip", catalog_id=856, resource_id=24417),
    DaneImpoResource(year=2025, filename="Impo_2025_2.zip", catalog_id=856, resource_id=24464),
)


class DaneHttpDownloader:
    """Cliente HTTP con soporte de reintentos y reanudación para descargas grandes (SRP)."""

    CHUNK_SIZE = 1024 * 1024  # 1 MB
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )

    def __init__(self, staging_dir: Path, max_retries: int = 5) -> None:
        self._staging_dir = staging_dir
        self._max_retries = max_retries
        self._ssl_ctx = ssl.create_default_context()
        self._ssl_ctx.check_hostname = False
        self._ssl_ctx.verify_mode = ssl.CERT_NONE

    def download_resource(self, resource: DaneImpoResource) -> Path:
        """Descarga un archivo ZIP desde DANE verificando su integridad estructural."""
        self._staging_dir.mkdir(parents=True, exist_ok=True)
        dest_path = self._staging_dir / resource.filename

        if dest_path.exists() and zipfile.is_zipfile(dest_path):
            logger.info(
                "Archivo existente y válido en caché de descarga: %s (%.2f MB)",
                dest_path.name,
                dest_path.stat().st_size / (1024**2),
            )
            return dest_path

        for attempt in range(1, self._max_retries + 1):
            try:
                existing_bytes = dest_path.stat().st_size if dest_path.exists() else 0
                headers = {"User-Agent": self.USER_AGENT}
                mode = "wb"
                if existing_bytes > 0:
                    headers["Range"] = f"bytes={existing_bytes}-"
                    mode = "ab"

                req = urllib.request.Request(resource.download_url, headers=headers)
                with urllib.request.urlopen(req, context=self._ssl_ctx, timeout=120) as resp:
                    status = getattr(resp, "status", 200)
                    if status == 200 and existing_bytes > 0:
                        # El servidor no aceptó Range; reiniciamos desde 0
                        mode = "wb"
                        existing_bytes = 0

                    content_length = resp.headers.get("Content-Length")
                    total_expected = (
                        existing_bytes + int(content_length)
                        if content_length and content_length.isdigit()
                        else 0
                    )

                    logger.info(
                        "[Intento %d/%d] Descargando %s (esperado: %.2f MB)...",
                        attempt,
                        self._max_retries,
                        resource.filename,
                        total_expected / (1024**2) if total_expected else 0.0,
                    )

                    downloaded = existing_bytes
                    last_log_mb = downloaded // (50 * 1024 * 1024)
                    with open(dest_path, mode) as out_file:
                        while True:
                            chunk = resp.read(self.CHUNK_SIZE)
                            if not chunk:
                                break
                            out_file.write(chunk)
                            downloaded += len(chunk)
                            current_50mb = downloaded // (50 * 1024 * 1024)
                            if current_50mb > last_log_mb:
                                last_log_mb = current_50mb
                                logger.info(
                                    "Progreso %s: %.1f MB descargados...",
                                    resource.filename,
                                    downloaded / (1024**2),
                                )

                if not zipfile.is_zipfile(dest_path):
                    # Verificar si es otro formato comprimido (ej. RAR/7z) o archivo truncado
                    with open(dest_path, "rb") as f_sig:
                        magic = f_sig.read(8)
                    logger.info(
                        "Archivo %s descargado (%.2f MB), firma mágica: %s",
                        dest_path.name,
                        dest_path.stat().st_size / (1024**2),
                        magic.hex(),
                    )
                    if dest_path.stat().st_size < 1024 * 1024:
                        raise IOError(f"Archivo descargado demasiado pequeño o corrupto: {dest_path}")

                logger.info(
                    "Completado: %s (%.2f MB)",
                    dest_path.name,
                    dest_path.stat().st_size / (1024**2),
                )
                return dest_path

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Error en intento %d para %s: %s",
                    attempt,
                    resource.filename,
                    exc,
                )
                if attempt == self._max_retries:
                    raise
                time.sleep(3 * attempt)

        raise RuntimeError(f"No se pudo descargar {resource.filename}")


def download_all_impo_archives(staging_dir: Path, max_workers: int = 3) -> List[Path]:
    """Descarga en paralelo todos los paquetes de importaciones requeridos."""
    downloader = DaneHttpDownloader(staging_dir=staging_dir)
    downloaded_files: List[Path] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_res = {
            executor.submit(downloader.download_resource, res): res
            for res in TARGET_IMPO_RESOURCES
        }
        for future in concurrent.futures.as_completed(future_to_res):
            res = future_to_res[future]
            path = future.result()
            downloaded_files.append(path)
            logger.info("Recurso listo: %s (año %d)", path.name, res.year)

    return sorted(downloaded_files)


if __name__ == "__main__":
    import sys

    target_staging = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else Path(__file__).resolve().parents[2] / "data" / "raw_impo_zips"
    )
    download_all_impo_archives(target_staging, max_workers=3)
