"""Módulo para la descompresión segura y normalización de archivos de exportaciones.

Cumple con principios SOLID, Clean Code, PEP 8 y lineamientos DAMA-BOK para
la ingesta y almacenamiento estructurado de datos históricos (2013-2025).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import stat
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Protocol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

VALID_MONTHS = (
    "Enero",
    "Febrero",
    "Marzo",
    "Abril",
    "Mayo",
    "Junio",
    "Julio",
    "Agosto",
    "Septiembre",
    "Octubre",
    "Noviembre",
    "Diciembre",
)


@dataclass(frozen=True)
class ExtractionResult:
    """Resultado trazable de la descompresión de un archivo ZIP."""

    zip_path: Path
    target_dir: Path
    extracted_files: List[Path] = field(default_factory=list)
    deleted_zip: bool = False
    success: bool = True
    error_message: str = ""


class PathNormalizer(Protocol):
    """Interfaz (ISP/DIP) para estrategias de normalización de rutas de destino."""

    def resolve_target_dir(self, zip_path: Path, root_dir: Path) -> Path:
        """Determina el directorio destino normalizado para un archivo ZIP."""


class YearMonthNormalizer:
    """Normaliza rutas bajo el estándar homogéneo `Expo_YYYY/<Mes>/`."""

    _MONTH_PATTERN = re.compile(
        r"(" + "|".join(VALID_MONTHS) + r")",
        re.IGNORECASE,
    )

    def resolve_target_dir(self, zip_path: Path, root_dir: Path) -> Path:
        """Resuelve la carpeta `Expo_YYYY/<Mes>` evitando anidamientos redundantes."""
        relative_parts = zip_path.relative_to(root_dir).parts
        year_folder = relative_parts[0]  # Ej. Expo_2014, Expo_2020

        match = self._MONTH_PATTERN.search(zip_path.stem)
        month_name = match.group(1).capitalize() if match else zip_path.stem.strip()

        return root_dir / year_folder / month_name


class SafeZipExtractor:
    """Servicio encargado de extraer, verificar integridad y eliminar archivos ZIP (SRP)."""

    def __init__(self, normalizer: PathNormalizer) -> None:
        self._normalizer = normalizer

    def process_zip(self, zip_path: Path, root_dir: Path) -> ExtractionResult:
        """Descomprime un ZIP, valida los archivos extraídos y elimina el ZIP original."""
        target_dir = self._normalizer.resolve_target_dir(zip_path, root_dir)
        extracted_paths: List[Path] = []

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                corrupt_file = zf.testzip()
                if corrupt_file is not None:
                    raise zipfile.BadZipFile(
                        f"Archivo corrupto detectado en el ZIP: {corrupt_file}"
                    )

                members = [m for m in zf.infolist() if not m.is_dir()]
                if not members:
                    raise ValueError(f"El archivo ZIP está vacío: {zip_path}")

                target_dir.mkdir(parents=True, exist_ok=True)
                zf.extractall(path=target_dir)

                for member in members:
                    extracted_file = target_dir / member.filename
                    if not extracted_file.exists() or extracted_file.stat().st_size == 0:
                        raise IOError(
                            f"Verificación fallida para el archivo extraído: {extracted_file}"
                        )
                    extracted_paths.append(extracted_file)

            # Solo eliminamos el .zip tras verificar que todos los archivos existen y tienen tamaño > 0
            zip_path.unlink()
            logger.info(
                "Extraído y eliminado: %s -> %s (%d archivos)",
                zip_path.relative_to(root_dir),
                target_dir.relative_to(root_dir),
                len(extracted_paths),
            )
            return ExtractionResult(
                zip_path=zip_path,
                target_dir=target_dir,
                extracted_files=extracted_paths,
                deleted_zip=True,
                success=True,
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("Error procesando %s: %s", zip_path, exc)
            return ExtractionResult(
                zip_path=zip_path,
                target_dir=target_dir,
                extracted_files=extracted_paths,
                deleted_zip=False,
                success=False,
                error_message=str(exc),
            )


class DirectoryCleaner:
    """Limpia carpetas anidadas vacías residuales como `Expo_YYYY/Expo_YYYY` (SRP)."""

    @staticmethod
    def remove_empty_nested_dirs(root_dir: Path) -> List[Path]:
        """Elimina directorios vacíos que hayan quedado tras eliminar los .zip."""
        removed: List[Path] = []
        for path in sorted(root_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                try:
                    os.chmod(path, stat.S_IWRITE)
                    path.rmdir()
                    removed.append(path)
                    logger.info("Directorio vacío eliminado: %s", path.relative_to(root_dir))
                except PermissionError as exc:
                    logger.warning(
                        "No se pudo eliminar directorio vacío %s (bloqueo de OneDrive): %s",
                        path.relative_to(root_dir),
                        exc,
                    )
        return removed


def run_pipeline(root_dir: Path) -> List[ExtractionResult]:
    """Ejecuta el pipeline completo de descompresión, validación y limpieza."""
    normalizer = YearMonthNormalizer()
    extractor = SafeZipExtractor(normalizer=normalizer)

    zip_files = sorted(root_dir.rglob("*.zip"))
    logger.info("Total de archivos .zip encontrados: %d", len(zip_files))

    results: List[ExtractionResult] = []
    for zip_path in zip_files:
        result = extractor.process_zip(zip_path, root_dir)
        results.append(result)

    failed = [r for r in results if not r.success]
    if failed:
        logger.error("Se presentaron errores en %d archivos ZIP.", len(failed))
    else:
        DirectoryCleaner.remove_empty_nested_dirs(root_dir)
        logger.info(
            "Pipeline completado con éxito: %d archivos ZIP descomprimidos y eliminados.",
            len(results),
        )

    return results


if __name__ == "__main__":
    workspace_root = Path(__file__).resolve().parents[2]
    run_pipeline(workspace_root)
