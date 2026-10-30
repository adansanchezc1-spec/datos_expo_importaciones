"""Módulo para la extracción y normalización estructural de microdatos de Importaciones (IMPO).

Estructura los archivos descargados de DANE ANDA bajo el esquema homogéneo
`Impo_YYYY/<Mes>/<archivo>` idéntico a `Expo_YYYY/<Mes>/<archivo>`, aplicando
principios SOLID, Clean Code, PEP 8 y lineamientos DAMA-BOK.
"""

from __future__ import annotations

import io
import logging
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

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

MONTH_REGEX = re.compile(r"(" + "|".join(VALID_MONTHS) + r")", re.IGNORECASE)
YEAR_REGEX = re.compile(r"(20\d{2})")


@dataclass(frozen=True)
class ImpoYearExtractionSummary:
    """Resumen trazable de la extracción de un paquete anual/semestral de importaciones."""

    archive_name: str
    year: int
    months_extracted: List[str] = field(default_factory=list)
    files_extracted: int = 0
    deleted_staging_zip: bool = False
    success: bool = True
    error_message: str = ""


class ImpoPathNormalizer:
    """Resuelve el año (`Impo_YYYY`) y el mes (`<Mes>`) a partir de nombres de archivo DANE (SRP)."""

    @staticmethod
    def extract_year(archive_name: str) -> int:
        """Extrae el año de cuatro dígitos desde el nombre del archivo ZIP."""
        match = YEAR_REGEX.search(archive_name)
        if not match:
            raise ValueError(f"No se pudo determinar el año en el archivo: {archive_name}")
        return int(match.group(1))

    @staticmethod
    def extract_month(member_name: str) -> Optional[str]:
        """Extrae el nombre del mes normalizado en español desde una ruta interna del ZIP."""
        match = MONTH_REGEX.search(member_name)
        if not match:
            return None
        return match.group(1).capitalize()


class ImpoArchiveExtractor:
    """Servicio que descomprime paquetes anuales/semestrales DANE IMPO hacia `Impo_YYYY/<Mes>/` (SRP)."""

    def __init__(self, workspace_root: Path) -> None:
        self._root = workspace_root

    def process_outer_archive(
        self,
        outer_zip_path: Path,
        delete_outer_after_success: bool = True,
    ) -> ImpoYearExtractionSummary:
        """Extrae un paquete `Impo_YYYY*.zip` (con ZIPs anidados o carpetas directas) y valida integridad."""
        year = ImpoPathNormalizer.extract_year(outer_zip_path.name)
        year_dir = self._root / f"Impo_{year}"
        months_set = set()
        total_files = 0

        try:
            with zipfile.ZipFile(outer_zip_path, "r") as outer_zf:
                members = [m for m in outer_zf.infolist() if not m.is_dir()]
                if not members:
                    raise ValueError(f"El archivo ZIP externo está vacío: {outer_zip_path}")

                for member in members:
                    month_name = ImpoPathNormalizer.extract_month(member.filename)
                    if not month_name:
                        logger.warning(
                            "No se identificó mes para miembro %s en %s; omitiendo.",
                            member.filename,
                            outer_zip_path.name,
                        )
                        continue

                    target_month_dir = year_dir / month_name
                    target_month_dir.mkdir(parents=True, exist_ok=True)

                    if member.filename.lower().endswith(".zip"):
                        # Caso 1: ZIP mensual anidado dentro del ZIP anual/semestral
                        nested_bytes = outer_zf.read(member.filename)
                        with zipfile.ZipFile(io.BytesIO(nested_bytes), "r") as inner_zf:
                            for inner_member in inner_zf.infolist():
                                if inner_member.is_dir():
                                    continue
                                base_name = Path(inner_member.filename).name
                                dest_file = target_month_dir / base_name
                                with inner_zf.open(inner_member, "r") as src, open(dest_file, "wb") as dst:
                                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                                if not dest_file.exists() or dest_file.stat().st_size == 0:
                                    raise IOError(f"Archivo extraído vacío o fallido: {dest_file}")
                                total_files += 1
                        months_set.add(month_name)
                    else:
                        # Caso 2: Archivo plano (.csv, .dta, .sav, .txt) directo dentro del ZIP externo
                        base_name = Path(member.filename).name
                        dest_file = target_month_dir / base_name
                        with outer_zf.open(member, "r") as src, open(dest_file, "wb") as dst:
                            shutil.copyfileobj(src, dst, length=1024 * 1024)
                        if not dest_file.exists() or dest_file.stat().st_size == 0:
                            raise IOError(f"Archivo extraído vacío o fallido: {dest_file}")
                        total_files += 1
                        months_set.add(month_name)

            deleted = False
            if delete_outer_after_success:
                outer_zip_path.unlink()
                deleted = True

            sorted_months = sorted(months_set)
            logger.info(
                "Procesado %s -> Impo_%d (%d meses, %d archivos extraídos, staging eliminado=%s)",
                outer_zip_path.name,
                year,
                len(sorted_months),
                total_files,
                deleted,
            )
            return ImpoYearExtractionSummary(
                archive_name=outer_zip_path.name,
                year=year,
                months_extracted=sorted_months,
                files_extracted=total_files,
                deleted_staging_zip=deleted,
                success=True,
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("Error extrayendo %s: %s", outer_zip_path.name, exc)
            return ImpoYearExtractionSummary(
                archive_name=outer_zip_path.name,
                year=year,
                months_extracted=sorted(months_set),
                files_extracted=total_files,
                deleted_staging_zip=False,
                success=False,
                error_message=str(exc),
            )


def run_impo_extraction_pipeline(staging_dir: Path, workspace_root: Path) -> List[ImpoYearExtractionSummary]:
    """Ejecuta secuencialmente la extracción de todos los paquetes `Impo_*.zip` liberando espacio."""
    extractor = ImpoArchiveExtractor(workspace_root=workspace_root)
    zip_files = sorted(staging_dir.glob("Impo_*.zip"))
    logger.info("Total de paquetes anuales/semestrales IMPO a extraer: %d", len(zip_files))

    summaries: List[ImpoYearExtractionSummary] = []
    for zip_path in zip_files:
        summary = extractor.process_outer_archive(zip_path, delete_outer_after_success=True)
        summaries.append(summary)
        free_gb = shutil.disk_usage(workspace_root).free / (1024**3)
        logger.info("Espacio libre disponible en disco tras %s: %.2f GB", zip_path.name, free_gb)

    return summaries


if __name__ == "__main__":
    import sys

    workspace = Path(__file__).resolve().parents[2]
    staging = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else workspace / "data" / "raw_impo_zips"
    )
    run_impo_extraction_pipeline(staging, workspace)
