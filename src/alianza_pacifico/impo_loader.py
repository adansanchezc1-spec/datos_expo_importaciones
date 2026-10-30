"""Cargador y normalizador multi-año de microdatos de Importaciones (IMPO) del DANE.

Genera el cubo agregado `data/processed/importaciones_agregadas_2013_2025.csv`
con la misma estructura taxonómica y dimensional que `exportaciones_agregadas_2013_2025.csv`,
homologando la codificación de países DIAN (2013-2024) y UN M49 (2025) a ISO-3 (`COD_PAI4`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.alianza_pacifico.crispdm_pipeline import (
    ALIANZA_PACIFICO_COUNTRIES,
    TAXONOMIA_DEMANDADOS,
    TAXONOMIA_OFERTADOS,
    NumericNormalizer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class CountryCodeHomologator:
    """Homologa códigos numéricos de país DANE/DIAN (2013-2024) y M49 (2025) a ISO-3 (`COD_PAI4`) (SRP)."""

    # Respaldo explícito garantizado para socios de la Alianza del Pacífico y principales socios
    DIAN_TO_ISO3_FALLBACK: Dict[str, str] = {
        "211": "CHL",
        "493": "MEX",
        "589": "PER",
        "249": "USA",
        "215": "CHN",
        "105": "BRA",
        "239": "ECU",
        "023": "DEU",
        "245": "ESP",
        "399": "JPN",
        "169": "COL",
    }

    M49_TO_ISO3_FALLBACK: Dict[str, str] = {
        "152": "CHL",
        "484": "MEX",
        "604": "PER",
        "840": "USA",
        "156": "CHN",
        "076": "BRA",
        "218": "ECU",
        "276": "DEU",
        "724": "ESP",
        "392": "JPN",
        "170": "COL",
    }

    @classmethod
    def build_mapping_from_expo(cls, root_dir: Path) -> Dict[str, Dict[str, str]]:
        """Construye tablas de homologación `PAIS (3 dígitos) -> COD_PAI4` desde los microdatos Expo."""
        dian_map = dict(cls.DIAN_TO_ISO3_FALLBACK)
        m49_map = dict(cls.M49_TO_ISO3_FALLBACK)

        sample_dian_files = sorted(root_dir.glob("Expo_2018/*/*.dta"))[:3] + sorted(
            root_dir.glob("Expo_2024/*/*.dta")
        )[:3]
        for dta_path in sample_dian_files:
            df = pd.read_stata(
                dta_path,
                columns=["PAIS", "COD_PAI4"],
                convert_categoricals=False,
            ).dropna()
            codes_num = (
                pd.to_numeric(df["PAIS"], errors="coerce")
                .fillna(0)
                .astype(int)
                .astype(str)
                .str.zfill(3)
            )
            codes_iso = df["COD_PAI4"].astype(str).str.strip().str.upper()
            for num_code, iso_code in zip(codes_num, codes_iso):
                if num_code != "000" and len(iso_code) == 3:
                    dian_map[num_code] = iso_code

        sample_m49_files = sorted(root_dir.glob("Expo_2025/*/*.dta"))[:4]
        for dta_path in sample_m49_files:
            df = pd.read_stata(
                dta_path,
                columns=["PAIS", "COD_PAI4"],
                convert_categoricals=False,
            ).dropna()
            codes_num = (
                pd.to_numeric(df["PAIS"], errors="coerce")
                .fillna(0)
                .astype(int)
                .astype(str)
                .str.zfill(3)
            )
            codes_iso = df["COD_PAI4"].astype(str).str.strip().str.upper()
            for num_code, iso_code in zip(codes_num, codes_iso):
                if num_code != "000" and len(iso_code) == 3:
                    m49_map[num_code] = iso_code

        # Garantizar prioridad absoluta en códigos de Alianza del Pacífico
        dian_map.update(cls.DIAN_TO_ISO3_FALLBACK)
        m49_map.update(cls.M49_TO_ISO3_FALLBACK)
        return {"DIAN": dian_map, "M49": m49_map}


class ImpoDataLoader:
    """Ingesta multi-año de los 84 archivos mensuales de Importaciones (`Impo_*/*/*.dta`) (SRP)."""

    REQUIRED_COLS = ["FECH", "PAISGEN", "NABAN", "VAFODO", "VACID", "VACIP", "PNK", "CANU"]

    def __init__(self, root_dir: Path) -> None:
        self._root_dir = root_dir
        self._country_maps = CountryCodeHomologator.build_mapping_from_expo(root_dir)

    def load_all_years(self, filter_alliance_only: bool = False) -> pd.DataFrame:
        """Carga y consolida los 84 cortes mensuales de Importaciones (2013-2025) con esquema homólogo a Expo."""
        dta_files = sorted(self._root_dir.glob("Impo_*/*/*.dta"))
        logger.info("Cargando %d archivos mensuales de Importaciones (.dta)...", len(dta_files))

        monthly_chunks: List[pd.DataFrame] = []
        for idx, dta_path in enumerate(dta_files, start=1):
            year_dir = dta_path.parents[1].name  # Impo_YYYY
            year_num = int(year_dir.split("_")[1])
            regime = "M49" if year_num >= 2025 else "DIAN"
            code_map = self._country_maps[regime]

            df = pd.read_stata(
                dta_path,
                columns=self.REQUIRED_COLS,
                convert_categoricals=False,
            )

            pais_num = (
                pd.to_numeric(df["PAISGEN"], errors="coerce")
                .fillna(0)
                .astype(int)
                .astype(str)
                .str.zfill(3)
            )
            df["COD_PAI4"] = pais_num.map(code_map).fillna(pais_num)

            if filter_alliance_only:
                df = df[df["COD_PAI4"].isin(ALIANZA_PACIFICO_COUNTRIES.keys())].copy()

            df["POSAR"] = NumericNormalizer.normalize_posar(df["NABAN"])
            df["VAFODO"] = NumericNormalizer.to_float(df["VAFODO"])
            df["VACID"] = NumericNormalizer.to_float(df["VACID"])
            df["VACIP"] = NumericNormalizer.to_float(df["VACIP"])
            df["PNK"] = NumericNormalizer.to_float(df["PNK"])
            df["CANU"] = NumericNormalizer.to_float(df["CANU"])

            # Estimación consistente de FOB en pesos a partir de la relación FOB/CIF en dólares
            ratio_fob_cif = np.where(
                df["VACID"] > 0,
                (df["VAFODO"] / df["VACID"]).clip(lower=0.0, upper=1.0),
                1.0,
            )
            df["FOBPES_EST"] = df["VACIP"] * ratio_fob_cif

            fech_num = pd.to_numeric(df["FECH"], errors="coerce").fillna(0).astype(int)
            df["anio"] = year_num
            df["mes"] = (fech_num % 100).clip(lower=1, upper=12)
            df["hs4"] = df["POSAR"].str[:4]
            df["es_alianza"] = df["COD_PAI4"].isin(ALIANZA_PACIFICO_COUNTRIES.keys())

            df["tipo_canasta"] = np.select(
                [
                    df["hs4"].isin(TAXONOMIA_OFERTADOS.keys()),
                    df["hs4"].isin(TAXONOMIA_DEMANDADOS.keys()),
                ],
                ["Canasta Ofertada", "Canasta Demandada (Referencia)"],
                default="Resto de Importaciones",
            )
            df["producto_clave"] = (
                df["hs4"]
                .map({**TAXONOMIA_OFERTADOS, **TAXONOMIA_DEMANDADOS})
                .fillna("Otros Productos")
            )

            grouped = (
                df.groupby(
                    [
                        "anio",
                        "mes",
                        "COD_PAI4",
                        "es_alianza",
                        "tipo_canasta",
                        "producto_clave",
                        "hs4",
                        "POSAR",
                    ],
                    as_index=False,
                )
                .agg(
                    fob_usd=("VAFODO", "sum"),
                    fob_cop=("FOBPES_EST", "sum"),
                    cif_usd=("VACID", "sum"),
                    cif_cop=("VACIP", "sum"),
                    peso_neto_kg=("PNK", "sum"),
                    cantidad=("CANU", "sum"),
                    num_transacciones=("VAFODO", "count"),
                )
            )
            monthly_chunks.append(grouped)
            if idx % 12 == 0 or idx == len(dta_files):
                logger.info(
                    "Progreso consolidación IMPO: %d/%d archivos mensuales procesados (último: %s/%s)",
                    idx,
                    len(dta_files),
                    year_dir,
                    dta_path.parent.name,
                )

        full_df = pd.concat(monthly_chunks, ignore_index=True)
        logger.info("Consolidación IMPO completada: %d registros agregados.", len(full_df))
        return full_df


def build_impo_aggregated_cache(root_dir: Path) -> Path:
    """Genera y guarda el archivo `data/processed/importaciones_agregadas_2013_2025.csv`."""
    processed_dir = root_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "importaciones_agregadas_2013_2025.csv"

    loader = ImpoDataLoader(root_dir=root_dir)
    df_impo = loader.load_all_years(filter_alliance_only=False)
    df_impo.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Caché agregado de importaciones guardado en %s", output_path)
    return output_path


if __name__ == "__main__":
    workspace = Path(__file__).resolve().parents[2]
    build_impo_aggregated_cache(workspace)
