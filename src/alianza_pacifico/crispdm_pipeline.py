"""Pipeline CRISP-DM Bilateral Desagregado por Producto (Exportaciones e Importaciones) — Alianza del Pacífico (2013-2025).

Implementa bajo principios SOLID, Clean Code, PEP 8 y DAMA-BOK:
1. Ingesta y cruce bilateral de Exportaciones (1.11M filas) e Importaciones (2.70M filas).
2. Desagregación exhaustiva en los 8 Productos Ofertados y los 8 Productos Demandados clave (HS4).
3. Ingeniería de Características (IPX, IPM, HHI Invertido, Margen Extensivo, SCR, Grubel-Lloyd, Theta).
4. Diagnóstico Probabilístico con `distfit` y pruebas de normalidad a nivel de panel y por cada uno de los 16 productos.
5. Triangulación Paramétrica (t-Pareada, ANOVA) y No Paramétrica (Wilcoxon, Signos, Kruskal-Wallis) global y por producto.
6. Econometría de Panel con Efectos Fijos (FE-OLS HC3) e Inferencia Causal con Double Machine Learning (DML).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from distfit import distfit
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, TimeSeriesSplit
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Países socios de Colombia en la Alianza del Pacífico
ALIANZA_PACIFICO_COUNTRIES: Dict[str, str] = {
    "MEX": "México",
    "PER": "Perú",
    "CHL": "Chile",
}

# 8 Principales Productos Ofertados (Exportados por Colombia hacia la Alianza del Pacífico)
PRODUCTOS_OFERTADOS_LIST: List[str] = [
    "Hullas, briquetas y combustibles sólidos similares obtenidos de la hulla",
    "Minerales de cobre y sus concentrados",
    "Insecticidas, fungicidas, herbicidas y similares",
    "Aceites crudos de petróleo",
    "Minerales de metales preciosos y sus concentrados (oro y cobre)",
    "Aceite de palma y sus fracciones",
    "Preparaciones de belleza, maquillaje y cuidado de la piel",
    "Aceites de petróleo (excepto crudos)",
]

# 8 Principales Productos Demandados (Importados por Colombia desde la Alianza del Pacífico)
PRODUCTOS_DEMANDADOS_LIST: List[str] = [
    "Automóviles de turismo y demás vehículos para transporte de personas",
    "Monitores, proyectores y aparatos receptores de televisión",
    "Alambre de cobre",
    "Medicamentos dosificados",
    "Manzanas, peras y membrillos frescos",
    "Máquinas automáticas para procesamiento de datos y sus unidades",
    "Tubos y perfiles huecos sin soldadura, de hierro o acero",
    "Tractores",
]

# Taxonomía Arancelaria (HS4 - primeros 4 dígitos de POSAR/NABAN a 10 dígitos)
TAXONOMIA_OFERTADOS: Dict[str, str] = {
    "2701": "Hullas, briquetas y combustibles sólidos similares obtenidos de la hulla",
    "2603": "Minerales de cobre y sus concentrados",
    "3808": "Insecticidas, fungicidas, herbicidas y similares",
    "2709": "Aceites crudos de petróleo",
    "2616": "Minerales de metales preciosos y sus concentrados (oro y cobre)",
    "7108": "Minerales de metales preciosos y sus concentrados (oro y cobre)",
    "1511": "Aceite de palma y sus fracciones",
    "3304": "Preparaciones de belleza, maquillaje y cuidado de la piel",
    "2710": "Aceites de petróleo (excepto crudos)",
}

TAXONOMIA_DEMANDADOS: Dict[str, str] = {
    "8703": "Automóviles de turismo y demás vehículos para transporte de personas",
    "8704": "Automóviles de turismo y demás vehículos para transporte de personas",
    "8528": "Monitores, proyectores y aparatos receptores de televisión",
    "7408": "Alambre de cobre",
    "3004": "Medicamentos dosificados",
    "0808": "Manzanas, peras y membrillos frescos",
    "8471": "Máquinas automáticas para procesamiento de datos y sus unidades",
    "7304": "Tubos y perfiles huecos sin soldadura, de hierro o acero",
    "7306": "Tubos y perfiles huecos sin soldadura, de hierro o acero",
    "8701": "Tractores",
}


@dataclass(frozen=True)
class HypothesisTestResult:
    """Estructura inmutable con resultados de contrastación paramétrica y no paramétrica."""

    comparison: str
    metric: str
    n_pairs: int
    mean_diff: float
    median_diff: float
    t_stat: float
    t_pvalue: float
    wilcoxon_stat: float
    wilcoxon_pvalue: float
    sign_positive: int
    sign_negative: int
    sign_pvalue: float
    reject_h0_alpha_05: bool


class NumericNormalizer:
    """Convierte columnas numéricas con formatos mixtos y normaliza subpartidas POSAR/NABAN (SRP)."""

    @staticmethod
    def to_float(series: pd.Series) -> pd.Series:
        """Limpia cadenas con separador de miles '.' y decimal ',' a float64."""
        if pd.api.types.is_numeric_dtype(series):
            return series.astype(float).fillna(0.0)
        cleaned = (
            series.astype(str)
            .str.strip()
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)

    @staticmethod
    def normalize_posar(series: pd.Series) -> pd.Series:
        """Normaliza la posición arancelaria (POSAR/NABAN) a cadena de 10 dígitos."""
        if pd.api.types.is_numeric_dtype(series):
            return (
                series.fillna(0)
                .astype(np.int64)
                .astype(str)
                .str.zfill(10)
            )
        cleaned = (
            series.astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .str.replace(".", "", regex=False)
        )
        return cleaned.str.zfill(10)

    @staticmethod
    def harmonize_product_labels(df: pd.DataFrame) -> pd.DataFrame:
        """Asegura que `hs4`, `producto_clave` y `tipo_canasta` usen los nombres exactos de los 16 productos."""
        df = df.copy()
        df["POSAR"] = df["POSAR"].astype(str).str.zfill(10)
        df["hs4"] = df["POSAR"].str[:4]
        df["producto_clave"] = (
            df["hs4"]
            .map({**TAXONOMIA_OFERTADOS, **TAXONOMIA_DEMANDADOS})
            .fillna("Otros Productos")
        )
        df["tipo_canasta"] = np.select(
            [
                df["hs4"].isin(TAXONOMIA_OFERTADOS.keys()),
                df["hs4"].isin(TAXONOMIA_DEMANDADOS.keys()),
            ],
            ["Canasta Ofertada", "Canasta Demandada (Referencia)"],
            default="Resto de Exportaciones",
        )
        return df


class TradeDataLoader:
    """Ingesta multi-año de archivos mensuales DANE Exportaciones (.dta) bajo DAMA-BOK (SRP)."""

    REQUIRED_COLS = ["FECH", "COD_PAI4", "POSAR", "FOBDOL", "FOBPES", "PNK", "CANTI"]

    def __init__(self, root_dir: Path) -> None:
        self._root_dir = root_dir

    def load_all_years(self, filter_alliance_only: bool = False) -> pd.DataFrame:
        """Carga y consolida los 84 archivos mensuales de Exportaciones (2013-2025)."""
        dta_files = sorted(self._root_dir.glob("Expo_*/*/*.dta"))
        logger.info("Cargando %d archivos mensuales EXPO (.dta)...", len(dta_files))

        monthly_chunks: List[pd.DataFrame] = []
        for dta_path in dta_files:
            year_dir = dta_path.parents[1].name
            year_num = int(year_dir.split("_")[1])

            df = pd.read_stata(
                dta_path,
                columns=self.REQUIRED_COLS,
                convert_categoricals=False,
            )
            df["COD_PAI4"] = df["COD_PAI4"].astype(str).str.strip().str.upper()

            if filter_alliance_only:
                df = df[df["COD_PAI4"].isin(ALIANZA_PACIFICO_COUNTRIES.keys())].copy()

            df["POSAR"] = NumericNormalizer.normalize_posar(df["POSAR"])
            df["FOBDOL"] = NumericNormalizer.to_float(df["FOBDOL"])
            df["FOBPES"] = NumericNormalizer.to_float(df["FOBPES"])
            df["PNK"] = NumericNormalizer.to_float(df["PNK"])
            df["CANTI"] = NumericNormalizer.to_float(df["CANTI"])

            fech_num = pd.to_numeric(df["FECH"], errors="coerce").fillna(0).astype(int)
            df["anio"] = year_num
            df["mes"] = (fech_num % 100).clip(lower=1, upper=12)
            df["hs4"] = df["POSAR"].str[:4]
            df["es_alianza"] = df["COD_PAI4"].isin(ALIANZA_PACIFICO_COUNTRIES.keys())

            df = NumericNormalizer.harmonize_product_labels(df)

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
                    fob_usd=("FOBDOL", "sum"),
                    fob_cop=("FOBPES", "sum"),
                    peso_neto_kg=("PNK", "sum"),
                    cantidad=("CANTI", "sum"),
                    num_transacciones=("FOBDOL", "count"),
                )
            )
            monthly_chunks.append(grouped)

        full_df = pd.concat(monthly_chunks, ignore_index=True)
        logger.info("Consolidación EXPO completada: %d registros agregados.", len(full_df))
        return full_df


class FeatureEngineeringPipeline:
    """Construye el panel bilateral (EXPO + IMPO) y los cubos desagregados para los 16 productos clave (SRP)."""

    @staticmethod
    def _compute_deflated_flows(df_ap: pd.DataFrame, index_col: str) -> pd.DataFrame:
        """Calcula el índice de valor unitario por capítulo HS2 (Base 2013=100, acotado en [50, 250])."""
        df_ap = df_ap.copy()
        df_ap["hs2"] = df_ap["POSAR"].str[:2]
        uv_hs2 = (
            df_ap.groupby(["anio", "hs2"], as_index=False)
            .agg(fob_total=("fob_usd", "sum"), pnk_total=("peso_neto_kg", "sum"))
        )
        uv_hs2["unit_value"] = np.where(
            uv_hs2["pnk_total"] > 0,
            uv_hs2["fob_total"] / uv_hs2["pnk_total"],
            1.0,
        )
        base_2013 = (
            uv_hs2[uv_hs2["anio"] == 2013][["hs2", "unit_value"]]
            .rename(columns={"unit_value": "uv_base_2013"})
        )
        uv_hs2 = uv_hs2.merge(base_2013, on="hs2", how="left")
        uv_hs2["uv_base_2013"] = uv_hs2["uv_base_2013"].fillna(uv_hs2["unit_value"])
        uv_hs2[index_col] = (
            (uv_hs2["unit_value"] / uv_hs2["uv_base_2013"]) * 100.0
        ).clip(lower=50.0, upper=250.0)

        df_ap = df_ap.merge(uv_hs2[["anio", "hs2", index_col]], on=["anio", "hs2"], how="left")
        df_ap[index_col] = df_ap[index_col].fillna(100.0)
        df_ap["real_usd"] = (df_ap["fob_usd"] / df_ap[index_col]) * 100.0
        return df_ap

    @classmethod
    def build_panel_features(
        cls,
        df_expo: pd.DataFrame,
        df_impo: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Construye el panel econométrico mensual de la Alianza del Pacífico ($N = 720$)."""
        df_ap_x = df_expo[df_expo["es_alianza"]].copy()
        df_ap_x = NumericNormalizer.harmonize_product_labels(df_ap_x)
        df_ap_x["pais_nombre"] = df_ap_x["COD_PAI4"].map(ALIANZA_PACIFICO_COUNTRIES)
        df_ap_x["fecha"] = pd.to_datetime(
            df_ap_x["anio"].astype(str) + "-" + df_ap_x["mes"].astype(str).str.zfill(2) + "-01"
        )
        df_ap_x = cls._compute_deflated_flows(df_ap_x, index_col="ipx")

        def _agg_expo_group(group: pd.DataFrame) -> pd.Series:
            active = group[group["fob_usd"] > 0]
            total_fob = active["fob_usd"].sum()
            total_real = active["real_usd"].sum()
            n_subpartidas = active["POSAR"].nunique()
            if total_fob > 0 and n_subpartidas > 1:
                shares = active.groupby("POSAR")["fob_usd"].sum() / total_fob
                hhi_inv = float(1.0 - np.sum(np.square(shares)))
            else:
                hhi_inv = 0.0
            return pd.Series(
                {
                    "fob_usd": total_fob,
                    "exportaciones_reales": total_real,
                    "peso_neto_kg": active["peso_neto_kg"].sum(),
                    "ipx_promedio": active["ipx"].mean() if len(active) else 100.0,
                    "n_subpartidas_activas": int(n_subpartidas),
                    "hhi_invertido": hhi_inv,
                }
            )

        panel = (
            df_ap_x.groupby(
                ["anio", "mes", "fecha", "COD_PAI4", "pais_nombre", "tipo_canasta"],
                as_index=False,
            )
            .apply(_agg_expo_group, include_groups=False)
        )

        pivot_rel = (
            panel.pivot_table(
                index=["anio", "mes", "fecha", "COD_PAI4", "pais_nombre"],
                columns="tipo_canasta",
                values=["fob_usd", "exportaciones_reales", "n_subpartidas_activas", "hhi_invertido"],
                aggfunc="sum",
                fill_value=0.0,
            )
        )
        pivot_rel.columns = [f"{col[0]}__{col[1]}" for col in pivot_rel.columns]
        pivot_rel = pivot_rel.reset_index()

        x_real_ofertada = pivot_rel.get(
            "exportaciones_reales__Canasta Ofertada",
            pd.Series(0.0, index=pivot_rel.index),
        )
        m_ref_real = pivot_rel.get(
            "exportaciones_reales__Canasta Demandada (Referencia)",
            pd.Series(0.0, index=pivot_rel.index),
        )
        denom = x_real_ofertada + m_ref_real
        pivot_rel["scr"] = np.where(denom == 0, 0.0, (x_real_ofertada - m_ref_real) / denom)
        pivot_rel["theta"] = np.log((x_real_ofertada + 1.0) / (m_ref_real + 1.0))

        panel = panel.merge(
            pivot_rel[["anio", "mes", "COD_PAI4", "scr", "theta"]],
            on=["anio", "mes", "COD_PAI4"],
            how="left",
        )

        if df_impo is not None and not df_impo.empty:
            df_ap_m = df_impo[df_impo["COD_PAI4"].isin(ALIANZA_PACIFICO_COUNTRIES.keys())].copy()
            df_ap_m = NumericNormalizer.harmonize_product_labels(df_ap_m)
            df_ap_m = cls._compute_deflated_flows(df_ap_m, index_col="ipm")

            def _agg_impo_group(group: pd.DataFrame) -> pd.Series:
                active = group[group["fob_usd"] > 0]
                total_fob = active["fob_usd"].sum()
                total_cif = active["cif_usd"].sum() if "cif_usd" in active.columns else total_fob
                total_real = active["real_usd"].sum()
                n_sub = active["POSAR"].nunique()
                if total_fob > 0 and n_sub > 1:
                    shares = active.groupby("POSAR")["fob_usd"].sum() / total_fob
                    hhi_inv = float(1.0 - np.sum(np.square(shares)))
                else:
                    hhi_inv = 0.0
                return pd.Series(
                    {
                        "impo_fob_usd": total_fob,
                        "impo_cif_usd": total_cif,
                        "importaciones_reales": total_real,
                        "impo_peso_neto_kg": active["peso_neto_kg"].sum(),
                        "ipm_promedio": active["ipm"].mean() if len(active) else 100.0,
                        "impo_n_subpartidas": int(n_sub),
                        "impo_hhi_invertido": hhi_inv,
                    }
                )

            impo_panel = (
                df_ap_m.groupby(
                    ["anio", "mes", "COD_PAI4", "tipo_canasta"],
                    as_index=False,
                )
                .apply(_agg_impo_group, include_groups=False)
            )
            panel = panel.merge(
                impo_panel,
                on=["anio", "mes", "COD_PAI4", "tipo_canasta"],
                how="left",
            )
            for col_fill, default_val in [
                ("impo_fob_usd", 0.0),
                ("impo_cif_usd", 0.0),
                ("importaciones_reales", 0.0),
                ("impo_peso_neto_kg", 0.0),
                ("ipm_promedio", 100.0),
                ("impo_n_subpartidas", 0),
                ("impo_hhi_invertido", 0.0),
            ]:
                panel[col_fill] = panel[col_fill].fillna(default_val)
            panel["impo_n_subpartidas"] = panel["impo_n_subpartidas"].astype(int)
        else:
            panel["impo_fob_usd"] = 0.0
            panel["impo_cif_usd"] = 0.0
            panel["importaciones_reales"] = 0.0
            panel["impo_peso_neto_kg"] = 0.0
            panel["ipm_promedio"] = 100.0
            panel["impo_n_subpartidas"] = 0
            panel["impo_hhi_invertido"] = 0.0

        denom_bil = panel["exportaciones_reales"] + panel["importaciones_reales"]
        panel["scr_bilateral"] = np.where(
            denom_bil > 0,
            (panel["exportaciones_reales"] - panel["importaciones_reales"]) / denom_bil,
            0.0,
        )
        panel["grubel_lloyd"] = np.where(
            denom_bil > 0,
            1.0 - np.abs(panel["scr_bilateral"]),
            0.0,
        )
        panel["theta_bilateral"] = np.log(
            (panel["exportaciones_reales"] + 1.0) / (panel["importaciones_reales"] + 1.0)
        )

        panel["bloque_temporal"] = np.select(
            [
                panel["anio"].isin([2013, 2014]),
                panel["anio"].isin([2018, 2019, 2020]),
                panel["anio"].isin([2024, 2025]),
            ],
            [
                "P1: Pre-Consolidación (2013-2014)",
                "P2: Consolidación/COVID (2018-2020)",
                "P3: Madurez Reciente (2024-2025)",
            ],
            default="Otro",
        )
        return panel

    @classmethod
    def build_product_monthly_and_block_tables(
        cls,
        df_expo: pd.DataFrame,
        df_impo: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Construye el panel mensual de los 16 productos clave y el resumen por bloque histórico."""
        ap_x = NumericNormalizer.harmonize_product_labels(df_expo[df_expo["es_alianza"]].copy())
        ap_m = NumericNormalizer.harmonize_product_labels(
            df_impo[df_impo["COD_PAI4"].isin(ALIANZA_PACIFICO_COUNTRIES.keys())].copy()
        )

        ap_x = cls._compute_deflated_flows(ap_x, index_col="ipx")
        ap_m = cls._compute_deflated_flows(ap_m, index_col="ipm")

        ap_x_key = ap_x[ap_x["producto_clave"] != "Otros Productos"].copy()
        ap_m_key = ap_m[ap_m["producto_clave"] != "Otros Productos"].copy()

        # Grilla cartesiana completa: 7 años x 12 meses x 3 socios x 16 productos = 4,032 celdas
        all_years = sorted(ap_x["anio"].unique())
        all_months = list(range(1, 13))
        all_countries = list(ALIANZA_PACIFICO_COUNTRIES.keys())
        product_meta = [
            ("Canasta Ofertada", p) for p in PRODUCTOS_OFERTADOS_LIST
        ] + [
            ("Canasta Demandada (Referencia)", p) for p in PRODUCTOS_DEMANDADOS_LIST
        ]

        grid_rows = [
            {
                "anio": y,
                "mes": m,
                "COD_PAI4": c,
                "tipo_canasta": tc,
                "producto_clave": prod,
            }
            for y in all_years
            for m in all_months
            for c in all_countries
            for tc, prod in product_meta
        ]
        grid_df = pd.DataFrame(grid_rows)

        grp_cols = ["anio", "mes", "COD_PAI4", "tipo_canasta", "producto_clave"]
        agg_x = (
            ap_x_key.groupby(grp_cols, as_index=False)
            .agg(
                expo_fob_usd=("fob_usd", "sum"),
                expo_real_usd=("real_usd", "sum"),
                expo_peso_kg=("peso_neto_kg", "sum"),
                expo_subpartidas_hs10=("POSAR", "nunique"),
                expo_transacciones=("num_transacciones", "sum"),
            )
        )
        agg_m = (
            ap_m_key.groupby(grp_cols, as_index=False)
            .agg(
                impo_fob_usd=("fob_usd", "sum"),
                impo_cif_usd=("cif_usd", "sum"),
                impo_real_usd=("real_usd", "sum"),
                impo_peso_kg=("peso_neto_kg", "sum"),
                impo_subpartidas_hs10=("POSAR", "nunique"),
                impo_transacciones=("num_transacciones", "sum"),
            )
        )

        monthly_prod = grid_df.merge(agg_x, on=grp_cols, how="left").merge(agg_m, on=grp_cols, how="left")
        num_cols = [c for c in monthly_prod.columns if c not in grp_cols]
        monthly_prod[num_cols] = monthly_prod[num_cols].fillna(0.0)
        monthly_prod["expo_subpartidas_hs10"] = monthly_prod["expo_subpartidas_hs10"].astype(int)
        monthly_prod["impo_subpartidas_hs10"] = monthly_prod["impo_subpartidas_hs10"].astype(int)
        monthly_prod["pais_nombre"] = monthly_prod["COD_PAI4"].map(ALIANZA_PACIFICO_COUNTRIES)
        monthly_prod["fecha"] = pd.to_datetime(
            monthly_prod["anio"].astype(str) + "-" + monthly_prod["mes"].astype(str).str.zfill(2) + "-01"
        )

        # Códigos HS4 asociados a cada producto clave
        prod_to_hs4: Dict[str, str] = {}
        for h_code, p_name in {**TAXONOMIA_OFERTADOS, **TAXONOMIA_DEMANDADOS}.items():
            prod_to_hs4.setdefault(p_name, [])
            prod_to_hs4[p_name].append(h_code)
        monthly_prod["hs4"] = monthly_prod["producto_clave"].map(
            lambda p: "/".join(sorted(prod_to_hs4.get(p, [])))
        )

        monthly_prod["bloque_temporal"] = np.select(
            [
                monthly_prod["anio"].isin([2013, 2014]),
                monthly_prod["anio"].isin([2018, 2019, 2020]),
                monthly_prod["anio"].isin([2024, 2025]),
            ],
            [
                "P1: Pre-Consolidación (2013-2014)",
                "P2: Consolidación/COVID (2018-2020)",
                "P3: Madurez Reciente (2024-2025)",
            ],
            default="Otro",
        )

        denom = monthly_prod["expo_real_usd"] + monthly_prod["impo_real_usd"]
        monthly_prod["balanza_real_usd"] = monthly_prod["expo_real_usd"] - monthly_prod["impo_real_usd"]
        monthly_prod["scr_bilateral"] = np.where(
            denom > 0,
            monthly_prod["balanza_real_usd"] / denom,
            0.0,
        )
        monthly_prod["grubel_lloyd"] = np.where(
            denom > 0,
            1.0 - np.abs(monthly_prod["scr_bilateral"]),
            0.0,
        )
        monthly_prod["theta_bilateral"] = np.log(
            (monthly_prod["expo_real_usd"] + 1.0) / (monthly_prod["impo_real_usd"] + 1.0)
        )

        # Flujo vocacional primario: Exportaciones Reales para Ofertados, Importaciones Reales para Demandados
        monthly_prod["flujo_vocacional_real_usd"] = np.where(
            monthly_prod["tipo_canasta"] == "Canasta Ofertada",
            monthly_prod["expo_real_usd"],
            monthly_prod["impo_real_usd"],
        )

        # Tabla agregada por bloque temporal, país y producto (T4)
        block_cols = ["bloque_temporal", "COD_PAI4", "pais_nombre", "tipo_canasta", "producto_clave", "hs4"]
        block_prod = (
            monthly_prod.groupby(block_cols, as_index=False)
            .agg(
                expo_fob_usd=("expo_fob_usd", "sum"),
                expo_real_usd=("expo_real_usd", "sum"),
                expo_peso_kg=("expo_peso_kg", "sum"),
                expo_subpartidas_hs10=("expo_subpartidas_hs10", "max"),
                expo_transacciones=("expo_transacciones", "sum"),
                impo_fob_usd=("impo_fob_usd", "sum"),
                impo_cif_usd=("impo_cif_usd", "sum"),
                impo_real_usd=("impo_real_usd", "sum"),
                impo_peso_kg=("impo_peso_kg", "sum"),
                impo_subpartidas_hs10=("impo_subpartidas_hs10", "max"),
                impo_transacciones=("impo_transacciones", "sum"),
            )
        )
        b_denom = block_prod["expo_real_usd"] + block_prod["impo_real_usd"]
        block_prod["balanza_real_usd"] = block_prod["expo_real_usd"] - block_prod["impo_real_usd"]
        block_prod["scr_bilateral"] = np.where(
            b_denom > 0, block_prod["balanza_real_usd"] / b_denom, 0.0
        )
        block_prod["grubel_lloyd"] = np.where(
            b_denom > 0, 1.0 - np.abs(block_prod["scr_bilateral"]), 0.0
        )
        block_prod["theta_bilateral"] = np.log(
            (block_prod["expo_real_usd"] + 1.0) / (block_prod["impo_real_usd"] + 1.0)
        )

        return monthly_prod, block_prod


class DistributionAnalyzer:
    """Ajusta distribuciones paramétricas con `distfit` a nivel global y para cada uno de los 16 productos (SRP)."""

    CANDIDATE_DISTS = ["norm", "lognorm", "gamma", "weibull_min", "expon", "pareto", "t", "beta"]

    @classmethod
    def fit_all_distributions(cls, panel: pd.DataFrame) -> pd.DataFrame:
        """Evalúa normalidad y mejor ajuste probabilístico para las variables clave del panel."""
        df_pos = panel[panel["exportaciones_reales"] > 0].copy()
        df_pos["ln_exportaciones_reales"] = np.log(df_pos["exportaciones_reales"])
        df_pos["ln_importaciones_reales"] = np.log(df_pos["importaciones_reales"].clip(lower=1.0))
        df_pos["ln_n_subpartidas"] = np.log(df_pos["n_subpartidas_activas"].clip(lower=1))

        series_map = {
            "Exportaciones Reales (USD Niveles)": df_pos["exportaciones_reales"].values,
            "ln(Exportaciones Reales USD)": df_pos["ln_exportaciones_reales"].values,
            "Importaciones Reales (USD Niveles)": df_pos[df_pos["importaciones_reales"] > 0]["importaciones_reales"].values,
            "ln(Importaciones Reales USD)": df_pos[df_pos["importaciones_reales"] > 0]["ln_importaciones_reales"].values,
            "HHI Invertido Exportador (D_it)": df_pos["hhi_invertido"].values,
            "Margen Extensivo Exportador (N_it)": df_pos["n_subpartidas_activas"].astype(float).values,
            "ln(Margen Extensivo N_it)": df_pos["ln_n_subpartidas"].values,
            "Razón Logarítmica Bilateral (Theta_bil)": df_pos["theta_bilateral"].values,
        }

        records: List[Dict[str, Any]] = []
        for var_label, arr in series_map.items():
            clean_arr = arr[np.isfinite(arr)]
            shapiro_stat, shapiro_p = stats.shapiro(clean_arr[:5000])
            dagostino_stat, dagostino_p = stats.normaltest(clean_arr)
            jb_stat, jb_p = stats.jarque_bera(clean_arr)
            skew_val = float(stats.skew(clean_arr))
            kurt_val = float(stats.kurtosis(clean_arr))

            dfit = distfit(distr=cls.CANDIDATE_DISTS, verbose=0)
            dfit.fit_transform(clean_arr)
            best_name = str(dfit.model["name"])
            best_rss = float(dfit.model["score"])
            second_name = (
                str(dfit.summary.iloc[1]["name"])
                if hasattr(dfit, "summary") and len(dfit.summary) > 1
                else "N/A"
            )

            records.append(
                {
                    "variable": var_label,
                    "n_obs": len(clean_arr),
                    "media": float(np.mean(clean_arr)),
                    "mediana": float(np.median(clean_arr)),
                    "desv_estandar": float(np.std(clean_arr, ddof=1)),
                    "asimetria_skewness": skew_val,
                    "curtosis_exceso": kurt_val,
                    "shapiro_w": float(shapiro_stat),
                    "shapiro_pvalue": float(shapiro_p),
                    "dagostino_k2": float(dagostino_stat),
                    "dagostino_pvalue": float(dagostino_p),
                    "jarque_bera_stat": float(jb_stat),
                    "jarque_bera_pvalue": float(jb_p),
                    "es_normal_alpha_01": bool((shapiro_p >= 0.01) and (dagostino_p >= 0.01)),
                    "mejor_distribucion_distfit": best_name,
                    "rss_mejor_dist": best_rss,
                    "segunda_distribucion_distfit": second_name,
                    "enfoque_recomendado": (
                        "Paramétrico + No Paramétrico"
                        if abs(skew_val) < 1.2
                        else "No Paramétrico (Rangos/Log-Escala)"
                    ),
                }
            )

        return pd.DataFrame(records)

    @classmethod
    def fit_product_distributions(cls, monthly_prod: pd.DataFrame) -> pd.DataFrame:
        """Ajusta `distfit` y pruebas de normalidad individualmente para cada uno de los 16 productos clave."""
        rows: List[Dict[str, Any]] = []
        for (canasta, prod), grp in monthly_prod.groupby(["tipo_canasta", "producto_clave"]):
            # Agregamos por mes (sumando los 3 socios) para evaluar la serie mensual de 84 cortes de cada producto
            ts_84 = (
                grp.groupby(["anio", "mes"], as_index=False)
                .agg(
                    expo_real_musd=("expo_real_usd", lambda s: s.sum() / 1e6),
                    impo_real_musd=("impo_real_usd", lambda s: s.sum() / 1e6),
                    vocacional_musd=("flujo_vocacional_real_usd", lambda s: s.sum() / 1e6),
                )
            )
            arr = ts_84["vocacional_musd"].values
            pos_arr = arr[arr > 0]
            if len(pos_arr) < 8:
                pos_arr = arr + 1e-6

            shapiro_stat, shapiro_p = stats.shapiro(arr)
            skew_val = float(stats.skew(arr))
            kurt_val = float(stats.kurtosis(arr))

            dfit = distfit(distr=cls.CANDIDATE_DISTS, verbose=0)
            dfit.fit_transform(pos_arr)
            best_dist = str(dfit.model["name"])
            rss_val = float(dfit.model["score"])

            rows.append(
                {
                    "tipo_canasta": canasta,
                    "producto_clave": prod,
                    "flujo_evaluado": "Exportaciones Reales (MUSD)" if canasta == "Canasta Ofertada" else "Importaciones Reales (MUSD)",
                    "meses_activos_de_84": int((arr > 0).sum()),
                    "media_mensual_musd": float(np.mean(arr)),
                    "mediana_mensual_musd": float(np.median(arr)),
                    "iqr_mensual_musd": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
                    "asimetria_skewness": skew_val,
                    "curtosis_exceso": kurt_val,
                    "shapiro_pvalue": float(shapiro_p),
                    "mejor_ley_distfit": best_dist,
                    "rss_distfit": rss_val,
                }
            )

        return pd.DataFrame(rows).sort_values(
            ["tipo_canasta", "media_mensual_musd"], ascending=[True, False]
        ).reset_index(drop=True)


class EconometricModeler:
    """Ejecuta pruebas paramétricas, no paramétricas (global y por producto), FE-OLS y ML Causal (SRP)."""

    @staticmethod
    def run_paired_tests(panel: pd.DataFrame) -> pd.DataFrame:
        """Contrasta H0 vs H1 mediante muestras emparejadas por (COD_PAI4, mes, tipo_canasta)."""
        comparisons = [
            ("P1: Pre-Consolidación (2013-2014)", "P2: Consolidación/COVID (2018-2020)"),
            ("P2: Consolidación/COVID (2018-2020)", "P3: Madurez Reciente (2024-2025)"),
            ("P1: Pre-Consolidación (2013-2014)", "P3: Madurez Reciente (2024-2025)"),
        ]
        metrics = [
            "exportaciones_reales",
            "importaciones_reales",
            "hhi_invertido",
            "n_subpartidas_activas",
            "theta",
            "theta_bilateral",
            "grubel_lloyd",
        ]
        available_metrics = [m for m in metrics if m in panel.columns]

        grouped_period = (
            panel.groupby(["bloque_temporal", "COD_PAI4", "mes", "tipo_canasta"], as_index=False)[
                available_metrics
            ]
            .mean()
        )

        results: List[HypothesisTestResult] = []
        for p_start, p_end in comparisons:
            df_a = grouped_period[grouped_period["bloque_temporal"] == p_start]
            df_b = grouped_period[grouped_period["bloque_temporal"] == p_end]
            merged = pd.merge(
                df_a,
                df_b,
                on=["COD_PAI4", "mes", "tipo_canasta"],
                suffixes=("_pre", "_post"),
            )
            for metric in available_metrics:
                diffs = merged[f"{metric}_post"] - merged[f"{metric}_pre"]
                non_zero_diffs = diffs[diffs != 0]
                n_pairs = len(diffs)

                t_res = stats.ttest_rel(merged[f"{metric}_post"], merged[f"{metric}_pre"])
                try:
                    w_res = stats.wilcoxon(non_zero_diffs)
                    w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
                except ValueError:
                    w_stat, w_pval = 0.0, 1.0

                pos_count = int((non_zero_diffs > 0).sum())
                neg_count = int((non_zero_diffs < 0).sum())
                n_eff = pos_count + neg_count
                sign_pval = (
                    float(stats.binomtest(pos_count, n=n_eff, p=0.5).pvalue)
                    if n_eff > 0
                    else 1.0
                )

                results.append(
                    HypothesisTestResult(
                        comparison=f"{p_start[:2]} -> {p_end[:2]}",
                        metric=metric,
                        n_pairs=n_pairs,
                        mean_diff=float(diffs.mean()),
                        median_diff=float(diffs.median()),
                        t_stat=float(t_res.statistic),
                        t_pvalue=float(t_res.pvalue),
                        wilcoxon_stat=w_stat,
                        wilcoxon_pvalue=w_pval,
                        sign_positive=pos_count,
                        sign_negative=neg_count,
                        sign_pvalue=sign_pval,
                        reject_h0_alpha_05=bool(
                            (t_res.pvalue < 0.05) or (w_pval < 0.05) or (sign_pval < 0.05)
                        ),
                    )
                )

        return pd.DataFrame([r.__dict__ for r in results])

    @staticmethod
    def run_product_level_hypothesis_tests(monthly_prod: pd.DataFrame) -> pd.DataFrame:
        """Ejecuta pruebas pareadas (P1->P3) y Kruskal-Wallis entre socios para cada uno de los 16 productos."""
        p1_label = "P1: Pre-Consolidación (2013-2014)"
        p3_label = "P3: Madurez Reciente (2024-2025)"

        grouped = (
            monthly_prod.groupby(
                ["bloque_temporal", "tipo_canasta", "producto_clave", "COD_PAI4", "mes"],
                as_index=False,
            )
            .agg(
                expo_real_musd=("expo_real_usd", lambda s: s.mean() / 1e6),
                impo_real_musd=("impo_real_usd", lambda s: s.mean() / 1e6),
                vocacional_musd=("flujo_vocacional_real_usd", lambda s: s.mean() / 1e6),
                scr_bilateral=("scr_bilateral", "mean"),
            )
        )

        rows: List[Dict[str, Any]] = []
        for (canasta, prod), grp in grouped.groupby(["tipo_canasta", "producto_clave"]):
            df_p1 = grp[grp["bloque_temporal"] == p1_label]
            df_p3 = grp[grp["bloque_temporal"] == p3_label]
            merged = pd.merge(
                df_p1,
                df_p3,
                on=["COD_PAI4", "mes"],
                suffixes=("_p1", "_p3"),
            )
            diffs = merged["vocacional_musd_p3"] - merged["vocacional_musd_p1"]
            nz = diffs[diffs != 0]

            t_res = stats.ttest_rel(merged["vocacional_musd_p3"], merged["vocacional_musd_p1"])
            t_stat = float(t_res.statistic) if np.isfinite(t_res.statistic) else 0.0
            t_pval = float(t_res.pvalue) if np.isfinite(t_res.pvalue) else 1.0

            try:
                w_res = stats.wilcoxon(nz)
                w_pval = float(w_res.pvalue)
            except ValueError:
                w_pval = 1.0

            pos_c = int((nz > 0).sum())
            neg_c = int((nz < 0).sum())
            n_eff = pos_c + neg_c
            sign_pval = float(stats.binomtest(pos_c, n=n_eff, p=0.5).pvalue) if n_eff > 0 else 1.0

            # Kruskal-Wallis entre Chile, México y Perú para el producto en todo el periodo
            raw_prod = monthly_prod[monthly_prod["producto_clave"] == prod]
            country_samples = [
                g["flujo_vocacional_real_usd"].values / 1e6
                for _, g in raw_prod.groupby("COD_PAI4")
            ]
            try:
                kw_res = stats.kruskal(*country_samples)
                kw_stat, kw_pval = float(kw_res.statistic), float(kw_res.pvalue)
            except ValueError:
                kw_stat, kw_pval = 0.0, 1.0

            socio_dom = (
                raw_prod.groupby("COD_PAI4")["flujo_vocacional_real_usd"]
                .sum()
                .idxmax()
            )

            rows.append(
                {
                    "tipo_canasta": canasta,
                    "producto_clave": prod,
                    "socio_predominante": socio_dom,
                    "media_p1_musd": float(merged["vocacional_musd_p1"].mean()),
                    "media_p3_musd": float(merged["vocacional_musd_p3"].mean()),
                    "delta_media_p1_p3_musd": float(diffs.mean()),
                    "delta_mediana_p1_p3_musd": float(diffs.median()),
                    "t_stat_p1_p3": t_stat,
                    "t_pvalue": t_pval,
                    "wilcoxon_pvalue": w_pval,
                    "sign_pvalue": sign_pval,
                    "kruskal_pais_h": kw_stat,
                    "kruskal_pais_pvalue": kw_pval,
                    "cambio_significativo_alpha_05": bool(
                        (t_pval < 0.05) or (w_pval < 0.05) or (sign_pval < 0.05)
                    ),
                }
            )

        return pd.DataFrame(rows).sort_values(
            ["tipo_canasta", "media_p3_musd"], ascending=[True, False]
        ).reset_index(drop=True)

    @staticmethod
    def fit_panel_regression(panel: pd.DataFrame) -> Dict[str, Any]:
        """Estima modelo econométrico OLS con efectos fijos y errores estándar robustos HC3."""
        df_reg = panel[panel["exportaciones_reales"] > 0].copy()
        df_reg["ln_x_real"] = np.log(df_reg["exportaciones_reales"])
        df_reg["ln_n_subpartidas"] = np.log(df_reg["n_subpartidas_activas"].clip(lower=1))
        df_reg["ln_m_real"] = np.log(df_reg["importaciones_reales"].clip(lower=1.0))

        dummies = pd.get_dummies(
            df_reg[["COD_PAI4", "tipo_canasta"]],
            drop_first=True,
            dtype=float,
        )
        X_df = pd.concat(
            [
                pd.Series(1.0, index=df_reg.index, name="constante"),
                df_reg[["hhi_invertido", "ln_n_subpartidas", "theta", "ln_m_real"]],
                dummies,
            ],
            axis=1,
        )
        y = df_reg["ln_x_real"].values

        ols_model = sm.OLS(y, X_df).fit(cov_type="HC3")
        coef_table = pd.DataFrame(
            {
                "variable": X_df.columns,
                "coeficiente": ols_model.params.values,
                "error_estandar_hc3": ols_model.bse.values,
                "t_stat_hc3": ols_model.tvalues.values,
                "p_value": ols_model.pvalues.values,
                "ic_inf_95": ols_model.conf_int()[0].values,
                "ic_sup_95": ols_model.conf_int()[1].values,
                "significativo_95": ols_model.pvalues.values < 0.05,
            }
        )
        return {
            "coef_table": coef_table,
            "r2": float(ols_model.rsquared),
            "r2_adj": float(ols_model.rsquared_adj),
            "f_stat": float(ols_model.fvalue),
            "f_pvalue": float(ols_model.f_pvalue),
            "aic": float(ols_model.aic),
            "bic": float(ols_model.bic),
            "n_obs": int(ols_model.nobs),
        }

    @staticmethod
    def fit_causal_dml_and_ml_benchmarks(panel: pd.DataFrame) -> Dict[str, Any]:
        """Ejecuta Double Machine Learning (DML - Chernozhukov et al., 2018) y evalúa modelos ML sin fuga."""
        df_ml = (
            panel[panel["exportaciones_reales"] > 0]
            .copy()
            .sort_values(["COD_PAI4", "tipo_canasta", "fecha"])
            .reset_index(drop=True)
        )
        df_ml["ln_x_real"] = np.log(df_ml["exportaciones_reales"])
        df_ml["ln_n_subpartidas"] = np.log(df_ml["n_subpartidas_activas"].clip(lower=1))
        df_ml["ln_m_real"] = np.log(df_ml["importaciones_reales"].clip(lower=1.0))

        df_ml["theta_bilateral_lag1"] = (
            df_ml.groupby(["COD_PAI4", "tipo_canasta"])["theta_bilateral"]
            .shift(1)
            .fillna(df_ml["theta_bilateral"].median())
        )
        df_ml["grubel_lloyd_lag1"] = (
            df_ml.groupby(["COD_PAI4", "tipo_canasta"])["grubel_lloyd"]
            .shift(1)
            .fillna(df_ml["grubel_lloyd"].median())
        )
        df_ml = df_ml.sort_values("fecha").reset_index(drop=True)

        dummies = pd.get_dummies(
            df_ml[["COD_PAI4", "tipo_canasta", "bloque_temporal"]],
            drop_first=False,
            dtype=float,
        )
        feature_cols = [
            "hhi_invertido",
            "ln_n_subpartidas",
            "theta",
            "theta_bilateral_lag1",
            "grubel_lloyd_lag1",
            "ln_m_real",
            "impo_n_subpartidas",
            "impo_hhi_invertido",
            "mes",
            "anio",
        ]
        X_all = pd.concat([df_ml[feature_cols], dummies], axis=1)
        y = df_ml["ln_x_real"].values

        treatments = [
            ("ln_n_subpartidas", "Elasticidad Causal Margen Extensivo ln(N_it)"),
            ("hhi_invertido", "Efecto Causal Diversificación HHI Invertido (D_it)"),
            ("theta", "Efecto Causal Especialización Relativa (Theta_it)"),
            ("theta_bilateral_lag1", "Efecto Causal Cobertura Bilateral Rezagada (Theta_bil,t-1)"),
            ("ln_m_real", "Efecto Causal Importaciones Reales ln(M_real)"),
        ]
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        dml_rows: List[Dict[str, Any]] = []

        for treat_col, treat_desc in treatments:
            control_cols = [c for c in X_all.columns if c != treat_col]
            W = X_all[control_cols].values
            T = X_all[treat_col].values

            y_tilde = np.zeros_like(y)
            t_tilde = np.zeros_like(T)

            for train_idx, test_idx in kf.split(W):
                model_y = RandomForestRegressor(
                    n_estimators=200, max_depth=8, min_samples_leaf=4, random_state=42, n_jobs=1
                )
                model_t = RandomForestRegressor(
                    n_estimators=200, max_depth=8, min_samples_leaf=4, random_state=42, n_jobs=1
                )
                model_y.fit(W[train_idx], y[train_idx])
                model_t.fit(W[train_idx], T[train_idx])

                y_tilde[test_idx] = y[test_idx] - model_y.predict(W[test_idx])
                t_tilde[test_idx] = T[test_idx] - model_t.predict(W[test_idx])

            theta_hat = float(np.sum(t_tilde * y_tilde) / np.sum(t_tilde**2))
            psi = (y_tilde - theta_hat * t_tilde) * t_tilde
            j_hat = np.mean(t_tilde**2)
            se_dml = float(np.sqrt(np.mean(psi**2) / (j_hat**2) / len(y)))
            z_stat = theta_hat / se_dml
            p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))

            dml_rows.append(
                {
                    "tratamiento": treat_col,
                    "descripcion": treat_desc,
                    "ate_dml_theta": theta_hat,
                    "se_dml_robusto": se_dml,
                    "z_stat": z_stat,
                    "p_value": p_val,
                    "ic_inf_95": theta_hat - 1.96 * se_dml,
                    "ic_sup_95": theta_hat + 1.96 * se_dml,
                    "r2_etapa1_y": float(1.0 - np.var(y_tilde) / np.var(y)),
                    "r2_etapa1_t": float(1.0 - np.var(t_tilde) / np.var(T)),
                    "significativo_95": p_val < 0.05,
                }
            )

        dml_df = pd.DataFrame(dml_rows)

        models = {
            "Ridge Regression (L2 Lineal Con Efectos Fijos)": Ridge(alpha=1.0),
            "Random Forest Regressor (No Paramétrico Bagging)": RandomForestRegressor(
                n_estimators=300, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=1
            ),
            "HistGradientBoosting Regressor (Boosting Árboles)": HistGradientBoostingRegressor(
                max_iter=250, max_depth=6, learning_rate=0.06, random_state=42
            ),
        }

        eval_rows: List[Dict[str, Any]] = []
        tscv = TimeSeriesSplit(n_splits=5)
        X_mat = X_all.values

        for model_name, est in models.items():
            oof_preds = np.zeros_like(y)
            for tr_idx, te_idx in kf.split(X_mat):
                est.fit(X_mat[tr_idx], y[tr_idx])
                oof_preds[te_idx] = est.predict(X_mat[te_idx])

            ts_r2_scores: List[float] = []
            ts_rmse_scores: List[float] = []
            for tr_idx, te_idx in tscv.split(X_mat):
                est.fit(X_mat[tr_idx], y[tr_idx])
                pred_ts = est.predict(X_mat[te_idx])
                ts_r2_scores.append(float(r2_score(y[te_idx], pred_ts)))
                ts_rmse_scores.append(float(np.sqrt(mean_squared_error(y[te_idx], pred_ts))))

            eval_rows.append(
                {
                    "modelo": model_name,
                    "cv_r2_oof": float(r2_score(y, oof_preds)),
                    "cv_rmse_log": float(np.sqrt(mean_squared_error(y, oof_preds))),
                    "cv_mae_log": float(mean_absolute_error(y, oof_preds)),
                    "cv_mape_pct": float(mean_absolute_percentage_error(y, oof_preds) * 100.0),
                    "timeseries_r2_mean": float(np.mean(ts_r2_scores)),
                    "timeseries_rmse_mean": float(np.mean(ts_rmse_scores)),
                }
            )

        ml_metrics_df = pd.DataFrame(eval_rows)

        rf_full = models["Random Forest Regressor (No Paramétrico Bagging)"]
        rf_full.fit(X_mat, y)
        perm = permutation_importance(
            rf_full, X_mat, y, n_repeats=15, random_state=42, n_jobs=1
        )
        importance_df = (
            pd.DataFrame(
                {
                    "feature": X_all.columns,
                    "importance_mean": perm.importances_mean,
                    "importance_std": perm.importances_std,
                }
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )

        return {
            "dml_df": dml_df,
            "ml_metrics_df": ml_metrics_df,
            "importance_df": importance_df,
        }


def generate_ydata_profile_report(panel: pd.DataFrame, root_dir: Path) -> Path:
    """Genera el reporte interactivo de auditoría de datos con `ydata-profiling` en `docs/`."""
    from ydata_profiling import ProfileReport

    report_path = root_dir / "docs" / "eda_ydata_profiling_alianza_pacifico.html"
    cols_profile = [
        "anio",
        "mes",
        "COD_PAI4",
        "tipo_canasta",
        "fob_usd",
        "exportaciones_reales",
        "impo_fob_usd",
        "importaciones_reales",
        "n_subpartidas_activas",
        "impo_n_subpartidas",
        "hhi_invertido",
        "impo_hhi_invertido",
        "scr_bilateral",
        "grubel_lloyd",
        "theta_bilateral",
    ]
    df_prof = panel[[c for c in cols_profile if c in panel.columns]].copy()
    profile = ProfileReport(
        df_prof,
        title="Auditoría Estadística DAMA-BOK — Panel Bilateral Alianza del Pacífico (2013-2025)",
        minimal=True,
        explorative=True,
        progress_bar=False,
    )
    profile.to_file(report_path)
    logger.info("Reporte ydata-profiling exportado en %s", report_path)
    return report_path


def execute_full_crispdm(
    root_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Ejecuta extremo a extremo el flujo CRISP-DM bilateral y desagregado para los 16 productos clave."""
    processed_dir = root_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    expo_cache = processed_dir / "exportaciones_agregadas_2013_2025.csv"
    if expo_cache.exists():
        logger.info("Cargando cubo T1 (EXPO) desde %s", expo_cache)
        df_expo = pd.read_csv(expo_cache, dtype={"POSAR": str, "hs4": str, "COD_PAI4": str})
        df_expo = NumericNormalizer.harmonize_product_labels(df_expo)
    else:
        loader_x = TradeDataLoader(root_dir)
        df_expo = loader_x.load_all_years(filter_alliance_only=False)
        df_expo.to_csv(expo_cache, index=False, encoding="utf-8")

    impo_cache = processed_dir / "importaciones_agregadas_2013_2025.csv"
    df_impo: pd.DataFrame | None = None
    if impo_cache.exists():
        logger.info("Cargando cubo T2 (IMPO) desde %s", impo_cache)
        df_impo = pd.read_csv(impo_cache, dtype={"POSAR": str, "hs4": str, "COD_PAI4": str})
        df_impo = NumericNormalizer.harmonize_product_labels(df_impo)

    # T3: Panel Bilateral Alianza del Pacífico
    panel = FeatureEngineeringPipeline.build_panel_features(df_expo, df_impo)
    panel_path = processed_dir / "panel_alianza_pacifico_2013_2025.csv"
    panel.to_csv(panel_path, index=False, encoding="utf-8")

    # T4: Panel Mensual de 16 Productos (4,032 registros) y Resumen por Bloque
    if df_impo is not None:
        monthly_prod, df_prod = FeatureEngineeringPipeline.build_product_monthly_and_block_tables(
            df_expo, df_impo
        )
        monthly_prod.to_csv(
            processed_dir / "panel_mensual_16_productos_ap.csv", index=False, encoding="utf-8"
        )
        df_prod.to_csv(
            processed_dir / "analisis_desagregado_productos_ap.csv", index=False, encoding="utf-8"
        )
        dist_prod_df = DistributionAnalyzer.fit_product_distributions(monthly_prod)
        dist_prod_df.to_csv(
            processed_dir / "distfit_16_productos_ap.csv", index=False, encoding="utf-8"
        )
        tests_prod_df = EconometricModeler.run_product_level_hypothesis_tests(monthly_prod)
        tests_prod_df.to_csv(
            processed_dir / "pruebas_hipotesis_16_productos_ap.csv", index=False, encoding="utf-8"
        )
    else:
        monthly_prod = pd.DataFrame()
        df_prod = pd.DataFrame()
        dist_prod_df = pd.DataFrame()
        tests_prod_df = pd.DataFrame()

    # T5: Diagnóstico Probabilístico con distfit (Global)
    dist_df = DistributionAnalyzer.fit_all_distributions(panel)
    dist_df.to_csv(processed_dir / "diagnostico_distribuciones_distfit.csv", index=False, encoding="utf-8")

    # T6: Pruebas Paramétricas y No Paramétricas Emparejadas
    paired_tests = EconometricModeler.run_paired_tests(panel)
    paired_tests.to_csv(processed_dir / "resultados_pruebas_hipotesis.csv", index=False, encoding="utf-8")

    # T7: Econometría de Panel (FE-OLS HC3) y Machine Learning Causal (DML)
    reg_results = EconometricModeler.fit_panel_regression(panel)
    reg_results["coef_table"].to_csv(
        processed_dir / "regresion_panel_efectos_fijos.csv", index=False, encoding="utf-8"
    )

    causal_ml = EconometricModeler.fit_causal_dml_and_ml_benchmarks(panel)
    causal_ml["dml_df"].to_csv(
        processed_dir / "modelos_econometricos_y_ml_causal.csv", index=False, encoding="utf-8"
    )
    causal_ml["ml_metrics_df"].to_csv(
        processed_dir / "evaluacion_predictiva_ml.csv", index=False, encoding="utf-8"
    )
    causal_ml["importance_df"].to_csv(
        processed_dir / "importancia_permutacion_ml.csv", index=False, encoding="utf-8"
    )

    reg_results["dml_df"] = causal_ml["dml_df"]
    reg_results["ml_metrics_df"] = causal_ml["ml_metrics_df"]
    reg_results["importance_df"] = causal_ml["importance_df"]
    reg_results["dist_df"] = dist_df
    reg_results["df_prod"] = df_prod
    reg_results["monthly_prod"] = monthly_prod
    reg_results["dist_prod_df"] = dist_prod_df
    reg_results["tests_prod_df"] = tests_prod_df

    return df_expo, panel, paired_tests, reg_results


if __name__ == "__main__":
    workspace = Path(__file__).resolve().parents[2]
    df_x, panel_df, tests_df, models_res = execute_full_crispdm(workspace)
    print("\n=== DISTRIBUCIONES DISTFIT POR CADA UNO DE LOS 16 PRODUCTOS CLAVE ===")
    print(models_res["dist_prod_df"].to_string(index=False))
    print("\n=== PRUEBAS DE HIPÓTESIS PAREADAS Y KRUSKAL-WALLIS POR CADA UNO DE LOS 16 PRODUCTOS ===")
    print(models_res["tests_prod_df"].to_string(index=False))
