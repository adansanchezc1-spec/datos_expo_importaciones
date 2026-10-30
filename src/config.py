"""Módulo centralizado de configuración y carga de variables de entorno (.env).

Aplica los principios SOLID (SRP, DIP) y el patrón GRASP Protected Variations
para eliminar el antipatrón Hard Coding, cargando parámetros desde `.env` mediante
`python-dotenv` con valores por defecto seguros y tipados.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv

# Carga automática del archivo .env desde la raíz del proyecto
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class AppConfig:
    """Objeto de valor inmutable con la configuración transversal del proyecto."""

    project_name: str
    project_version: str
    app_env: str
    log_level: str
    project_root: Path
    data_raw_dir: Path
    data_processed_dir: Path
    docs_output_dir: Path
    notebook_path: Path
    dashboard_html_path: Path
    eda_profile_html_path: Path
    dane_anda_base_url: str
    dane_impo_catalog_historical: int
    dane_impo_catalog_current: int
    dane_staging_dir: Path
    dane_http_timeout_seconds: int
    dane_http_max_retries: int
    dane_http_chunk_size_bytes: int
    dane_max_workers: int
    ap_partner_countries: Tuple[str, ...]
    study_years: Tuple[int, ...]
    random_seed: int
    significance_alpha: float
    fe_ols_cov_type: str
    dml_cv_folds: int
    rf_n_estimators: int
    hgbr_max_iter: int
    permutation_repeats: int
    matplotlib_dpi: int
    plotly_renderer: str
    plotly_template: str

    @classmethod
    def from_env(cls, root_override: Path | None = None) -> "AppConfig":
        """Construye la configuración leyendo variables desde `.env` o el entorno del SO."""
        root = root_override or Path(os.getenv("PROJECT_ROOT", str(_PROJECT_ROOT))).resolve()
        partners_raw = os.getenv("AP_PARTNER_COUNTRIES", "CHL,MEX,PER")
        years_raw = os.getenv("STUDY_YEARS", "2013,2014,2018,2019,2020,2024,2025")

        return cls(
            project_name=os.getenv("PROJECT_NAME", "DatosExpoImportaciones_AlianzaPacifico"),
            project_version=os.getenv("PROJECT_VERSION", "2.0.0"),
            app_env=os.getenv("APP_ENV", "production"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            project_root=root,
            data_raw_dir=root / os.getenv("DATA_RAW_DIR", "data/raw"),
            data_processed_dir=root / os.getenv("DATA_PROCESSED_DIR", "data/processed"),
            docs_output_dir=root / os.getenv("DOCS_OUTPUT_DIR", "docs"),
            notebook_path=root / os.getenv("NOTEBOOK_PATH", "Analisis_exportaciones.ipynb"),
            dashboard_html_path=root
            / os.getenv("DASHBOARD_HTML_PATH", "docs/dashboard_analitico_alianza_pacifico.html"),
            eda_profile_html_path=root
            / os.getenv(
                "EDA_PROFILE_HTML_PATH", "docs/eda_ydata_profiling_alianza_pacifico.html"
            ),
            dane_anda_base_url=os.getenv(
                "DANE_ANDA_BASE_URL", "https://microdatos.dane.gov.co/index.php/catalog"
            ),
            dane_impo_catalog_historical=int(os.getenv("DANE_IMPO_CATALOG_HISTORICAL", "473")),
            dane_impo_catalog_current=int(os.getenv("DANE_IMPO_CATALOG_CURRENT", "856")),
            dane_staging_dir=Path(
                os.getenv("DANE_STAGING_DIR", str(root / "tmp" / "dane_impo_staging"))
            ),
            dane_http_timeout_seconds=int(os.getenv("DANE_HTTP_TIMEOUT_SECONDS", "120")),
            dane_http_max_retries=int(os.getenv("DANE_HTTP_MAX_RETRIES", "5")),
            dane_http_chunk_size_bytes=int(os.getenv("DANE_HTTP_CHUNK_SIZE_BYTES", "1048576")),
            dane_max_workers=int(os.getenv("DANE_MAX_WORKERS", "2")),
            ap_partner_countries=tuple(
                c.strip().upper() for c in partners_raw.split(",") if c.strip()
            ),
            study_years=tuple(int(y.strip()) for y in years_raw.split(",") if y.strip()),
            random_seed=int(os.getenv("RANDOM_SEED", "42")),
            significance_alpha=float(os.getenv("SIGNIFICANCE_ALPHA", "0.05")),
            fe_ols_cov_type=os.getenv("FE_OLS_COV_TYPE", "HC3"),
            dml_cv_folds=int(os.getenv("DML_CV_FOLDS", "5")),
            rf_n_estimators=int(os.getenv("RF_N_ESTIMATORS", "300")),
            hgbr_max_iter=int(os.getenv("HGBR_MAX_ITER", "250")),
            permutation_repeats=int(os.getenv("PERMUTATION_REPEATS", "30")),
            matplotlib_dpi=int(os.getenv("MATPLOTLIB_DPI", "135")),
            plotly_renderer=os.getenv("PLOTLY_RENDERER", "notebook_connected"),
            plotly_template=os.getenv("PLOTLY_TEMPLATE", "plotly_white"),
        )
