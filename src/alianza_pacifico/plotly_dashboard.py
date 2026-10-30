"""Generador del Dashboard Analítico Interactivo con Plotly para la Alianza del Pacífico.

Este módulo implementa los patrones Builder y Factory Method (GoF) bajo los
principios SOLID y Clean Code (PEP 8) para construir:
1. Un Dashboard Analítico Ejecutivo y Estadístico autónomo en HTML5 + Plotly.js
   (`docs/dashboard_analitico_alianza_pacifico.html`) con navegación por pestañas,
   barras de slicers interactivos de desagregación (por Producto, Socio Comercial,
   Bloque Histórico, Frecuencia Temporal y Tipo de Gráfico), tarjetas individuales
   espaciadas para cada uno de los 8 Productos Ofertados y 8 Productos Demandados,
   y cero superposición visual.
2. Figuras maestras interactivas bien espaciadas integrables directamente en
   `Analisis_exportaciones.ipynb`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.alianza_pacifico.crispdm_pipeline import (
    PRODUCTOS_DEMANDADOS_LIST,
    PRODUCTOS_OFERTADOS_LIST,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

COUNTRY_NAMES: Dict[str, str] = {
    "CHL": "Chile",
    "MEX": "México",
    "PER": "Perú",
}

SHORT_BLOCKS: Dict[str, str] = {
    "P1: Pre-Consolidación (2013-2014)": "P1 (2013–14)",
    "P2: Consolidación/COVID (2018-2020)": "P2 (2018–20)",
    "P3: Madurez Reciente (2024-2025)": "P3 (2024–25)",
}


@dataclass(frozen=True)
class DashboardPaths:
    """Rutas de entrada y salida para el Dashboard Analítico."""

    processed_dir: Path
    output_html_path: Path

    @classmethod
    def from_root(cls, root_dir: Path) -> "DashboardPaths":
        """Construye las rutas estándar a partir de la raíz del proyecto."""
        return cls(
            processed_dir=root_dir / "data" / "processed",
            output_html_path=root_dir / "docs" / "dashboard_analitico_alianza_pacifico.html",
        )


class DashboardDataRepository:
    """Repositorio de lectura de las Tablas Objetivo T3–T11 (Patrón Repository / SRP)."""

    def __init__(self, processed_dir: Path) -> None:
        self._dir = processed_dir

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Carga todas las tablas procesadas necesarias para el dashboard."""
        panel = pd.read_csv(self._dir / "panel_alianza_pacifico_2013_2025.csv")
        panel["fecha"] = pd.to_datetime(panel["fecha"])
        panel["periodo_ym"] = panel["fecha"].dt.strftime("%Y-%m")
        panel["bloque_corto"] = panel["bloque_temporal"].map(SHORT_BLOCKS).fillna(panel["bloque_temporal"])
        panel["fob_expo_real_musd"] = panel["exportaciones_reales"] / 1e6
        panel["cif_impo_real_musd"] = panel["importaciones_reales"] / 1e6

        prod_monthly = pd.read_csv(self._dir / "panel_mensual_16_productos_ap.csv")
        prod_monthly["fecha"] = pd.to_datetime(prod_monthly["fecha"])
        prod_monthly["periodo_ym"] = prod_monthly["fecha"].dt.strftime("%Y-%m")
        prod_monthly["bloque_corto"] = (
            prod_monthly["bloque_temporal"].map(SHORT_BLOCKS).fillna(prod_monthly["bloque_temporal"])
        )
        prod_monthly["fob_expo_real_musd"] = prod_monthly["expo_real_usd"] / 1e6
        prod_monthly["cif_impo_real_musd"] = prod_monthly["impo_real_usd"] / 1e6

        prod_block = pd.read_csv(self._dir / "analisis_desagregado_productos_ap.csv")
        prod_block["bloque_corto"] = (
            prod_block["bloque_temporal"].map(SHORT_BLOCKS).fillna(prod_block["bloque_temporal"])
        )
        prod_block["fob_expo_real_musd"] = prod_block["expo_real_usd"] / 1e6
        prod_block["cif_impo_real_musd"] = prod_block["impo_real_usd"] / 1e6

        return {
            "panel": panel,
            "prod_block": prod_block,
            "prod_monthly": prod_monthly,
            "distfit_global": pd.read_csv(self._dir / "diagnostico_distribuciones_distfit.csv"),
            "distfit_16": pd.read_csv(self._dir / "distfit_16_productos_ap.csv"),
            "tests_global": pd.read_csv(self._dir / "resultados_pruebas_hipotesis.csv"),
            "tests_16": pd.read_csv(self._dir / "pruebas_hipotesis_16_productos_ap.csv"),
            "reg_fe": pd.read_csv(self._dir / "regresion_panel_efectos_fijos.csv"),
            "dml_causal": pd.read_csv(self._dir / "modelos_econometricos_y_ml_causal.csv"),
            "ml_eval": pd.read_csv(self._dir / "evaluacion_predictiva_ml.csv"),
            "perm_imp": pd.read_csv(self._dir / "importancia_permutacion_ml.csv"),
        }


class PlotlyChartFactory:
    """Fábrica de gráficos interactivos Plotly desagregados y sin superposición (Factory Method / OCP)."""

    PALETTE: Dict[str, str] = {
        "CHL": "#1b4f72",
        "MEX": "#117864",
        "PER": "#b9770e",
        "Ofertada": "#0e6251",
        "Demandada": "#78281f",
    }

    VARIABLE_LABELS: Dict[str, str] = {
        "hhi_invertido": "Diversificación EXPO (1 - HHI)",
        "ln_n_subpartidas": "Margen Extensivo ln(N EXPO)",
        "theta": "Razón Relacional (θ_t)",
        "theta_bilateral_lag1": "Razón Relacional Rezago (θ_{t-1})",
        "ln_m_real": "Importaciones Reales ln(M_real)",
        "COD_PAI4_MEX": "Efecto Fijo Socio: México",
        "COD_PAI4_PER": "Efecto Fijo Socio: Perú",
        "COD_PAI4_CHL": "Efecto Fijo Socio: Chile",
        "tipo_canasta_Canasta Ofertada": "Canasta Ofertada (Vocacional)",
        "tipo_canasta_Canasta Demandada (Referencia)": "Canasta Demandada (Referencia)",
        "tipo_canasta_Resto de Exportaciones": "Resto de Exportaciones",
        "impo_hhi_invertido": "Diversificación IMPO (1 - HHI)",
        "impo_n_subpartidas": "Subpartidas Activas IMPO (N_M)",
        "anio": "Tendencia Temporal (Año)",
        "mes": "Estacionalidad Mensual (Mes)",
        "grubel_lloyd_lag1": "Grubel-Lloyd Rezago (GL_{t-1})",
    }

    @classmethod
    def _clean_var_label(cls, raw_name: str) -> str:
        """Traduce nombres técnicos de variables a etiquetas cortas y legibles sin solapamiento."""
        if raw_name in cls.VARIABLE_LABELS:
            return cls.VARIABLE_LABELS[raw_name]
        if raw_name.startswith("bloque_temporal_"):
            return raw_name.replace("bloque_temporal_", "").split("(")[0].strip()
        return raw_name[:32]

    @classmethod
    def build_tab1_executive_overview(cls, data: Dict[str, pd.DataFrame]) -> go.Figure:
        """Construye el lienzo ejecutivo bien espaciado con eje categórico sin huecos y controles."""
        panel = data["panel"]
        ts_ofertada = (
            panel[panel["tipo_canasta"] == "Canasta Ofertada"]
            .groupby(["periodo_ym", "anio", "pais_nombre", "COD_PAI4"], as_index=False)
            .agg(
                fob_expo_real_musd=("fob_expo_real_musd", "sum"),
                cif_impo_real_musd=("cif_impo_real_musd", "sum"),
                scr_bilateral=("scr_bilateral", "mean"),
                grubel_lloyd=("grubel_lloyd", "mean"),
                hhi_invertido=("hhi_invertido", "mean"),
                n_subpartidas=("n_subpartidas_activas", "sum"),
            )
            .sort_values("periodo_ym")
        )

        impo_ts = (
            panel.groupby(["periodo_ym", "anio", "pais_nombre", "COD_PAI4"], as_index=False)[
                "cif_impo_real_musd"
            ]
            .sum()
            .sort_values("periodo_ym")
        )

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "1. Exportaciones Reales Canasta Ofertada por Socio (MUSD Base 2013)",
                "2. Importaciones Reales Totales desde la Alianza por Socio (MUSD Base 2013)",
                "3. Saldo Comercial Relativo Bilateral (SCR) vs. Grubel-Lloyd (GL)",
                "4. Diversificación (1 - HHI) vs. Margen Extensivo (N Subpartidas)",
            ),
            vertical_spacing=0.22,
            horizontal_spacing=0.12,
        )

        for code, name in COUNTRY_NAMES.items():
            sub_exp = ts_ofertada[ts_ofertada["COD_PAI4"] == code]
            sub_imp = impo_ts[impo_ts["COD_PAI4"] == code]
            color = cls.PALETTE.get(code, "#333333")

            fig.add_trace(
                go.Scatter(
                    x=sub_exp["periodo_ym"],
                    y=sub_exp["fob_expo_real_musd"],
                    mode="lines+markers",
                    name=f"EXPO Ofertada → {name}",
                    line=dict(color=color, width=2.2),
                    marker=dict(size=4.5),
                    legendgroup=code,
                    hovertemplate=(
                        f"<b>{name} (Canasta Ofertada)</b><br>"
                        "Corte: %{x}<br>EXPO Real: %{y:.2f} MUSD<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=sub_imp["periodo_ym"],
                    y=sub_imp["cif_impo_real_musd"],
                    mode="lines+markers",
                    name=f"IMPO Total ← {name}",
                    line=dict(color=color, width=2.0, dash="dot"),
                    marker=dict(size=4.5, symbol="diamond"),
                    legendgroup=code,
                    hovertemplate=(
                        f"<b>{name} (Importaciones Reales)</b><br>"
                        "Corte: %{x}<br>IMPO Real: %{y:.2f} MUSD<extra></extra>"
                    ),
                ),
                row=1,
                col=2,
            )

            fig.add_trace(
                go.Scatter(
                    x=sub_exp["scr_bilateral"],
                    y=sub_exp["grubel_lloyd"],
                    mode="markers",
                    name=f"SCR vs GL ({name})",
                    marker=dict(
                        color=color,
                        size=np.clip(sub_exp["fob_expo_real_musd"] / 2.5, 6, 20),
                        opacity=0.75,
                        line=dict(width=0.8, color="white"),
                    ),
                    legendgroup=code,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{name}</b><br>SCR Bilateral: %{{x:.3f}}<br>"
                        "Grubel-Lloyd (GL): %{y:.3f}<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=sub_exp["n_subpartidas"],
                    y=sub_exp["hhi_invertido"],
                    mode="markers",
                    name=f"Diversificación vs N ({name})",
                    marker=dict(color=color, size=8, opacity=0.75, symbol="circle"),
                    legendgroup=code,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{name}</b><br>Subpartidas Activas (N): %{{x}}<br>"
                        "Diversificación (1 - HHI): %{y:.3f}<extra></extra>"
                    ),
                ),
                row=2,
                col=2,
            )

        fig.update_xaxes(title_text="Corte Mensual (84 Meses Efectivos)", type="category", nticks=14, row=1, col=1)
        fig.update_xaxes(title_text="Corte Mensual (84 Meses Efectivos)", type="category", nticks=14, row=1, col=2)
        fig.update_xaxes(title_text="Saldo Comercial Relativo (SCR ∈ [-1, 1])", row=2, col=1)
        fig.update_xaxes(title_text="Número de Subpartidas Activas (N_it)", row=2, col=2)

        fig.update_yaxes(title_text="MUSD Reales (Exportaciones)", automargin=True, row=1, col=1)
        fig.update_yaxes(title_text="MUSD Reales (Importaciones)", automargin=True, row=1, col=2)
        fig.update_yaxes(title_text="Índice Grubel-Lloyd (GL ∈ [0, 1])", automargin=True, row=2, col=1)
        fig.update_yaxes(title_text="Diversificación D_it = 1 - HHI", automargin=True, row=2, col=2)

        # Botones de Slicer por Socio Comercial dentro de la figura (para uso en Notebook)
        fig.update_layout(
            template="plotly_white",
            height=940,
            title=dict(
                text="<b>Panorama Ejecutivo Bilateral: Colombia — Alianza del Pacífico (2013–2025)</b>",
                x=0.01,
                xanchor="left",
                y=0.98,
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.09,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.95)",
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.01,
                    xanchor="left",
                    y=1.08,
                    yanchor="top",
                    bgcolor="#eef5f9",
                    bordercolor="#1b4f72",
                    font=dict(size=11.5),
                    buttons=[
                        dict(
                            label="🌎 Todos los Socios (AP)",
                            method="update",
                            args=[{"visible": [True] * 12}],
                        ),
                        dict(
                            label="🇨🇱 Solo Chile",
                            method="update",
                            args=[{"visible": [True] * 4 + [False] * 8}],
                        ),
                        dict(
                            label="🇲🇽 Solo México",
                            method="update",
                            args=[{"visible": [False] * 4 + [True] * 4 + [False] * 4}],
                        ),
                        dict(
                            label="🇵🇪 Solo Perú",
                            method="update",
                            args=[{"visible": [False] * 8 + [True] * 4}],
                        ),
                    ],
                )
            ],
            margin=dict(t=135, b=110, l=75, r=45),
        )
        return fig

    @classmethod
    def build_product_basket_dashboard(
        cls,
        data: Dict[str, pd.DataFrame],
        basket_type: str,
    ) -> go.Figure:
        """Construye la vista general desagregada de los 8 productos con espaciado amplio y etiquetas cortas."""
        prod_block = data["prod_block"]
        prod_monthly = data["prod_monthly"]
        distfit_16 = data["distfit_16"]
        tests_16 = data["tests_16"]

        if basket_type == "Canasta Ofertada":
            product_list = PRODUCTOS_OFERTADOS_LIST
            val_col = "fob_expo_real_musd"
            flow_label = "Exportaciones Reales FOB (MUSD, Base 2013=100)"
            title_prefix = "8 Productos Ofertados (Exportados por Colombia hacia la AP)"
        else:
            product_list = PRODUCTOS_DEMANDADOS_LIST
            val_col = "cif_impo_real_musd"
            flow_label = "Importaciones Reales CIF (MUSD, Base 2013=100)"
            title_prefix = "8 Productos Demandados (Importados por Colombia desde la AP)"

        dist_map = dict(zip(distfit_16["producto_clave"], distfit_16["mejor_ley_distfit"]))
        pval_map = dict(zip(tests_16["producto_clave"], tests_16["wilcoxon_pvalue"]))

        short_titles = [
            f"<b>{idx + 1}. {p[:36]}{'...' if len(p) > 36 else ''}</b> "
            f"<span style='font-size:11px;color:#566573;'>({dist_map.get(p, 'N/A')}, p={pval_map.get(p, 1.0):.2g})</span>"
            for idx, p in enumerate(product_list)
        ]

        fig = make_subplots(
            rows=4,
            cols=2,
            subplot_titles=short_titles,
            vertical_spacing=0.135,
            horizontal_spacing=0.10,
        )

        n_bar_traces = 0
        for idx, prod_name in enumerate(product_list):
            r = (idx // 2) + 1
            c = (idx % 2) + 1
            sub_blk = prod_block[
                (prod_block["tipo_canasta"] == basket_type)
                & (prod_block["producto_clave"] == prod_name)
            ]
            for code, country_label in COUNTRY_NAMES.items():
                sub_c = sub_blk[sub_blk["COD_PAI4"] == code].sort_values("bloque_temporal")
                fig.add_trace(
                    go.Bar(
                        x=sub_c["bloque_corto"],
                        y=sub_c[val_col],
                        name=country_label,
                        marker_color=cls.PALETTE[code],
                        legendgroup=code,
                        showlegend=(idx == 0),
                        visible=True,
                        hovertemplate=(
                            f"<b>{prod_name}</b><br>Socio: {country_label}<br>"
                            "Bloque: %{x}<br>Valor Real Acumulado: %{y:.2f} MUSD<extra></extra>"
                        ),
                    ),
                    row=r,
                    col=c,
                )
                n_bar_traces += 1

        # Serie Anual Limpia (7 años) en lugar de 84 puntos apretados en mini-grilla
        anual_df = (
            prod_monthly[prod_monthly["tipo_canasta"] == basket_type]
            .groupby(["anio", "COD_PAI4", "producto_clave"], as_index=False)[val_col]
            .sum()
            .sort_values("anio")
        )
        n_line_traces = 0
        for idx, prod_name in enumerate(product_list):
            r = (idx // 2) + 1
            c = (idx % 2) + 1
            sub_a = anual_df[anual_df["producto_clave"] == prod_name]
            for code, country_label in COUNTRY_NAMES.items():
                sub_ac = sub_a[sub_a["COD_PAI4"] == code]
                fig.add_trace(
                    go.Scatter(
                        x=sub_ac["anio"].astype(str),
                        y=sub_ac[val_col],
                        mode="lines+markers",
                        name=f"{country_label} (Anual)",
                        line=dict(color=cls.PALETTE[code], width=2.2),
                        marker=dict(size=6),
                        legendgroup=code,
                        showlegend=(idx == 0),
                        visible=False,
                        hovertemplate=(
                            f"<b>{prod_name}</b><br>Socio: {country_label}<br>"
                            "Año: %{x}<br>Flujo Anual: %{y:.2f} MUSD<extra></extra>"
                        ),
                    ),
                    row=r,
                    col=c,
                )
                n_line_traces += 1

        vis_bars = [True] * n_bar_traces + [False] * n_line_traces
        vis_lines = [False] * n_bar_traces + [True] * n_line_traces

        fig.update_xaxes(tickangle=0, tickfont=dict(size=10.5), automargin=True)
        fig.update_yaxes(title_text="MUSD", title_standoff=6, tickfont=dict(size=10), automargin=True)

        fig.update_layout(
            template="plotly_white",
            height=1480,
            barmode="group",
            bargap=0.24,
            bargroupgap=0.10,
            title=dict(
                text=f"<b>Desagregación por Producto: {title_prefix}</b><br><sup>{flow_label}</sup>",
                x=0.01,
                xanchor="left",
                y=0.985,
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.04,
                xanchor="center",
                x=0.5,
                font=dict(size=12),
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.01,
                    xanchor="left",
                    y=1.045,
                    yanchor="top",
                    bgcolor="#eef5f9",
                    bordercolor="#1b4f72",
                    font=dict(size=11.5),
                    buttons=[
                        dict(
                            label="📊 Vista por Bloques (P1, P2, P3)",
                            method="update",
                            args=[{"visible": vis_bars}],
                        ),
                        dict(
                            label="📈 Evolución Anual (2013–2025)",
                            method="update",
                            args=[{"visible": vis_lines}],
                        ),
                    ],
                )
            ],
            margin=dict(t=165, b=95, l=70, r=40),
        )
        return fig

    @classmethod
    def build_product_drilldown_explorer(cls, data: Dict[str, pd.DataFrame]) -> go.Figure:
        """Construye un explorador interactivo individual con dropdown, eje sin huecos y alto espaciado."""
        prod_monthly = data["prod_monthly"]
        distfit_16 = data["distfit_16"]
        tests_16 = data["tests_16"]

        all_products = PRODUCTOS_OFERTADOS_LIST + PRODUCTOS_DEMANDADOS_LIST
        dist_map = dict(zip(distfit_16["producto_clave"], distfit_16["mejor_ley_distfit"]))
        test_map = tests_16.set_index("producto_clave").to_dict(orient="index")

        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.56, 0.44],
            subplot_titles=(
                "1. Evolución Mensual Desagregada por Socio (Chile, México, Perú — 84 Cortes Efectivos)",
                "2. Distribución Empírica Mensual por Socio (Diagrama de Caja Horizontal + Media)",
            ),
            vertical_spacing=0.25,
        )

        traces_per_product = 6
        for p_idx, prod_name in enumerate(all_products):
            sub_p = prod_monthly[prod_monthly["producto_clave"] == prod_name].sort_values("periodo_ym")
            is_ofertada = prod_name in PRODUCTOS_OFERTADOS_LIST
            val_col = "fob_expo_real_musd" if is_ofertada else "cif_impo_real_musd"
            visible_flag = p_idx == 0

            for code, country_label in COUNTRY_NAMES.items():
                sub_c = sub_p[sub_p["COD_PAI4"] == code]
                fig.add_trace(
                    go.Scatter(
                        x=sub_c["periodo_ym"],
                        y=sub_c[val_col],
                        mode="lines+markers",
                        name=f"{country_label} (Serie Mensual)",
                        legendgroup=code,
                        line=dict(color=cls.PALETTE[code], width=2.2),
                        marker=dict(size=5),
                        visible=visible_flag,
                        hovertemplate=(
                            f"<b>{prod_name}</b><br>Socio: {country_label}<br>"
                            "Corte: %{x}<br>Valor Real: %{y:.3f} MUSD<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )

            for code, country_label in COUNTRY_NAMES.items():
                sub_c = sub_p[sub_p["COD_PAI4"] == code]
                fig.add_trace(
                    go.Box(
                        x=sub_c[val_col],
                        y=[country_label] * len(sub_c),
                        orientation="h",
                        name=f"{country_label} (Boxplot)",
                        legendgroup=code,
                        showlegend=False,
                        marker_color=cls.PALETTE[code],
                        boxmean=True,
                        visible=visible_flag,
                        hovertemplate=(
                            f"<b>{prod_name} ({country_label})</b><br>"
                            "Valor Mensual: %{x:.3f} MUSD<extra></extra>"
                        ),
                    ),
                    row=2,
                    col=1,
                )

        buttons = []
        total_traces = len(all_products) * traces_per_product
        for p_idx, prod_name in enumerate(all_products):
            vis = [False] * total_traces
            start = p_idx * traces_per_product
            for k in range(start, start + traces_per_product):
                vis[k] = True

            is_ofertada = prod_name in PRODUCTOS_OFERTADOS_LIST
            basket_tag = "OFERTADO (EXPO)" if is_ofertada else "DEMANDADO (IMPO)"
            t_info = test_map.get(prod_name, {})
            d_law = dist_map.get(prod_name, "N/A")
            p_wilcox = t_info.get("wilcoxon_pvalue", 1.0)
            delta_m = t_info.get("delta_media_p1_p3_musd", 0.0)

            buttons.append(
                dict(
                    label=f"[{basket_tag[:4]}] {prod_name[:52]}",
                    method="update",
                    args=[
                        {"visible": vis},
                        {
                            "title.text": (
                                f"<b>Explorador Individual: {prod_name[:68]}</b><br>"
                                f"<sup>Canasta: {basket_tag} | Ajuste distfit: {d_law} | "
                                f"Δ Media (P3-P1): {delta_m:+.2f} MUSD/mes | "
                                f"Wilcoxon p-valor: {p_wilcox:.4g}</sup>"
                            )
                        },
                    ],
                )
            )

        first_prod = all_products[0]
        first_info = test_map.get(first_prod, {})
        fig.update_xaxes(
            title_text="Corte Mensual Efectivo (2013–2025)",
            type="category",
            nticks=16,
            automargin=True,
            row=1,
            col=1,
        )
        fig.update_yaxes(title_text="Millones USD Reales (Base 2013=100)", automargin=True, row=1, col=1)
        fig.update_xaxes(title_text="Millones USD Reales Mensuales (Escala Nivel)", automargin=True, row=2, col=1)
        fig.update_yaxes(title_text="Socio Comercial", automargin=True, row=2, col=1)

        fig.update_layout(
            template="plotly_white",
            height=900,
            title=dict(
                text=(
                    f"<b>Explorador Individual: {first_prod[:68]}</b><br>"
                    f"<sup>Canasta: OFERTADO (EXPO) | Ajuste distfit: {dist_map.get(first_prod)} | "
                    f"Δ Media (P3-P1): {first_info.get('delta_media_p1_p3_musd', 0.0):+.2f} MUSD/mes | "
                    f"Wilcoxon p-valor: {first_info.get('wilcoxon_pvalue', 1.0):.4g}</sup>"
                ),
                x=0.01,
                xanchor="left",
                y=0.98,
                yanchor="top",
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.10,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.95)",
            ),
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.08,
                    yanchor="top",
                    bgcolor="#eef5f9",
                    bordercolor="#1b4f72",
                    font=dict(size=11.5),
                )
            ],
            margin=dict(t=165, b=105, l=90, r=45),
        )
        return fig

    @classmethod
    def build_tab4_statistical_inference_and_distfit(
        cls, data: Dict[str, pd.DataFrame]
    ) -> go.Figure:
        """Construye el panel de diagnóstico estadístico en disposición 2x1 a ancho completo."""
        tests_16 = data["tests_16"].copy()
        distfit_16 = data["distfit_16"].copy()
        prod_block = data["prod_block"].copy()

        merged = pd.merge(
            tests_16,
            distfit_16[["producto_clave", "mejor_ley_distfit", "asimetria_skewness"]],
            on="producto_clave",
            how="left",
        ).sort_values(["tipo_canasta", "media_p3_musd"], ascending=[True, True])

        short_names = [
            f"[{'OF' if tc == 'Canasta Ofertada' else 'DEM'}] "
            + (p[:32] + "..." if len(p) > 32 else p)
            + f" [{law} | p={pv:.2g}]"
            for tc, p, law, pv in zip(
                merged["tipo_canasta"],
                merged["producto_clave"],
                merged["mejor_ley_distfit"],
                merged["wilcoxon_pvalue"],
            )
        ]

        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.50, 0.50],
            subplot_titles=(
                "1. Cambio Estructural de Medias Mensuales: P1 (2013–2014) vs. P3 (2024–2025) [Ley distfit y p-valor Wilcoxon en eje Y]",
                "2. Mapa de Calor Bilateral: Saldo Comercial Relativo (SCR ∈ [-1, +1]) por Producto Clave, Socio y Bloque Temporal",
            ),
            vertical_spacing=0.14,
        )

        max_bar_val = float(
            max(merged["media_p1_musd"].max(), merged["media_p3_musd"].max(), 1.0)
        )

        fig.add_trace(
            go.Bar(
                y=short_names,
                x=merged["media_p1_musd"],
                name="Media Mensual P1 (2013–2014)",
                orientation="h",
                marker_color="#2874a6",
                hovertemplate="<b>%{y}</b><br>Media P1: %{x:.2f} MUSD/mes<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                y=short_names,
                x=merged["media_p3_musd"],
                name="Media Mensual P3 (2024–2025)",
                orientation="h",
                marker_color="#117a65",
                hovertemplate="<b>%{y}</b><br>Media P3: %{x:.2f} MUSD/mes<extra></extra>",
            ),
            row=1,
            col=1,
        )

        prod_block["col_key"] = (
            prod_block["COD_PAI4"] + " — " + prod_block["bloque_temporal"].str.slice(0, 2)
        )
        ordered_products = PRODUCTOS_OFERTADOS_LIST + PRODUCTOS_DEMANDADOS_LIST
        pivot_scr = (
            prod_block.pivot_table(
                index="producto_clave",
                columns="col_key",
                values="scr_bilateral",
                aggfunc="mean",
            )
            .reindex(ordered_products)
            .fillna(0.0)
        )
        pivot_gl = (
            prod_block.pivot_table(
                index="producto_clave",
                columns="col_key",
                values="grubel_lloyd",
                aggfunc="mean",
            )
            .reindex(ordered_products)
            .fillna(0.0)
        )

        y_heat = [
            f"[{'OF' if p in PRODUCTOS_OFERTADOS_LIST else 'DEM'}] "
            + (p[:40] + "..." if len(p) > 40 else p)
            for p in pivot_scr.index
        ]
        fig.add_trace(
            go.Heatmap(
                z=pivot_scr.values,
                x=list(pivot_scr.columns),
                y=y_heat,
                colorscale="RdBu",
                zmid=0.0,
                zmin=-1.0,
                zmax=1.0,
                text=np.round(pivot_scr.values, 2),
                texttemplate="%{text:.2f}",
                textfont=dict(size=10),
                colorbar=dict(
                    title="SCR",
                    len=0.42,
                    y=0.21,
                    yanchor="middle",
                    x=1.02,
                ),
                customdata=pivot_gl.values,
                hovertemplate=(
                    "<b>%{y}</b><br>Socio-Periodo: %{x}<br>"
                    "SCR Bilateral: %{z:.3f}<br>Grubel-Lloyd (GL): %{customdata:.3f}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

        fig.update_xaxes(
            title_text="Flujo Mensual Promedio Real (Millones USD, Base 2013=100)",
            range=[0, max_bar_val * 1.12],
            automargin=True,
            row=1,
            col=1,
        )
        fig.update_yaxes(automargin=True, tickfont=dict(size=10.5), row=1, col=1)
        fig.update_xaxes(
            title_text="Socio Comercial y Bloque Temporal (P1: 2013-14 | P2: 2018-20 | P3: 2024-25)",
            automargin=True,
            row=2,
            col=1,
        )
        fig.update_yaxes(
            autorange="reversed",
            automargin=True,
            tickfont=dict(size=10.5),
            row=2,
            col=1,
        )

        fig.update_layout(
            template="plotly_white",
            height=1360,
            barmode="group",
            bargap=0.22,
            bargroupgap=0.08,
            title=dict(
                text=(
                    "<b>Inferencia Estadística No Paramétrica/Paramétrica y Especialización "
                    "Intraindustrial (16 Productos Clave)</b>"
                ),
                x=0.01,
                xanchor="left",
                y=0.98,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.03,
                xanchor="right",
                x=1.0,
            ),
            margin=dict(t=125, b=75, l=345, r=80),
        )
        return fig

    @classmethod
    def build_tab5_econometrics_and_causal_ml(cls, data: Dict[str, pd.DataFrame]) -> go.Figure:
        """Construye el lienzo de Econometría FE-OLS, DML Causal y Evaluación ML en disposición 4x1."""
        reg_fe = data["reg_fe"]
        dml_df = data["dml_causal"].copy()
        ml_eval = data["ml_eval"]
        perm_imp = data["perm_imp"].head(8).sort_values("importance_mean", ascending=True).copy()

        fe_rows = reg_fe[reg_fe["variable"] != "constante"].copy()
        fe_rows["label_clean"] = [
            f"{cls._clean_var_label(v)} (β={c:+.2f}, p={p:.2g})"
            for v, c, p in zip(fe_rows["variable"], fe_rows["coeficiente"], fe_rows["p_value"])
        ]
        dml_df["label_clean"] = [
            f"{cls._clean_var_label(v)} (θ={c:+.2f}, p={p:.2g})"
            for v, c, p in zip(
                dml_df["tratamiento"], dml_df["ate_dml_theta"], dml_df["p_value"]
            )
        ]
        perm_imp["label_clean"] = [
            f"{cls._clean_var_label(v)} (ΔR²={m:.3f})"
            for v, m in zip(perm_imp["feature"], perm_imp["importance_mean"])
        ]

        fig = make_subplots(
            rows=4,
            cols=1,
            row_heights=[0.26, 0.22, 0.24, 0.28],
            subplot_titles=(
                "1. Elasticidades Econométricas FE-OLS (Intervalo de Confianza 95% Robusto HC3) sobre ln(X_real)",
                "2. Efectos Causales Ortogonalizados — Double Machine Learning (DML — Robinson 1988, IC 95%)",
                "3. Desempeño Predictivo Out-of-Fold (5-Fold CV vs. Walk-Forward TimeSeriesSplit): R² y RMSE",
                "4. Importancia Predictiva por Permutación Fuera de Muestra (Caída Media en R² ± DE)",
            ),
            vertical_spacing=0.10,
        )

        fig.add_trace(
            go.Scatter(
                x=fe_rows["coeficiente"],
                y=fe_rows["label_clean"],
                mode="markers",
                name="FE-OLS (IC 95% HC3)",
                marker=dict(color="#1b4f72", size=10),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=fe_rows["ic_sup_95"] - fe_rows["coeficiente"],
                    arrayminus=fe_rows["coeficiente"] - fe_rows["ic_inf_95"],
                    color="#1b4f72",
                    thickness=2.2,
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>Coeficiente β: %{x:.4f}<br>"
                    "p-valor: %{customdata:.4g}<extra></extra>"
                ),
                customdata=fe_rows["p_value"],
            ),
            row=1,
            col=1,
        )
        fig.add_vline(x=0.0, line_dash="dash", line_color="firebrick", row=1, col=1)

        fig.add_trace(
            go.Bar(
                y=dml_df["label_clean"],
                x=dml_df["ate_dml_theta"],
                orientation="h",
                name="Efecto Causal DML (θ ± IC 95%)",
                marker_color=[
                    "#117864" if sig else "#7f8c8d" for sig in dml_df["significativo_95"]
                ],
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=dml_df["ic_sup_95"] - dml_df["ate_dml_theta"],
                    arrayminus=dml_df["ate_dml_theta"] - dml_df["ic_inf_95"],
                    color="#1b2631",
                    thickness=1.8,
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>Efecto Causal DML: %{x:.4f}<br>"
                    "p-valor: %{customdata:.4g}<extra></extra>"
                ),
                customdata=dml_df["p_value"],
            ),
            row=2,
            col=1,
        )
        fig.add_vline(x=0.0, line_dash="dot", line_color="firebrick", row=2, col=1)

        short_ml = [
            "Ridge FE (Lineal Panel)",
            "Random Forest (No Paramétrico)",
            "HistGradientBoosting (Mejor Modelo)",
        ]
        fig.add_trace(
            go.Bar(
                x=short_ml,
                y=ml_eval["cv_r2_oof"],
                name="R² 5-Fold Out-of-Fold",
                marker_color="#1f618d",
                text=[f"R²={v:.3f}" for v in ml_eval["cv_r2_oof"]],
                textposition="outside",
                cliponaxis=False,
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=short_ml,
                y=ml_eval["timeseries_r2_mean"],
                name="R² Walk-Forward Temporal",
                marker_color="#117a65",
                text=[f"TS R²={v:.3f}" for v in ml_eval["timeseries_r2_mean"]],
                textposition="outside",
                cliponaxis=False,
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=short_ml,
                y=ml_eval["cv_rmse_log"],
                name="RMSE (Escala Logarítmica)",
                marker_color="#d68910",
                text=[f"RMSE={v:.3f}" for v in ml_eval["cv_rmse_log"]],
                textposition="outside",
                cliponaxis=False,
            ),
            row=3,
            col=1,
        )

        max_perm = float(
            (perm_imp["importance_mean"] + perm_imp["importance_std"]).max()
        )
        fig.add_trace(
            go.Bar(
                y=perm_imp["label_clean"],
                x=perm_imp["importance_mean"],
                orientation="h",
                name="Importancia Permutación (Δ R²)",
                marker_color="#117a65",
                error_x=dict(type="data", array=perm_imp["importance_std"], color="#1b2631"),
                hovertemplate="<b>%{y}</b><br>Caída media en R²: %{x:.4f}<extra></extra>",
            ),
            row=4,
            col=1,
        )

        fig.update_xaxes(title_text="Coeficiente β (IC 95% Robusto HC3)", automargin=True, row=1, col=1)
        fig.update_yaxes(automargin=True, tickfont=dict(size=10.5), row=1, col=1)
        fig.update_xaxes(title_text="Efecto Causal Directo θ (IC 95% DML)", automargin=True, row=2, col=1)
        fig.update_yaxes(automargin=True, tickfont=dict(size=10.5), row=2, col=1)
        fig.update_xaxes(title_text="Arquitectura del Modelo Evaluado", automargin=True, row=3, col=1)
        fig.update_yaxes(
            title_text="Valor de la Métrica (R² / RMSE)",
            range=[0, 1.18],
            automargin=True,
            row=3,
            col=1,
        )
        fig.update_xaxes(
            title_text="Reducción en R² Fuera de Muestra al Permutar Variable (Δ R²)",
            range=[0, max_perm * 1.12],
            automargin=True,
            row=4,
            col=1,
        )
        fig.update_yaxes(automargin=True, tickfont=dict(size=10.5), row=4, col=1)

        fig.update_layout(
            template="plotly_white",
            height=1460,
            barmode="group",
            title=dict(
                text=(
                    "<b>Modelado Econométrico de Efectos Fijos (FE-OLS HC3) y "
                    "Machine Learning Causal (Double Machine Learning)</b>"
                ),
                x=0.01,
                xanchor="left",
                y=0.985,
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.06,
                xanchor="center",
                x=0.5,
            ),
            margin=dict(t=110, b=95, l=330, r=55),
        )
        return fig


class AnalyticalDashboardBuilder:
    """Constructor del archivo HTML autónomo con Slicers Interactivos y Tarjetas Espaciadas."""

    def __init__(self, paths: DashboardPaths) -> None:
        self._paths = paths
        self._repo = DashboardDataRepository(paths.processed_dir)

    def _prepare_js_payload(self, data: Dict[str, pd.DataFrame]) -> str:
        """Serializa los cubos procesados en JSON compacto para los Slicers interactivos del navegador."""
        panel = data["panel"].copy()
        prod_monthly = data["prod_monthly"].copy()
        prod_block = data["prod_block"].copy()
        distfit_16 = data["distfit_16"].copy()
        tests_16 = data["tests_16"].copy()

        panel_records = (
            panel[
                [
                    "anio",
                    "mes",
                    "periodo_ym",
                    "bloque_corto",
                    "COD_PAI4",
                    "pais_nombre",
                    "tipo_canasta",
                    "fob_expo_real_musd",
                    "cif_impo_real_musd",
                    "scr_bilateral",
                    "grubel_lloyd",
                    "hhi_invertido",
                    "n_subpartidas_activas",
                ]
            ]
            .round(4)
            .to_dict(orient="records")
        )

        pm_records = (
            prod_monthly[
                [
                    "anio",
                    "mes",
                    "periodo_ym",
                    "bloque_corto",
                    "COD_PAI4",
                    "pais_nombre",
                    "tipo_canasta",
                    "producto_clave",
                    "hs4",
                    "fob_expo_real_musd",
                    "cif_impo_real_musd",
                    "scr_bilateral",
                    "grubel_lloyd",
                ]
            ]
            .round(4)
            .to_dict(orient="records")
        )

        pb_records = (
            prod_block[
                [
                    "bloque_corto",
                    "COD_PAI4",
                    "pais_nombre",
                    "tipo_canasta",
                    "producto_clave",
                    "hs4",
                    "fob_expo_real_musd",
                    "cif_impo_real_musd",
                    "scr_bilateral",
                    "grubel_lloyd",
                ]
            ]
            .round(4)
            .to_dict(orient="records")
        )

        meta_map: Dict[str, Any] = {}
        dist_dict = distfit_16.set_index("producto_clave").to_dict(orient="index")
        test_dict = tests_16.set_index("producto_clave").to_dict(orient="index")
        hs_map = (
            prod_block[["producto_clave", "hs4"]]
            .drop_duplicates()
            .set_index("producto_clave")["hs4"]
            .to_dict()
        )

        for p_name in PRODUCTOS_OFERTADOS_LIST + PRODUCTOS_DEMANDADOS_LIST:
            d_info = dist_dict.get(p_name, {})
            t_info = test_dict.get(p_name, {})
            is_of = p_name in PRODUCTOS_OFERTADOS_LIST
            sub_pb = prod_block[prod_block["producto_clave"] == p_name]
            total_val = (
                float(sub_pb["fob_expo_real_musd"].sum())
                if is_of
                else float(sub_pb["cif_impo_real_musd"].sum())
            )
            meta_map[p_name] = {
                "canasta": "Canasta Ofertada (Exportaciones)" if is_of else "Canasta Demandada (Importaciones)",
                "is_ofertada": is_of,
                "hs4": str(hs_map.get(p_name, "")),
                "total_musd": round(total_val, 2),
                "distfit": str(d_info.get("mejor_ley_distfit", "N/A")),
                "skew": round(float(d_info.get("asimetria_skewness", 0.0)), 2),
                "media_p1": round(float(t_info.get("media_p1_musd", 0.0)), 2),
                "media_p3": round(float(t_info.get("media_p3_musd", 0.0)), 2),
                "delta_p1_p3": round(float(t_info.get("delta_media_p1_p3_musd", 0.0)), 2),
                "t_stat": round(float(t_info.get("t_stat_p1_p3", 0.0)), 3),
                "t_pval": float(t_info.get("t_pvalue", 1.0)),
                "wilcoxon_pval": float(t_info.get("wilcoxon_pvalue", 1.0)),
                "kruskal_h": round(float(t_info.get("kruskal_pais_h", 0.0)), 2),
                "kruskal_pval": float(t_info.get("kruskal_pais_pvalue", 1.0)),
                "socio_pred": str(t_info.get("socio_predominante", "")),
                "sig": bool(t_info.get("cambio_significativo_alpha_05", False)),
            }

        payload = {
            "ofertados_list": PRODUCTOS_OFERTADOS_LIST,
            "demandados_list": PRODUCTOS_DEMANDADOS_LIST,
            "panel": panel_records,
            "prod_monthly": pm_records,
            "prod_block": pb_records,
            "meta": meta_map,
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _build_product_cards_html(product_list: List[str], prefix: str, meta_json_raw: Dict[str, Any]) -> str:
        """Construye las 8 tarjetas HTML individuales y espaciadas para una canasta de productos."""
        cards_html: List[str] = []
        for idx, p_name in enumerate(product_list):
            m = meta_json_raw[p_name]
            flow_tag = "Total Exportado Real" if m["is_ofertada"] else "Total Importado Real"
            delta_sign = "+" if m["delta_p1_p3"] >= 0 else ""
            badge_bg = "#0e6251" if m["sig"] else "#5d6d7e"
            sig_label = "Cambio P1→P3 Significativo" if m["sig"] else "Flujo P1→P3 Estable"
            cards_html.append(
                f"""
                <div class="product-card" id="{prefix}_card_{idx}" data-prod-index="{idx}">
                    <div class="product-card-header">
                        <div class="product-card-title-row">
                            <span class="product-index-badge">#{idx + 1}</span>
                            <h4 class="product-card-title">{p_name}</h4>
                        </div>
                        <div class="product-badges-row">
                            <span class="meta-pill navy">HS4: {m['hs4']}</span>
                            <span class="meta-pill teal">{flow_tag}: <b>${m['total_musd']:,.2f} MUSD</b></span>
                            <span class="meta-pill slate">distfit: <b>{m['distfit']}</b></span>
                            <span class="meta-pill amber">Δ Media P1→P3: <b>{delta_sign}{m['delta_p1_p3']:.2f} MUSD/mes</b></span>
                            <span class="meta-pill" style="background:{badge_bg};color:#fff;">Wilcoxon p={m['wilcoxon_pval']:.3g} ({sig_label})</span>
                        </div>
                    </div>
                    <div id="{prefix}_plot_{idx}" class="product-plot-canvas"></div>
                </div>
                """
            )
        return "\n".join(cards_html)

    def build_standalone_html(self) -> Path:
        """Genera el archivo HTML interactivo completo con Slicers de desagregación y cero superposición."""
        data = self._repo.load_all()
        js_payload_str = self._prepare_js_payload(data)
        js_payload_dict = json.loads(js_payload_str)

        fig_stats = PlotlyChartFactory.build_tab4_statistical_inference_and_distfit(data)
        fig_models = PlotlyChartFactory.build_tab5_econometrics_and_causal_ml(data)

        panel = data["panel"]
        total_expo_ofertada = panel.loc[
            panel["tipo_canasta"] == "Canasta Ofertada", "fob_expo_real_musd"
        ].sum()
        total_impo_ap = panel["cif_impo_real_musd"].sum()

        html_stats = fig_stats.to_html(
            full_html=False, include_plotlyjs=False, div_id="plot_stats"
        )
        html_models = fig_models.to_html(
            full_html=False, include_plotlyjs=False, div_id="plot_models"
        )

        ofertados_cards_html = self._build_product_cards_html(
            PRODUCTOS_OFERTADOS_LIST, "of", js_payload_dict["meta"]
        )
        demandados_cards_html = self._build_product_cards_html(
            PRODUCTOS_DEMANDADOS_LIST, "dem", js_payload_dict["meta"]
        )

        # Opciones para el selector del Explorador de Productos (Pestaña 1)
        opt_ofertados = "\n".join(
            f'<option value="{p}">[OFERTADO #{i+1}] {p}</option>'
            for i, p in enumerate(PRODUCTOS_OFERTADOS_LIST)
        )
        opt_demandados = "\n".join(
            f'<option value="{p}">[DEMANDADO #{i+1}] {p}</option>'
            for i, p in enumerate(PRODUCTOS_DEMANDADOS_LIST)
        )

        # Opciones para el Slicer de Producto en Pestaña 2 y Pestaña 3
        slicer_opts_of = '<option value="ALL">🔍 Ver los 8 Productos Ofertados (Grilla Completa)</option>\n' + "\n".join(
            f'<option value="{i}">#{i+1} — {p}</option>'
            for i, p in enumerate(PRODUCTOS_OFERTADOS_LIST)
        )
        slicer_opts_dem = '<option value="ALL">🔍 Ver los 8 Productos Demandados (Grilla Completa)</option>\n' + "\n".join(
            f'<option value="{i}">#{i+1} — {p}</option>'
            for i, p in enumerate(PRODUCTOS_DEMANDADOS_LIST)
        )

        tests_16_records = data["tests_16"].to_dict(orient="records")
        table_rows_html = []
        for row in tests_16_records:
            badge_color = "#117a65" if row["cambio_significativo_alpha_05"] else "#7f8c8d"
            badge_txt = "SÍ (p < 0.05)" if row["cambio_significativo_alpha_05"] else "Estable"
            table_rows_html.append(
                f"<tr>"
                f"<td><b>{row['tipo_canasta']}</b></td>"
                f"<td>{row['producto_clave']}</td>"
                f"<td style='text-align:center;'><code>{row['socio_predominante']}</code></td>"
                f"<td style='text-align:right;'>{row['media_p1_musd']:.2f}</td>"
                f"<td style='text-align:right;'>{row['media_p3_musd']:.2f}</td>"
                f"<td style='text-align:right; font-weight:600;'>{row['delta_media_p1_p3_musd']:+.2f}</td>"
                f"<td style='text-align:right;'>{row['t_stat_p1_p3']:.3f} ({row['t_pvalue']:.3g})</td>"
                f"<td style='text-align:right;'>{row['wilcoxon_pvalue']:.3g}</td>"
                f"<td style='text-align:right;'>{row['kruskal_pais_h']:.1f} ({row['kruskal_pais_pvalue']:.2g})</td>"
                f"<td style='text-align:center;'><span style='background:{badge_color};color:white;padding:3px 8px;border-radius:10px;font-size:11px;'>{badge_txt}</span></td>"
                f"</tr>"
            )
        rows_joined = "\n".join(table_rows_html)

        full_html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard Analítico — Alianza del Pacífico (2013–2025)</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
    <style>
        :root {{
            --primary: #1b4f72;
            --secondary: #117864;
            --accent: #b9770e;
            --bg: #f0f4f8;
            --card: #ffffff;
            --text: #1c2833;
            --border: #d6e0ea;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.45;
        }}
        header {{
            background: linear-gradient(135deg, #1b4f72 0%, #117864 100%);
            color: white;
            padding: 24px 38px;
            box-shadow: 0 3px 14px rgba(0,0,0,0.16);
        }}
        header h1 {{
            margin: 0 0 6px 0;
            font-size: 24px;
            font-weight: 700;
        }}
        header p {{
            margin: 0;
            font-size: 14px;
            opacity: 0.94;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            padding: 20px 38px 10px 38px;
        }}
        .kpi-card {{
            background: var(--card);
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            border-left: 5px solid var(--primary);
        }}
        .kpi-card.green {{ border-left-color: var(--secondary); }}
        .kpi-card.amber {{ border-left-color: var(--accent); }}
        .kpi-title {{
            font-size: 11.5px;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            color: #566573;
            margin-bottom: 5px;
            font-weight: 600;
        }}
        .kpi-value {{
            font-size: 22px;
            font-weight: 700;
            color: var(--text);
        }}
        .kpi-sub {{
            font-size: 11.5px;
            color: #6c7a89;
            margin-top: 4px;
        }}
        .tabs-bar {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            padding: 16px 38px 0 38px;
            border-bottom: 2px solid #cfd9e2;
            background: #e9eff5;
            margin-top: 10px;
        }}
        .tab-btn {{
            background: #dce6ef;
            border: 1px solid transparent;
            padding: 12px 20px;
            border-radius: 10px 10px 0 0;
            font-size: 13.5px;
            font-weight: 600;
            color: #2c3e50;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .tab-btn:hover {{
            background: #c9ddef;
        }}
        .tab-btn.active {{
            background: var(--primary);
            color: white;
            box-shadow: 0 -2px 8px rgba(27,79,114,0.18);
        }}
        .tab-content {{
            display: none;
            padding: 28px 38px 54px 38px;
        }}
        .tab-content.active {{
            display: block;
        }}
        /* Bloques de Sección bien separados */
        .section-block {{
            background: var(--card);
            border-radius: 14px;
            padding: 26px 28px;
            margin-bottom: 38px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.06);
            border: 1px solid var(--border);
        }}
        .section-header {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 14px;
            margin-bottom: 18px;
            padding-bottom: 14px;
            border-bottom: 2px solid #eef2f7;
        }}
        .section-title {{
            margin: 0;
            font-size: 19px;
            font-weight: 700;
            color: var(--primary);
        }}
        .section-subtitle {{
            margin: 4px 0 0 0;
            font-size: 13px;
            color: #5d6d7e;
        }}
        /* Barra de Slicers de Desagregación */
        .slicer-bar {{
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            align-items: flex-end;
            background: #f7fafd;
            border: 1px solid #d0dfed;
            border-radius: 11px;
            padding: 14px 18px;
            margin-bottom: 24px;
        }}
        .slicer-group {{
            display: flex;
            flex-direction: column;
            gap: 5px;
            min-width: 195px;
            flex: 1;
        }}
        .slicer-group.wide {{
            min-width: 320px;
            flex: 2;
        }}
        .slicer-label {{
            font-size: 11.5px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            color: #1b4f72;
        }}
        .slicer-select {{
            padding: 9px 12px;
            border-radius: 8px;
            border: 1px solid #b5c7d9;
            background: #ffffff;
            font-size: 13px;
            font-weight: 600;
            color: #1c2833;
            cursor: pointer;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }}
        .slicer-select:focus {{
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(27,79,114,0.15);
        }}
        /* Grillas de gráficos espaciadas */
        .exec-charts-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 30px;
        }}
        .explorer-charts-grid {{
            display: grid;
            grid-template-columns: 1.25fr 1fr;
            gap: 26px;
            align-items: stretch;
        }}
        @media (max-width: 1150px) {{
            .explorer-charts-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        .sub-chart-box {{
            background: #ffffff;
            border: 1px solid #e4ebf2;
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 1px 5px rgba(0,0,0,0.03);
        }}
        /* Tarjetas de los 8 productos Ofertados y Demandados */
        .product-cards-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 28px;
            margin-top: 26px;
        }}
        .product-cards-grid.single-col {{
            grid-template-columns: 1fr;
        }}
        @media (max-width: 1200px) {{
            .product-cards-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        .product-card {{
            background: #ffffff;
            border: 1px solid #d8e2ec;
            border-radius: 13px;
            padding: 20px 22px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.05);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        .product-card:hover {{
            box-shadow: 0 6px 18px rgba(27,79,114,0.11);
        }}
        .product-card-header {{
            margin-bottom: 12px;
            padding-bottom: 12px;
            border-bottom: 1px solid #edf2f7;
        }}
        .product-card-title-row {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }}
        .product-index-badge {{
            background: var(--primary);
            color: #ffffff;
            font-weight: 700;
            font-size: 12px;
            padding: 4px 9px;
            border-radius: 7px;
            flex-shrink: 0;
        }}
        .product-card-title {{
            margin: 0;
            font-size: 15.5px;
            font-weight: 700;
            color: #1c2833;
            line-height: 1.3;
        }}
        .product-badges-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 7px;
        }}
        .meta-pill {{
            font-size: 11.5px;
            padding: 4px 10px;
            border-radius: 20px;
            background: #eef3f8;
            color: #2c3e50;
            white-space: nowrap;
        }}
        .meta-pill.navy {{ background: #e8f1f8; color: #1b4f72; font-weight: 600; }}
        .meta-pill.teal {{ background: #e6f5f2; color: #0e6251; }}
        .meta-pill.amber {{ background: #fef5e7; color: #9c640c; }}
        .meta-pill.slate {{ background: #ebedef; color: #34495e; }}
        .product-plot-canvas {{
            width: 100%;
            height: 360px;
        }}
        .product-cards-grid.single-col .product-plot-canvas {{
            height: 460px;
        }}
        /* Banda de resumen estadístico en el Explorador */
        .explorer-meta-strip {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 20px;
            padding: 12px 16px;
            background: #eef5fb;
            border-left: 4px solid var(--primary);
            border-radius: 8px;
        }}
        table.stat-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12.5px;
            background: white;
            border-radius: 8px;
            overflow: hidden;
        }}
        table.stat-table th {{
            background: #1b4f72;
            color: white;
            padding: 11px 12px;
            text-align: left;
        }}
        table.stat-table td {{
            padding: 9px 12px;
            border-bottom: 1px solid #ebedef;
        }}
        table.stat-table tr:hover {{
            background: #f2f9ff;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Dashboard Analítico Interactivo — Comercio Bilateral Colombia & Alianza del Pacífico (2013–2025)</h1>
        <p>Microdatos Oficiales DANE (EXPO + IMPO) | 16 Productos Clave (8 Ofertados y 8 Demandados) | Precios Reales Constantes (Base 2013=100) | Slicers de Desagregación Dinámica</p>
    </header>

    <section class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Microdatos Procesados (EXPO + IMPO)</div>
            <div class="kpi-value">3,809,047</div>
            <div class="kpi-sub">1.11M registros EXPO + 2.70M registros IMPO (84 meses)</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-title">Exportaciones Reales Canasta Ofertada</div>
            <div class="kpi-value">${total_expo_ofertada:,.1f} MUSD</div>
            <div class="kpi-sub">8 productos estratégicos hacia CHL, MEX y PER</div>
        </div>
        <div class="kpi-card amber">
            <div class="kpi-title">Importaciones Reales desde AP</div>
            <div class="kpi-value">${total_impo_ap:,.1f} MUSD</div>
            <div class="kpi-sub">Valor real deflactado (Base 2013=100)</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-title">Elasticidad Causal DML Variedad (ln N)</div>
            <div class="kpi-value">θ = +0.6370 (p = 8.8e-10)</div>
            <div class="kpi-sub">Margen extensivo impulsa las exportaciones reales</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Capacidad Predictiva ML (OOF 5-Fold)</div>
            <div class="kpi-value">R² = 0.9228</div>
            <div class="kpi-sub">HistGradientBoostingRegressor sin fuga simultánea</div>
        </div>
    </section>

    <nav class="tabs-bar">
        <button class="tab-btn active" onclick="openTab(event, 'tab_exec')">1. Panorama Ejecutivo y Explorador por Producto</button>
        <button class="tab-btn" onclick="openTab(event, 'tab_ofertados')">2. Los 8 Productos Ofertados (Exportaciones)</button>
        <button class="tab-btn" onclick="openTab(event, 'tab_demandados')">3. Los 8 Productos Demandados (Importaciones)</button>
        <button class="tab-btn" onclick="openTab(event, 'tab_stats')">4. Ajustes distfit y Pruebas de Hipótesis (16 Productos)</button>
        <button class="tab-btn" onclick="openTab(event, 'tab_models')">5. Econometría FE-OLS y Machine Learning Causal (DML)</button>
    </nav>

    <main>
        <!-- ================================================================== -->
        <!-- PESTAÑA 1: PANORAMA EJECUTIVO Y EXPLORADOR POR PRODUCTO CON SLICERS -->
        <!-- ================================================================== -->
        <section id="tab_exec" class="tab-content active">
            <!-- SECCIÓN 1A: PANORAMA EJECUTIVO BILATERAL -->
            <div class="section-block">
                <div class="section-header">
                    <div>
                        <h2 class="section-title">1A. Panorama Ejecutivo Bilateral: Flujos Reales, Equilibrio Comercial y Diversificación</h2>
                        <p class="section-subtitle">Utilice los slicers de desagregación para filtrar por Socio Comercial, Bloque Histórico, Frecuencia Temporal y Canasta.</p>
                    </div>
                </div>

                <div class="slicer-bar">
                    <div class="slicer-group">
                        <label class="slicer-label" for="slicer_exec_pais">🌎 Slicer Socio Comercial</label>
                        <select id="slicer_exec_pais" class="slicer-select" onchange="updateExecutiveOverview()">
                            <option value="ALL">Todos los Socios (Alianza del Pacífico)</option>
                            <option value="CHL">🇨🇱 Solo Chile (CHL)</option>
                            <option value="MEX">🇲🇽 Solo México (MEX)</option>
                            <option value="PER">🇵🇪 Solo Perú (PER)</option>
                        </select>
                    </div>
                    <div class="slicer-group">
                        <label class="slicer-label" for="slicer_exec_bloque">📅 Slicer Bloque Histórico</label>
                        <select id="slicer_exec_bloque" class="slicer-select" onchange="updateExecutiveOverview()">
                            <option value="ALL">Todo el Periodo (2013–2025)</option>
                            <option value="P1 (2013–14)">P1: Pre-Consolidación (2013–2014)</option>
                            <option value="P2 (2018–20)">P2: Consolidación / COVID (2018–2020)</option>
                            <option value="P3 (2024–25)">P3: Madurez Reciente (2024–2025)</option>
                        </select>
                    </div>
                    <div class="slicer-group">
                        <label class="slicer-label" for="slicer_exec_freq">⏱️ Slicer Granularidad Temporal</label>
                        <select id="slicer_exec_freq" class="slicer-select" onchange="updateExecutiveOverview()">
                            <option value="ANUAL">Evolución Anual Consolidada (7 Años)</option>
                            <option value="BLOQUE">Comparativa por Bloques (P1, P2, P3)</option>
                            <option value="MENSUAL">Serie Mensual Detallada (84 Cortes)</option>
                        </select>
                    </div>
                    <div class="slicer-group">
                        <label class="slicer-label" for="slicer_exec_canasta">📦 Slicer Canasta de Análisis</label>
                        <select id="slicer_exec_canasta" class="slicer-select" onchange="updateExecutiveOverview()">
                            <option value="Canasta Ofertada">Canasta Ofertada (8 Productos Clave)</option>
                            <option value="Canasta Demandada (Referencia)">Canasta Demandada (8 Productos Clave)</option>
                            <option value="ALL">Comercio Bilateral Total (Todas las Canastas)</option>
                        </select>
                    </div>
                </div>

                <div class="exec-charts-grid">
                    <div class="sub-chart-box">
                        <div id="plot_exec_flows" style="width:100%; height:460px;"></div>
                    </div>
                    <div class="sub-chart-box">
                        <div id="plot_exec_diagnostics" style="width:100%; height:440px;"></div>
                    </div>
                </div>
            </div>

            <!-- SECCIÓN 1B: EXPLORADOR DESAGREGADO POR PRODUCTO CLAVE -->
            <div class="section-block">
                <div class="section-header">
                    <div>
                        <h2 class="section-title">1B. Explorador Individual de los 16 Productos Clave con Slicers de Desagregación</h2>
                        <p class="section-subtitle">Seleccione cualquiera de los 8 Productos Ofertados o 8 Demandados y filtre por socio comercial, bloque histórico y nivel de agregación.</p>
                    </div>
                </div>

                <div class="slicer-bar">
                    <div class="slicer-group wide">
                        <label class="slicer-label" for="slicer_exp_prod">🔎 Slicer Producto Clave (16 Complejos Arancelarios)</label>
                        <select id="slicer_exp_prod" class="slicer-select" onchange="updateProductExplorer()">
                            <optgroup label="A. 8 Productos Ofertados (Exportaciones de Colombia)">
                                {opt_ofertados}
                            </optgroup>
                            <optgroup label="B. 8 Productos Demandados (Importaciones de Colombia)">
                                {opt_demandados}
                            </optgroup>
                        </select>
                    </div>
                    <div class="slicer-group">
                        <label class="slicer-label" for="slicer_exp_pais">🌎 Slicer Socio Comercial</label>
                        <select id="slicer_exp_pais" class="slicer-select" onchange="updateProductExplorer()">
                            <option value="ALL">Todos (Chile, México y Perú)</option>
                            <option value="CHL">🇨🇱 Solo Chile (CHL)</option>
                            <option value="MEX">🇲🇽 Solo México (MEX)</option>
                            <option value="PER">🇵🇪 Solo Perú (PER)</option>
                        </select>
                    </div>
                    <div class="slicer-group">
                        <label class="slicer-label" for="slicer_exp_bloque">📅 Slicer Bloque Temporal</label>
                        <select id="slicer_exp_bloque" class="slicer-select" onchange="updateProductExplorer()">
                            <option value="ALL">Todos los Bloques (2013–2025)</option>
                            <option value="P1 (2013–14)">P1: Pre-Consolidación (2013–2014)</option>
                            <option value="P2 (2018–20)">P2: Consolidación / COVID (2018–2020)</option>
                            <option value="P3 (2024–25)">P3: Madurez Reciente (2024–2025)</option>
                        </select>
                    </div>
                    <div class="slicer-group">
                        <label class="slicer-label" for="slicer_exp_freq">📊 Slicer Frecuencia Serie</label>
                        <select id="slicer_exp_freq" class="slicer-select" onchange="updateProductExplorer()">
                            <option value="ANUAL">Serie Anual por Socio (2013–2025)</option>
                            <option value="BLOQUE">Barras por Bloque (P1, P2, P3)</option>
                            <option value="MENSUAL">Serie Mensual (84 Cortes)</option>
                        </select>
                    </div>
                </div>

                <div id="explorer_meta_strip" class="explorer-meta-strip"></div>

                <div class="explorer-charts-grid">
                    <div class="sub-chart-box">
                        <div id="plot_explorer_ts" style="width:100%; height:440px;"></div>
                    </div>
                    <div class="sub-chart-box">
                        <div id="plot_explorer_dist" style="width:100%; height:440px;"></div>
                    </div>
                </div>
            </div>
        </section>

        <!-- ================================================================== -->
        <!-- PESTAÑA 2: LOS 8 PRODUCTOS OFERTADOS (TARJETAS ESPACIADAS + SLICERS) -->
        <!-- ================================================================== -->
        <section id="tab_ofertados" class="tab-content">
            <div class="section-block">
                <div class="section-header">
                    <div>
                        <h2 class="section-title">2. Desagregación de los 8 Productos Ofertados (Exportaciones Reales de Colombia hacia la Alianza)</h2>
                        <p class="section-subtitle">Cada producto cuenta con su propio lienzo individual espaciado y encabezados estadísticos sin superposición. Use los slicers para filtrar.</p>
                    </div>
                </div>

                <div class="slicer-bar">
                    <div class="slicer-group wide">
                        <label class="slicer-label" for="slicer_of_prod">🔎 Slicer de Producto Ofertado</label>
                        <select id="slicer_of_prod" class="slicer-select" onchange="updateBasketTab('of')">
                            {slicer_opts_of}
                        </select>
                    </div>
                    <div class="slicer-group">
                        <label class="slicer-label" for="slicer_of_pais">🌎 Slicer Socio Destino</label>
                        <select id="slicer_of_pais" class="slicer-select" onchange="updateBasketTab('of')">
                            <option value="ALL">Todos (Chile + México + Perú)</option>
                            <option value="CHL">🇨🇱 Solo Chile (CHL)</option>
                            <option value="MEX">🇲🇽 Solo México (MEX)</option>
                            <option value="PER">🇵🇪 Solo Perú (PER)</option>
                        </select>
                    </div>
                    <div class="slicer-group">
                        <label class="slicer-label" for="slicer_of_freq">⏱️ Slicer Eje Temporal</label>
                        <select id="slicer_of_freq" class="slicer-select" onchange="updateBasketTab('of')">
                            <option value="BLOQUE">Por Bloques Históricos (P1, P2, P3)</option>
                            <option value="ANUAL">Evolución Anual (2013–2025)</option>
                            <option value="MENSUAL">Serie Mensual (84 Cortes)</option>
                        </select>
                    </div>
                    <div class="slicer-group">
                        <label class="slicer-label" for="slicer_of_mode">📊 Slicer Estilo Visual</label>
                        <select id="slicer_of_mode" class="slicer-select" onchange="updateBasketTab('of')">
                            <option value="group">Barras Agrupadas por Socio</option>
                            <option value="stack">Barras Apiladas por Socio</option>
                            <option value="lines">Líneas de Tendencia</option>
                        </select>
                    </div>
                </div>

                <!-- Ranking resumen comparativo limpio -->
                <div class="sub-chart-box" style="margin-bottom: 28px;">
                    <div id="plot_of_summary" style="width:100%; height:390px;"></div>
                </div>

                <!-- Grilla de 8 tarjetas independientes espaciadas -->
                <div id="of_cards_grid" class="product-cards-grid">
                    {ofertados_cards_html}
                </div>
            </div>
        </section>

        <!-- ================================================================== -->
        <!-- PESTAÑA 3: LOS 8 PRODUCTOS DEMANDADOS (TARJETAS ESPACIADAS + SLICERS) -->
        <!-- ================================================================== -->
        <section id="tab_demandados" class="tab-content">
            <div class="section-block">
                <div class="section-header">
                    <div>
                        <h2 class="section-title">3. Desagregación de los 8 Productos Demandados (Importaciones Reales de Colombia desde la Alianza)</h2>
                        <p class="section-subtitle">Cada producto demandado dispone de su propia tarjeta independiente de alta legibilidad. Use los slicers para filtrar socio, periodo o producto.</p>
                    </div>
                </div>

                <div class="slicer-bar">
                    <div class="slicer-group wide">
                        <label class="slicer-label" for="slicer_dem_prod">🔎 Slicer de Producto Demandado</label>
                        <select id="slicer_dem_prod" class="slicer-select" onchange="updateBasketTab('dem')">
                            {slicer_opts_dem}
                        </select>
                    </div>
                    <div class="slicer-group">
                        <label class="slicer-label" for="slicer_dem_pais">🌎 Slicer Socio Origen</label>
                        <select id="slicer_dem_pais" class="slicer-select" onchange="updateBasketTab('dem')">
                            <option value="ALL">Todos (Chile + México + Perú)</option>
                            <option value="CHL">🇨🇱 Solo Chile (CHL)</option>
                            <option value="MEX">🇲🇽 Solo México (MEX)</option>
                            <option value="PER">🇵🇪 Solo Perú (PER)</option>
                        </select>
                    </div>
                    <div class="slicer-group">
                        <label class="slicer-label" for="slicer_dem_freq">⏱️ Slicer Eje Temporal</label>
                        <select id="slicer_dem_freq" class="slicer-select" onchange="updateBasketTab('dem')">
                            <option value="BLOQUE">Por Bloques Históricos (P1, P2, P3)</option>
                            <option value="ANUAL">Evolución Anual (2013–2025)</option>
                            <option value="MENSUAL">Serie Mensual (84 Cortes)</option>
                        </select>
                    </div>
                    <div class="slicer-group">
                        <label class="slicer-label" for="slicer_dem_mode">📊 Slicer Estilo Visual</label>
                        <select id="slicer_dem_mode" class="slicer-select" onchange="updateBasketTab('dem')">
                            <option value="group">Barras Agrupadas por Socio</option>
                            <option value="stack">Barras Apiladas por Socio</option>
                            <option value="lines">Líneas de Tendencia</option>
                        </select>
                    </div>
                </div>

                <!-- Ranking resumen comparativo limpio -->
                <div class="sub-chart-box" style="margin-bottom: 28px;">
                    <div id="plot_dem_summary" style="width:100%; height:390px;"></div>
                </div>

                <!-- Grilla de 8 tarjetas independientes espaciadas -->
                <div id="dem_cards_grid" class="product-cards-grid">
                    {demandados_cards_html}
                </div>
            </div>
        </section>

        <!-- ================================================================== -->
        <!-- PESTAÑA 4: AJUSTES DISTFIT Y PRUEBAS DE HIPÓTESIS                  -->
        <!-- ================================================================== -->
        <section id="tab_stats" class="tab-content">
            <div class="section-block">{html_stats}</div>
            <div class="section-block">
                <h3 style="margin-top:0;color:#1b4f72;">Matriz de Pruebas Paramétricas y No Paramétricas por Producto (P1: 2013–2014 vs. P3: 2024–2025)</h3>
                <div style="overflow-x:auto;">
                    <table class="stat-table">
                        <thead>
                            <tr>
                                <th>Canasta</th>
                                <th>Producto Clave</th>
                                <th>Socio Predom.</th>
                                <th>Media P1 (MUSD)</th>
                                <th>Media P3 (MUSD)</th>
                                <th>Δ Media (P3-P1)</th>
                                <th>t-Student Pareada (p)</th>
                                <th>Wilcoxon p-valor</th>
                                <th>Kruskal-Wallis H (p)</th>
                                <th>Significancia (α=0.05)</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows_joined}
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- ================================================================== -->
        <!-- PESTAÑA 5: ECONOMETRÍA FE-OLS Y MACHINE LEARNING CAUSAL (DML)      -->
        <!-- ================================================================== -->
        <section id="tab_models" class="tab-content">
            <div class="section-block">{html_models}</div>
        </section>
    </main>

    <script>
        const DASH_DATA = {js_payload_str};
        const PALETTE = {{ CHL: "#1b4f72", MEX: "#117864", PER: "#b9770e" }};
        const COUNTRY_LABELS = {{ CHL: "Chile", MEX: "México", PER: "Perú" }};
        const BLOCK_ORDER = ["P1 (2013–14)", "P2 (2018–20)", "P3 (2024–25)"];
        const YEAR_ORDER = ["2013", "2014", "2018", "2019", "2020", "2024", "2025"];

        function openTab(evt, tabId) {{
            var contents = document.getElementsByClassName("tab-content");
            for (var i = 0; i < contents.length; i++) {{
                contents[i].classList.remove("active");
            }}
            var buttons = document.getElementsByClassName("tab-btn");
            for (var j = 0; j < buttons.length; j++) {{
                buttons[j].classList.remove("active");
            }}
            var target = document.getElementById(tabId);
            target.classList.add("active");
            evt.currentTarget.classList.add("active");
            setTimeout(function() {{
                window.dispatchEvent(new Event('resize'));
            }}, 60);
        }}

        function groupSumBy(rows, keyFn, valField) {{
            const map = new Map();
            for (const r of rows) {{
                const k = keyFn(r);
                map.set(k, (map.get(k) || 0) + (Number(r[valField]) || 0));
            }}
            return map;
        }}

        /* ====================================================================
           1A. ACTUALIZACIÓN CON SLICERS DEL PANORAMA EJECUTIVO
           ==================================================================== */
        function updateExecutiveOverview() {{
            const paisFilter = document.getElementById("slicer_exec_pais").value;
            const bloqueFilter = document.getElementById("slicer_exec_bloque").value;
            const freq = document.getElementById("slicer_exec_freq").value;
            const canastaFilter = document.getElementById("slicer_exec_canasta").value;

            const countries = paisFilter === "ALL" ? ["CHL", "MEX", "PER"] : [paisFilter];
            let rows = DASH_DATA.panel.filter(r => {{
                if (paisFilter !== "ALL" && r.COD_PAI4 !== paisFilter) return false;
                if (bloqueFilter !== "ALL" && r.bloque_corto !== bloqueFilter) return false;
                if (canastaFilter !== "ALL" && r.tipo_canasta !== canastaFilter) return false;
                return true;
            }});

            let xLabels = [];
            let keyFn = null;
            if (freq === "BLOQUE") {{
                xLabels = BLOCK_ORDER.filter(b => bloqueFilter === "ALL" || b === bloqueFilter);
                keyFn = r => r.bloque_corto;
            }} else if (freq === "ANUAL") {{
                const setY = new Set(rows.map(r => String(r.anio)));
                xLabels = YEAR_ORDER.filter(y => setY.has(y));
                keyFn = r => String(r.anio);
            }} else {{
                xLabels = Array.from(new Set(rows.map(r => r.periodo_ym))).sort();
                keyFn = r => r.periodo_ym;
            }}

            const tracesFlows = [];
            for (const c of countries) {{
                const subC = rows.filter(r => r.COD_PAI4 === c);
                const expMap = groupSumBy(subC, keyFn, "fob_expo_real_musd");
                const impMap = groupSumBy(subC, keyFn, "cif_impo_real_musd");
                const yExp = xLabels.map(x => Number((expMap.get(x) || 0).toFixed(2)));
                const yImp = xLabels.map(x => Number((impMap.get(x) || 0).toFixed(2)));

                if (freq === "BLOQUE") {{
                    tracesFlows.push({{
                        x: xLabels,
                        y: yExp,
                        type: "bar",
                        name: `EXPO Real → ${{COUNTRY_LABELS[c]}}`,
                        marker: {{ color: PALETTE[c] }},
                        xaxis: "x",
                        yaxis: "y",
                        hovertemplate: `<b>EXPO → ${{COUNTRY_LABELS[c]}}</b><br>Periodo: %{{x}}<br>Valor Real: %{{y:.2f}} MUSD<extra></extra>`
                    }});
                    tracesFlows.push({{
                        x: xLabels,
                        y: yImp,
                        type: "bar",
                        name: `IMPO Real ← ${{COUNTRY_LABELS[c]}}`,
                        marker: {{ color: PALETTE[c], opacity: 0.65 }},
                        xaxis: "x2",
                        yaxis: "y2",
                        hovertemplate: `<b>IMPO ← ${{COUNTRY_LABELS[c]}}</b><br>Periodo: %{{x}}<br>Valor Real: %{{y:.2f}} MUSD<extra></extra>`
                    }});
                }} else {{
                    tracesFlows.push({{
                        x: xLabels,
                        y: yExp,
                        type: "scatter",
                        mode: "lines+markers",
                        name: `EXPO Real → ${{COUNTRY_LABELS[c]}}`,
                        line: {{ color: PALETTE[c], width: 2.5 }},
                        marker: {{ size: 6 }},
                        xaxis: "x",
                        yaxis: "y",
                        hovertemplate: `<b>EXPO → ${{COUNTRY_LABELS[c]}}</b><br>Periodo: %{{x}}<br>Valor Real: %{{y:.2f}} MUSD<extra></extra>`
                    }});
                    tracesFlows.push({{
                        x: xLabels,
                        y: yImp,
                        type: "scatter",
                        mode: "lines+markers",
                        name: `IMPO Real ← ${{COUNTRY_LABELS[c]}}`,
                        line: {{ color: PALETTE[c], width: 2.2, dash: "dot" }},
                        marker: {{ size: 6, symbol: "diamond" }},
                        xaxis: "x2",
                        yaxis: "y2",
                        hovertemplate: `<b>IMPO ← ${{COUNTRY_LABELS[c]}}</b><br>Periodo: %{{x}}<br>Valor Real: %{{y:.2f}} MUSD<extra></extra>`
                    }});
                }}
            }}

            const canastaTitle = canastaFilter === "ALL" ? "Todas las Canastas" : canastaFilter;
            const layoutFlows = {{
                template: "plotly_white",
                height: 460,
                barmode: "group",
                title: {{
                    text: `<b>Flujos Reales Deflactados (${{canastaTitle}}) — Desagregación: ${{freq}}</b>`,
                    x: 0.01,
                    font: {{ size: 15.5, color: "#1b4f72" }}
                }},
                grid: {{ rows: 1, columns: 2, pattern: "independent" }},
                annotations: [
                    {{ text: "<b>Exportaciones Reales FOB (MUSD Base 2013=100)</b>", x: 0.22, y: 1.07, xref: "paper", yref: "paper", showarrow: false, font: {{ size: 13 }} }},
                    {{ text: "<b>Importaciones Reales CIF (MUSD Base 2013=100)</b>", x: 0.78, y: 1.07, xref: "paper", yref: "paper", showarrow: false, font: {{ size: 13 }} }}
                ],
                xaxis: {{ title: "Periodo", type: "category", domain: [0, 0.46], automargin: true }},
                yaxis: {{ title: "Millones USD Reales (EXPO)", automargin: true }},
                xaxis2: {{ title: "Periodo", type: "category", domain: [0.54, 1.0], automargin: true }},
                yaxis2: {{ title: "Millones USD Reales (IMPO)", automargin: true }},
                legend: {{ orientation: "h", y: -0.18, x: 0.5, xanchor: "center" }},
                margin: {{ t: 75, b: 85, l: 65, r: 30 }}
            }};
            Plotly.react("plot_exec_flows", tracesFlows, layoutFlows, {{ responsive: true }});

            // Diagnóstico estructural: SCR vs Grubel-Lloyd y (1-HHI) vs N Subpartidas
            const tracesDiag = [];
            for (const c of countries) {{
                const subC = rows.filter(r => r.COD_PAI4 === c);
                tracesDiag.push({{
                    x: subC.map(r => r.scr_bilateral),
                    y: subC.map(r => r.grubel_lloyd),
                    text: subC.map(r => r.periodo_ym),
                    type: "scatter",
                    mode: "markers",
                    name: `${{COUNTRY_LABELS[c]}} (SCR vs GL)`,
                    marker: {{
                        color: PALETTE[c],
                        size: subC.map(r => Math.max(7, Math.min(20, r.fob_expo_real_musd / 2.2))),
                        opacity: 0.78,
                        line: {{ width: 0.8, color: "#fff" }}
                    }},
                    xaxis: "x",
                    yaxis: "y",
                    hovertemplate: `<b>${{COUNTRY_LABELS[c]}} (%{{text}})</b><br>SCR Bilateral: %{{x:.3f}}<br>Grubel-Lloyd: %{{y:.3f}}<extra></extra>`
                }});

                tracesDiag.push({{
                    x: subC.map(r => r.n_subpartidas_activas),
                    y: subC.map(r => r.hhi_invertido),
                    text: subC.map(r => r.periodo_ym),
                    type: "scatter",
                    mode: "markers",
                    name: `${{COUNTRY_LABELS[c]}} (1-HHI vs N)`,
                    showlegend: false,
                    marker: {{ color: PALETTE[c], size: 8.5, opacity: 0.78, symbol: "circle" }},
                    xaxis: "x2",
                    yaxis: "y2",
                    hovertemplate: `<b>${{COUNTRY_LABELS[c]}} (%{{text}})</b><br>Subpartidas Activas (N): %{{x}}<br>Diversificación (1 - HHI): %{{y:.3f}}<extra></extra>`
                }});
            }}

            const layoutDiag = {{
                template: "plotly_white",
                height: 440,
                title: {{
                    text: `<b>Diagnóstico Estructural: Especialización Intraindustrial (SCR vs. GL) y Diversificación Arancelaria (1 - HHI vs. N)</b>`,
                    x: 0.01,
                    font: {{ size: 15.5, color: "#1b4f72" }}
                }},
                grid: {{ rows: 1, columns: 2, pattern: "independent" }},
                annotations: [
                    {{ text: "<b>Saldo Comercial Relativo (SCR) vs. Grubel-Lloyd (GL)</b>", x: 0.22, y: 1.07, xref: "paper", yref: "paper", showarrow: false, font: {{ size: 13 }} }},
                    {{ text: "<b>Diversificación (1 - HHI) vs. Margen Extensivo (N Subpartidas)</b>", x: 0.78, y: 1.07, xref: "paper", yref: "paper", showarrow: false, font: {{ size: 13 }} }}
                ],
                xaxis: {{ title: "Saldo Comercial Relativo (SCR ∈ [-1, +1])", domain: [0, 0.46], automargin: true }},
                yaxis: {{ title: "Índice Grubel-Lloyd (GL ∈ [0, 1])", automargin: true }},
                xaxis2: {{ title: "Número de Subpartidas Activas (N_it)", domain: [0.54, 1.0], automargin: true }},
                yaxis2: {{ title: "Diversificación (1 - HHI)", automargin: true }},
                legend: {{ orientation: "h", y: -0.18, x: 0.5, xanchor: "center" }},
                margin: {{ t: 75, b: 85, l: 65, r: 30 }}
            }};
            Plotly.react("plot_exec_diagnostics", tracesDiag, layoutDiag, {{ responsive: true }});
        }}

        /* ====================================================================
           1B. ACTUALIZACIÓN CON SLICERS DEL EXPLORADOR POR PRODUCTO CLAVE
           ==================================================================== */
        function updateProductExplorer() {{
            const prodName = document.getElementById("slicer_exp_prod").value;
            const paisFilter = document.getElementById("slicer_exp_pais").value;
            const bloqueFilter = document.getElementById("slicer_exp_bloque").value;
            const freq = document.getElementById("slicer_exp_freq").value;

            const meta = DASH_DATA.meta[prodName];
            const valField = meta.is_ofertada ? "fob_expo_real_musd" : "cif_impo_real_musd";
            const flowUnitLabel = meta.is_ofertada ? "Exportaciones Reales FOB (MUSD)" : "Importaciones Reales CIF (MUSD)";
            const countries = paisFilter === "ALL" ? ["CHL", "MEX", "PER"] : [paisFilter];

            // Actualizar banda informativa del producto
            const sigBadge = meta.sig
                ? `<span class="meta-pill" style="background:#0e6251;color:#fff;">Cambio Significativo (Wilcoxon p=${{meta.wilcoxon_pval.toExponential(2)}})</span>`
                : `<span class="meta-pill" style="background:#5d6d7e;color:#fff;">Flujo Estable (Wilcoxon p=${{meta.wilcoxon_pval.toFixed(3)}})</span>`;

            document.getElementById("explorer_meta_strip").innerHTML = `
                <span class="meta-pill navy"><b>${{meta.canasta}}</b></span>
                <span class="meta-pill navy">Código HS4: <b>${{meta.hs4}}</b></span>
                <span class="meta-pill teal">Acumulado 2013–2025: <b>$${{meta.total_musd.toLocaleString()}} MUSD</b></span>
                <span class="meta-pill slate">Mejor Ley distfit: <b>${{meta.distfit}}</b> (Asimetría=${{meta.skew}})</span>
                <span class="meta-pill amber">Media P1: <b>$${{meta.media_p1}} MUSD/mes</b> → Media P3: <b>$${{meta.media_p3}} MUSD/mes</b> (Δ = ${{meta.delta_p1_p3 >= 0 ? '+' : ''}}${{meta.delta_p1_p3}} MUSD)</span>
                <span class="meta-pill slate">Socio Predominante: <b>${{meta.socio_pred}}</b> (Kruskal H=${{meta.kruskal_h}}, p=${{meta.kruskal_pval.toExponential(2)}})</span>
                ${{sigBadge}}
            `;

            const rows = DASH_DATA.prod_monthly.filter(r => {{
                if (r.producto_clave !== prodName) return false;
                if (paisFilter !== "ALL" && r.COD_PAI4 !== paisFilter) return false;
                if (bloqueFilter !== "ALL" && r.bloque_corto !== bloqueFilter) return false;
                return true;
            }});

            let xLabels = [];
            let keyFn = null;
            if (freq === "BLOQUE") {{
                xLabels = BLOCK_ORDER.filter(b => bloqueFilter === "ALL" || b === bloqueFilter);
                keyFn = r => r.bloque_corto;
            }} else if (freq === "ANUAL") {{
                const setY = new Set(rows.map(r => String(r.anio)));
                xLabels = YEAR_ORDER.filter(y => setY.has(y));
                keyFn = r => String(r.anio);
            }} else {{
                xLabels = Array.from(new Set(rows.map(r => r.periodo_ym))).sort();
                keyFn = r => r.periodo_ym;
            }}

            const tracesTS = [];
            for (const c of countries) {{
                const subC = rows.filter(r => r.COD_PAI4 === c);
                const valMap = groupSumBy(subC, keyFn, valField);
                const yVals = xLabels.map(x => Number((valMap.get(x) || 0).toFixed(3)));

                if (freq === "BLOQUE") {{
                    tracesTS.push({{
                        x: xLabels,
                        y: yVals,
                        type: "bar",
                        name: COUNTRY_LABELS[c],
                        marker: {{ color: PALETTE[c] }},
                        text: yVals.map(v => v > 0 ? v.toFixed(1) : ""),
                        textposition: "outside",
                        cliponaxis: false,
                        hovertemplate: `<b>${{prodName}}</b><br>Socio: ${{COUNTRY_LABELS[c]}}<br>Bloque: %{{x}}<br>${{flowUnitLabel}}: %{{y:.2f}} MUSD<extra></extra>`
                    }});
                }} else {{
                    tracesTS.push({{
                        x: xLabels,
                        y: yVals,
                        type: "scatter",
                        mode: "lines+markers",
                        name: COUNTRY_LABELS[c],
                        line: {{ color: PALETTE[c], width: 2.4 }},
                        marker: {{ size: 6 }},
                        hovertemplate: `<b>${{prodName}}</b><br>Socio: ${{COUNTRY_LABELS[c]}}<br>Periodo: %{{x}}<br>${{flowUnitLabel}}: %{{y:.3f}} MUSD<extra></extra>`
                    }});
                }}
            }}

            const layoutTS = {{
                template: "plotly_white",
                height: 440,
                barmode: "group",
                title: {{
                    text: `<b>1. Evolución Temporal (${{freq}}): ${{prodName.slice(0, 56)}}</b>`,
                    x: 0.01,
                    font: {{ size: 14.5, color: "#1b4f72" }}
                }},
                xaxis: {{ title: "Periodo", type: "category", automargin: true }},
                yaxis: {{ title: flowUnitLabel, automargin: true }},
                legend: {{ orientation: "h", y: -0.18, x: 0.5, xanchor: "center" }},
                margin: {{ t: 60, b: 80, l: 65, r: 25 }}
            }};
            Plotly.react("plot_explorer_ts", tracesTS, layoutTS, {{ responsive: true }});

            // Gráfico derecho: Distribución Mensual (Boxplot Horizontal + Media por Socio)
            const tracesDist = [];
            for (const c of countries) {{
                const subC = rows.filter(r => r.COD_PAI4 === c);
                tracesDist.push({{
                    x: subC.map(r => r[valField]),
                    y: subC.map(() => COUNTRY_LABELS[c]),
                    type: "box",
                    orientation: "h",
                    name: COUNTRY_LABELS[c],
                    marker: {{ color: PALETTE[c] }},
                    boxmean: true,
                    boxpoints: "outliers",
                    hovertemplate: `<b>${{COUNTRY_LABELS[c]}}</b><br>Flujo Mensual: %{{x:.3f}} MUSD<extra></extra>`
                }});
            }}

            const layoutDist = {{
                template: "plotly_white",
                height: 440,
                title: {{
                    text: `<b>2. Distribución Empírica Mensual por Socio (Boxplot + Media, Ley: ${{meta.distfit}})</b>`,
                    x: 0.01,
                    font: {{ size: 14.5, color: "#1b4f72" }}
                }},
                xaxis: {{ title: `${{flowUnitLabel}} Mensual`, automargin: true }},
                yaxis: {{ title: "Socio Comercial", automargin: true }},
                showlegend: false,
                margin: {{ t: 60, b: 80, l: 75, r: 25 }}
            }};
            Plotly.react("plot_explorer_dist", tracesDist, layoutDist, {{ responsive: true }});
        }}

        /* ====================================================================
           2 Y 3. ACTUALIZACIÓN CON SLICERS DE LAS 8 TARJETAS DE PRODUCTOS
           ==================================================================== */
        function updateBasketTab(prefix) {{
            const isOfertada = prefix === "of";
            const productList = isOfertada ? DASH_DATA.ofertados_list : DASH_DATA.demandados_list;
            const valField = isOfertada ? "fob_expo_real_musd" : "cif_impo_real_musd";
            const flowLabel = isOfertada ? "Exportaciones Reales FOB (MUSD)" : "Importaciones Reales CIF (MUSD)";

            const prodFilter = document.getElementById(`slicer_${{prefix}}_prod`).value;
            const paisFilter = document.getElementById(`slicer_${{prefix}}_pais`).value;
            const freq = document.getElementById(`slicer_${{prefix}}_freq`).value;
            const mode = document.getElementById(`slicer_${{prefix}}_mode`).value;

            const countries = paisFilter === "ALL" ? ["CHL", "MEX", "PER"] : [paisFilter];
            const gridEl = document.getElementById(`${{prefix}}_cards_grid`);

            if (prodFilter === "ALL") {{
                gridEl.classList.remove("single-col");
            }} else {{
                gridEl.classList.add("single-col");
            }}

            // 1. Actualizar el gráfico de Ranking Resumen Superior
            const summaryTraces = [];
            for (const c of countries) {{
                const xVals = [];
                const yLabels = [];
                for (let idx = productList.length - 1; idx >= 0; idx--) {{
                    const pName = productList[idx];
                    const shortP = `#${{idx + 1}} ` + (pName.length > 42 ? pName.slice(0, 42) + "..." : pName);
                    const subRows = DASH_DATA.prod_block.filter(r => r.producto_clave === pName && r.COD_PAI4 === c);
                    const sumVal = subRows.reduce((acc, r) => acc + (Number(r[valField]) || 0), 0);
                    yLabels.push(shortP);
                    xVals.push(Number(sumVal.toFixed(2)));
                }}
                summaryTraces.push({{
                    x: xVals,
                    y: yLabels,
                    type: "bar",
                    orientation: "h",
                    name: COUNTRY_LABELS[c],
                    marker: {{ color: PALETTE[c] }},
                    hovertemplate: `<b>%{{y}}</b><br>Socio: ${{COUNTRY_LABELS[c]}}<br>Acumulado Real: %{{x:.2f}} MUSD<extra></extra>`
                }});
            }}

            const summaryLayout = {{
                template: "plotly_white",
                height: 390,
                barmode: mode === "stack" ? "stack" : "group",
                title: {{
                    text: `<b>Ranking Comparativo Acumulado (2013–2025) — ${{isOfertada ? '8 Productos Ofertados (EXPO)' : '8 Productos Demandados (IMPO)'}}</b>`,
                    x: 0.01,
                    font: {{ size: 15, color: "#1b4f72" }}
                }},
                xaxis: {{ title: `${{flowLabel}} Acumulados (2013–2025)`, automargin: true }},
                yaxis: {{ automargin: true, tickfont: {{ size: 11.5 }} }},
                legend: {{ orientation: "h", y: 1.11, x: 0.99, xanchor: "right" }},
                margin: {{ t: 65, b: 55, l: 290, r: 35 }}
            }};
            Plotly.react(`plot_${{prefix}}_summary`, summaryTraces, summaryLayout, {{ responsive: true }});

            // 2. Actualizar cada una de las 8 tarjetas individuales
            for (let idx = 0; idx < productList.length; idx++) {{
                const cardEl = document.getElementById(`${{prefix}}_card_${{idx}}`);
                if (prodFilter !== "ALL" && Number(prodFilter) !== idx) {{
                    cardEl.style.display = "none";
                    continue;
                }}
                cardEl.style.display = "flex";

                const pName = productList[idx];
                const rows = DASH_DATA.prod_monthly.filter(r => r.producto_clave === pName);

                let xLabels = [];
                let keyFn = null;
                if (freq === "BLOQUE") {{
                    xLabels = BLOCK_ORDER;
                    keyFn = r => r.bloque_corto;
                }} else if (freq === "ANUAL") {{
                    xLabels = YEAR_ORDER;
                    keyFn = r => String(r.anio);
                }} else {{
                    xLabels = Array.from(new Set(rows.map(r => r.periodo_ym))).sort();
                    keyFn = r => r.periodo_ym;
                }}

                const cardTraces = [];
                for (const c of countries) {{
                    const subC = rows.filter(r => r.COD_PAI4 === c);
                    const valMap = groupSumBy(subC, keyFn, valField);
                    const yVals = xLabels.map(x => Number((valMap.get(x) || 0).toFixed(2)));

                    if (mode === "lines" || freq === "MENSUAL") {{
                        cardTraces.push({{
                            x: xLabels,
                            y: yVals,
                            type: "scatter",
                            mode: freq === "MENSUAL" ? "lines" : "lines+markers",
                            name: COUNTRY_LABELS[c],
                            line: {{ color: PALETTE[c], width: 2.3 }},
                            marker: {{ size: 6 }},
                            hovertemplate: `<b>${{COUNTRY_LABELS[c]}}</b><br>Periodo: %{{x}}<br>Flujo Real: %{{y:.2f}} MUSD<extra></extra>`
                        }});
                    }} else {{
                        cardTraces.push({{
                            x: xLabels,
                            y: yVals,
                            type: "bar",
                            name: COUNTRY_LABELS[c],
                            marker: {{ color: PALETTE[c] }},
                            hovertemplate: `<b>${{COUNTRY_LABELS[c]}}</b><br>Periodo: %{{x}}<br>Flujo Real: %{{y:.2f}} MUSD<extra></extra>`
                        }});
                    }}
                }}

                const cardHeight = prodFilter === "ALL" ? 350 : 450;
                const cardLayout = {{
                    template: "plotly_white",
                    height: cardHeight,
                    barmode: mode === "stack" ? "stack" : "group",
                    bargap: 0.24,
                    xaxis: {{
                        title: freq === "BLOQUE" ? "Bloque Histórico" : (freq === "ANUAL" ? "Año" : "Corte Mensual"),
                        type: "category",
                        nticks: freq === "MENSUAL" ? 10 : undefined,
                        tickangle: 0,
                        automargin: true
                    }},
                    yaxis: {{
                        title: "MUSD Reales",
                        automargin: true
                    }},
                    legend: {{
                        orientation: "h",
                        y: 1.14,
                        x: 0.5,
                        xanchor: "center",
                        font: {{ size: 11.5 }}
                    }},
                    margin: {{ t: 42, b: 52, l: 60, r: 20 }}
                }};
                Plotly.react(`${{prefix}}_plot_${{idx}}`, cardTraces, cardLayout, {{ responsive: true }});
            }}
        }}

        document.addEventListener("DOMContentLoaded", function() {{
            updateExecutiveOverview();
            updateProductExplorer();
            updateBasketTab("of");
            updateBasketTab("dem");
        }});
    </script>
</body>
</html>
"""
        self._paths.output_html_path.parent.mkdir(parents=True, exist_ok=True)
        self._paths.output_html_path.write_text(full_html, encoding="utf-8")
        LOGGER.info(
            "Dashboard Analítico Plotly generado exitosamente en: %s (%.2f KB)",
            self._paths.output_html_path,
            self._paths.output_html_path.stat().st_size / 1024.0,
        )
        return self._paths.output_html_path


def main() -> None:
    """Punto de entrada para construir el Dashboard Analítico con Plotly."""
    root = Path(__file__).resolve().parents[2]
    paths = DashboardPaths.from_root(root)
    builder = AnalyticalDashboardBuilder(paths)
    builder.build_standalone_html()


if __name__ == "__main__":
    main()
