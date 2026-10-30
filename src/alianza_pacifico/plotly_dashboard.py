"""Generador del Dashboard Analítico Interactivo con Plotly para la Alianza del Pacífico.

Este módulo implementa los patrones Builder y Factory Method (GoF) bajo los
principios SOLID y Clean Code (PEP 8) para construir:
1. Un Dashboard Analítico Ejecutivo y Estadístico autónomo en HTML5 + Plotly.js
   (`docs/dashboard_analitico_alianza_pacifico.html`) con navegación por pestañas,
   tarjetas KPI, filtros interactivos por producto/socio/periodo y múltiples lienzos
   interactivos de alta densidad informativa.
2. Figuras maestras interactivas con controles `updatemenus` (dropdowns y botones) y
   `rangeslider` integrables directamente en `Analisis_exportaciones.ipynb`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

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
        panel["fob_expo_real_musd"] = panel["exportaciones_reales"] / 1e6
        panel["cif_impo_real_musd"] = panel["importaciones_reales"] / 1e6

        prod_monthly = pd.read_csv(self._dir / "panel_mensual_16_productos_ap.csv")
        prod_monthly["fecha"] = pd.to_datetime(prod_monthly["fecha"])
        prod_monthly["fob_expo_real_musd"] = prod_monthly["expo_real_usd"] / 1e6
        prod_monthly["cif_impo_real_musd"] = prod_monthly["impo_real_usd"] / 1e6

        prod_block = pd.read_csv(self._dir / "analisis_desagregado_productos_ap.csv")
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
    """Fábrica de gráficos interactivos Plotly desagregados (Factory Method / OCP)."""

    PALETTE: Dict[str, str] = {
        "CHL": "#1b4f72",
        "MEX": "#117864",
        "PER": "#b9770e",
        "Ofertada": "#0e6251",
        "Demandada": "#78281f",
    }

    @classmethod
    def build_tab1_executive_overview(cls, data: Dict[str, pd.DataFrame]) -> go.Figure:
        """Construye el lienzo ejecutivo de series reales, SCR, Grubel-Lloyd y HHI."""
        panel = data["panel"]
        ts_ofertada = (
            panel[panel["tipo_canasta"] == "Canasta Ofertada"]
            .groupby(["fecha", "pais_nombre", "COD_PAI4"], as_index=False)
            .agg(
                fob_expo_real_musd=("fob_expo_real_musd", "sum"),
                cif_impo_real_musd=("cif_impo_real_musd", "sum"),
                scr_bilateral=("scr_bilateral", "mean"),
                grubel_lloyd=("grubel_lloyd", "mean"),
                hhi_invertido=("hhi_invertido", "mean"),
                n_subpartidas=("n_subpartidas_activas", "sum"),
            )
            .sort_values("fecha")
        )

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Exportaciones Reales de la Canasta Ofertada por Socio (MUSD, Base 2013=100)",
                "Importaciones Reales Totales desde la Alianza por Socio (MUSD, Base 2013=100)",
                "Saldo Comercial Relativo Bilateral (SCR) vs. Índice Grubel-Lloyd (GL)",
                "Diversificación (1 - HHI) vs. Margen Extensivo (N Subpartidas Activas)",
            ),
            vertical_spacing=0.14,
            horizontal_spacing=0.09,
        )

        impo_ts = (
            panel.groupby(["fecha", "pais_nombre", "COD_PAI4"], as_index=False)[
                "cif_impo_real_musd"
            ]
            .sum()
            .sort_values("fecha")
        )

        for code, name in COUNTRY_NAMES.items():
            sub_exp = ts_ofertada[ts_ofertada["COD_PAI4"] == code]
            sub_imp = impo_ts[impo_ts["COD_PAI4"] == code]
            color = cls.PALETTE.get(code, "#333333")

            fig.add_trace(
                go.Scatter(
                    x=sub_exp["fecha"],
                    y=sub_exp["fob_expo_real_musd"],
                    mode="lines+markers",
                    name=f"EXPO Ofertada -> {name}",
                    line=dict(color=color, width=2.2),
                    marker=dict(size=5),
                    legendgroup=code,
                    hovertemplate=(
                        f"<b>{name} (Canasta Ofertada)</b><br>"
                        "Fecha: %{x|%Y-%m}<br>EXPO Real: %{y:.2f} MUSD<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=sub_imp["fecha"],
                    y=sub_imp["cif_impo_real_musd"],
                    mode="lines+markers",
                    name=f"IMPO Total <- {name}",
                    line=dict(color=color, width=2.0, dash="dot"),
                    marker=dict(size=5, symbol="diamond"),
                    legendgroup=code,
                    hovertemplate=(
                        f"<b>{name} (Importaciones Reales)</b><br>"
                        "Fecha: %{x|%Y-%m}<br>IMPO Real: %{y:.2f} MUSD<extra></extra>"
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
                        size=np.clip(sub_exp["fob_expo_real_musd"] / 2.5, 6, 22),
                        opacity=0.75,
                        line=dict(width=0.8, color="white"),
                    ),
                    legendgroup=code,
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

        fig.update_xaxes(title_text="Fecha Mensual", row=1, col=1)
        fig.update_xaxes(title_text="Fecha Mensual", row=1, col=2)
        fig.update_xaxes(title_text="Saldo Comercial Relativo (SCR ∈ [-1, 1])", row=2, col=1)
        fig.update_xaxes(title_text="Número de Subpartidas Activas (N_it)", row=2, col=2)

        fig.update_yaxes(title_text="Millones USD Reales (Exportaciones)", row=1, col=1)
        fig.update_yaxes(title_text="Millones USD Reales (Importaciones)", row=1, col=2)
        fig.update_yaxes(title_text="Índice Grubel-Lloyd (GL ∈ [0, 1])", row=2, col=1)
        fig.update_yaxes(title_text="Índice Diversificación D_it = 1 - HHI", row=2, col=2)

        fig.update_layout(
            template="plotly_white",
            height=780,
            title_text=(
                "<b>Panorama Ejecutivo Bilateral: Colombia — Alianza del Pacífico (2013–2025)</b>"
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5),
            margin=dict(t=100, b=50, l=60, r=40),
        )
        return fig

    @classmethod
    def build_product_basket_dashboard(
        cls,
        data: Dict[str, pd.DataFrame],
        basket_type: str,
    ) -> go.Figure:
        """Construye un dashboard interactivo con selector de vista y grilla 4x2 por producto."""
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
            flow_label = "Importaciones Reales (MUSD, Base 2013=100)"
            title_prefix = "8 Productos Demandados (Importados por Colombia desde la AP)"

        dist_map = dict(zip(distfit_16["producto_clave"], distfit_16["mejor_ley_distfit"]))
        pval_map = dict(zip(tests_16["producto_clave"], tests_16["wilcoxon_pvalue"]))

        short_titles = [
            f"{p[:44]}... [distfit: {dist_map.get(p, 'N/A')} | p_W={pval_map.get(p, 1.0):.3g}]"
            if len(p) > 44
            else f"{p} [distfit: {dist_map.get(p, 'N/A')} | p_W={pval_map.get(p, 1.0):.3g}]"
            for p in product_list
        ]

        fig = make_subplots(
            rows=4,
            cols=2,
            subplot_titles=short_titles,
            vertical_spacing=0.09,
            horizontal_spacing=0.07,
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
                        x=sub_c["bloque_temporal"],
                        y=sub_c[val_col],
                        name=country_label,
                        marker_color=cls.PALETTE[code],
                        legendgroup=code,
                        showlegend=(idx == 0),
                        visible=True,
                        text=[f"{v:.1f}" if v >= 0.5 else "" for v in sub_c[val_col]],
                        textposition="auto",
                        hovertemplate=(
                            f"<b>{prod_name}</b><br>Socio: {country_label}<br>"
                            "Periodo: %{x}<br>Valor Real Acumulado: %{y:.2f} MUSD<extra></extra>"
                        ),
                    ),
                    row=r,
                    col=c,
                )
                n_bar_traces += 1

        n_line_traces = 0
        for idx, prod_name in enumerate(product_list):
            r = (idx // 2) + 1
            c = (idx % 2) + 1
            sub_m = prod_monthly[
                (prod_monthly["tipo_canasta"] == basket_type)
                & (prod_monthly["producto_clave"] == prod_name)
            ].sort_values("fecha")
            for code, country_label in COUNTRY_NAMES.items():
                sub_mc = sub_m[sub_m["COD_PAI4"] == code]
                fig.add_trace(
                    go.Scatter(
                        x=sub_mc["fecha"],
                        y=sub_mc[val_col],
                        mode="lines",
                        name=f"{country_label} (Mensual)",
                        line=dict(color=cls.PALETTE[code], width=1.6),
                        legendgroup=code,
                        showlegend=(idx == 0),
                        visible=False,
                        hovertemplate=(
                            f"<b>{prod_name}</b><br>Socio: {country_label}<br>"
                            "Fecha: %{x|%Y-%m}<br>Flujo Mensual: %{y:.2f} MUSD<extra></extra>"
                        ),
                    ),
                    row=r,
                    col=c,
                )
                n_line_traces += 1

        vis_bars = [True] * n_bar_traces + [False] * n_line_traces
        vis_lines = [False] * n_bar_traces + [True] * n_line_traces

        fig.update_layout(
            template="plotly_white",
            height=1120,
            barmode="group",
            title_text=f"<b>Desagregación Completa: {title_prefix}</b><br><sup>{flow_label}</sup>",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.5,
                    xanchor="center",
                    y=1.065,
                    yanchor="top",
                    buttons=[
                        dict(
                            label="📊 Vista por Bloques Temporales (P1, P2, P3)",
                            method="update",
                            args=[{"visible": vis_bars}],
                        ),
                        dict(
                            label="📈 Vista de Series Mensuales (84 Meses: 2013–2025)",
                            method="update",
                            args=[{"visible": vis_lines}],
                        ),
                    ],
                )
            ],
            margin=dict(t=135, b=50, l=55, r=35),
        )
        return fig

    @classmethod
    def build_product_drilldown_explorer(cls, data: Dict[str, pd.DataFrame]) -> go.Figure:
        """Construye un explorador interactivo individual con dropdown para los 16 productos."""
        prod_monthly = data["prod_monthly"]
        distfit_16 = data["distfit_16"]
        tests_16 = data["tests_16"]

        all_products = PRODUCTOS_OFERTADOS_LIST + PRODUCTOS_DEMANDADOS_LIST
        dist_map = dict(zip(distfit_16["producto_clave"], distfit_16["mejor_ley_distfit"]))
        test_map = tests_16.set_index("producto_clave").to_dict(orient="index")

        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.65, 0.35],
            subplot_titles=(
                "Serie Mensual Desagregada por Socio (CHL, MEX, PER) — 84 Meses",
                "Distribución Mensual por Socio (Boxplot + Media)",
            ),
            horizontal_spacing=0.08,
        )

        traces_per_product = 6
        for p_idx, prod_name in enumerate(all_products):
            sub_p = prod_monthly[prod_monthly["producto_clave"] == prod_name].sort_values("fecha")
            is_ofertada = prod_name in PRODUCTOS_OFERTADOS_LIST
            val_col = "fob_expo_real_musd" if is_ofertada else "cif_impo_real_musd"
            visible_flag = p_idx == 0

            for code, country_label in COUNTRY_NAMES.items():
                sub_c = sub_p[sub_p["COD_PAI4"] == code]
                fig.add_trace(
                    go.Scatter(
                        x=sub_c["fecha"],
                        y=sub_c[val_col],
                        mode="lines+markers",
                        name=f"{country_label} (Serie)",
                        line=dict(color=cls.PALETTE[code], width=2.0),
                        marker=dict(size=4),
                        visible=visible_flag,
                        hovertemplate=(
                            f"<b>{prod_name}</b><br>Socio: {country_label}<br>"
                            "Fecha: %{x|%Y-%m}<br>Valor Real: %{y:.3f} MUSD<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )

            for code, country_label in COUNTRY_NAMES.items():
                sub_c = sub_p[sub_p["COD_PAI4"] == code]
                fig.add_trace(
                    go.Box(
                        y=sub_c[val_col],
                        name=country_label,
                        marker_color=cls.PALETTE[code],
                        boxmean=True,
                        visible=visible_flag,
                        hovertemplate=(
                            f"<b>{prod_name} ({country_label})</b><br>"
                            "Valor Mensual: %{y:.3f} MUSD<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=2,
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
                    label=f"[{basket_tag[:4]}] {prod_name[:48]}",
                    method="update",
                    args=[
                        {"visible": vis},
                        {
                            "title.text": (
                                f"<b>Explorador Individual de Producto: {prod_name}</b><br>"
                                f"<sup>Canasta: {basket_tag} | Mejor Ajuste distfit: {d_law} | "
                                f"Δ Media (P3-P1): {delta_m:+.2f} MUSD/mes | "
                                f"Wilcoxon p-valor: {p_wilcox:.4g}</sup>"
                            )
                        },
                    ],
                )
            )

        first_prod = all_products[0]
        first_info = test_map.get(first_prod, {})
        fig.update_layout(
            template="plotly_white",
            height=540,
            title_text=(
                f"<b>Explorador Individual de Producto: {first_prod}</b><br>"
                f"<sup>Canasta: OFERTADO (EXPO) | Mejor Ajuste distfit: {dist_map.get(first_prod)} | "
                f"Δ Media (P3-P1): {first_info.get('delta_media_p1_p3_musd', 0.0):+.2f} MUSD/mes | "
                f"Wilcoxon p-valor: {first_info.get('wilcoxon_pvalue', 1.0):.4g}</sup>"
            ),
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=0.0,
                    xanchor="left",
                    y=1.18,
                    yanchor="top",
                    bgcolor="#f8f9f9",
                    bordercolor="#1b4f72",
                )
            ],
            margin=dict(t=120, b=50, l=55, r=35),
        )
        fig.update_xaxes(rangeslider_visible=True, row=1, col=1)
        return fig

    @classmethod
    def build_tab4_statistical_inference_and_distfit(
        cls, data: Dict[str, pd.DataFrame]
    ) -> go.Figure:
        """Construye el panel de diagnóstico estadístico (distfit, P1 vs P3 y Matriz SCR/GL)."""
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
            p[:38] + "..." if len(p) > 38 else p for p in merged["producto_clave"]
        ]

        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.48, 0.52],
            subplot_titles=(
                "Cambio Estructural P1 (2013–14) vs P3 (2024–25) y Ley distfit",
                "Mapa de Calor Bilateral: Saldo Comercial Relativo (SCR) por Producto y Socio",
            ),
            horizontal_spacing=0.12,
        )

        fig.add_trace(
            go.Bar(
                y=short_names,
                x=merged["media_p1_musd"],
                name="Media P1 (2013–2014)",
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
                name="Media P3 (2024–2025)",
                orientation="h",
                marker_color="#117a65",
                text=[
                    f"{law} (p={pv:.2g})"
                    for law, pv in zip(merged["mejor_ley_distfit"], merged["wilcoxon_pvalue"])
                ],
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>Media P3: %{x:.2f} MUSD/mes<br>"
                    "Ajuste y Wilcoxon: %{text}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        prod_block["col_key"] = (
            prod_block["COD_PAI4"] + "_" + prod_block["bloque_temporal"].str.slice(0, 2)
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

        y_heat = [p[:38] + "..." if len(p) > 38 else p for p in pivot_scr.index]
        fig.add_trace(
            go.Heatmap(
                z=pivot_scr.values,
                x=list(pivot_scr.columns),
                y=y_heat,
                colorscale="RdBu",
                zmid=0.0,
                zmin=-1.0,
                zmax=1.0,
                colorbar=dict(title="SCR", x=1.02),
                customdata=pivot_gl.values,
                hovertemplate=(
                    "<b>%{y}</b><br>Socio-Periodo: %{x}<br>"
                    "SCR Bilateral: %{z:.3f}<br>Grubel-Lloyd (GL): %{customdata:.3f}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            template="plotly_white",
            height=680,
            barmode="group",
            title_text=(
                "<b>Inferencia Estadística No Paramétrica/Paramétrica y Especialización "
                "Intraindustrial (16 Productos Clave)</b>"
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
            margin=dict(t=95, b=50, l=220, r=60),
        )
        return fig

    @classmethod
    def build_tab5_econometrics_and_causal_ml(cls, data: Dict[str, pd.DataFrame]) -> go.Figure:
        """Construye el lienzo de Econometría FE-OLS, DML Causal y Evaluación ML."""
        reg_fe = data["reg_fe"]
        dml_df = data["dml_causal"]
        ml_eval = data["ml_eval"]
        perm_imp = data["perm_imp"].head(8).sort_values("importance_mean", ascending=True)

        fe_rows = reg_fe[reg_fe["variable"] != "constante"].copy()

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Elasticidades FE-OLS (IC 95% Robusto HC3) sobre ln(X_real)",
                "Efectos Causales Double Machine Learning (DML — Robinson 1988)",
                "Desempeño Predictivo Out-of-Fold (5-Fold CV vs. Walk-Forward TS): R² y RMSE",
                "Importancia por Permutación (Δ R² Out-of-Sample)",
            ),
            vertical_spacing=0.16,
            horizontal_spacing=0.10,
        )

        fig.add_trace(
            go.Scatter(
                x=fe_rows["coeficiente"],
                y=fe_rows["variable"],
                mode="markers",
                name="FE-OLS (HC3)",
                marker=dict(color="#1b4f72", size=11),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=fe_rows["ic_sup_95"] - fe_rows["coeficiente"],
                    arrayminus=fe_rows["coeficiente"] - fe_rows["ic_inf_95"],
                    color="#1b4f72",
                    thickness=2.0,
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
                x=dml_df["tratamiento"],
                y=dml_df["ate_dml_theta"],
                name="Efecto Causal DML (θ)",
                marker_color=[
                    "#117864" if sig else "#7f8c8d" for sig in dml_df["significativo_95"]
                ],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=dml_df["ic_sup_95"] - dml_df["ate_dml_theta"],
                    arrayminus=dml_df["ate_dml_theta"] - dml_df["ic_inf_95"],
                ),
                text=[
                    f"θ={v:+.3f} (p={p:.2g})"
                    for v, p in zip(dml_df["ate_dml_theta"], dml_df["p_value"])
                ],
                textposition="auto",
                hovertemplate="<b>%{x}</b><br>Efecto Causal DML: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        short_ml = ["Ridge FE", "Random Forest", "HistGradBoost"]
        fig.add_trace(
            go.Bar(
                x=short_ml,
                y=ml_eval["cv_r2_oof"],
                name="R² 5-Fold OOF",
                marker_color="#1f618d",
                text=[f"R²={v:.3f}" for v in ml_eval["cv_r2_oof"]],
                textposition="auto",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=short_ml,
                y=ml_eval["timeseries_r2_mean"],
                name="R² Walk-Forward TS",
                marker_color="#117a65",
                text=[f"TS R²={v:.3f}" for v in ml_eval["timeseries_r2_mean"]],
                textposition="auto",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=short_ml,
                y=ml_eval["cv_rmse_log"],
                name="RMSE (escala log)",
                marker_color="#d68910",
                text=[f"RMSE={v:.3f}" for v in ml_eval["cv_rmse_log"]],
                textposition="auto",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                y=perm_imp["feature"],
                x=perm_imp["importance_mean"],
                orientation="h",
                name="Importancia Permutación (Δ R²)",
                marker_color="#117a65",
                error_x=dict(type="data", array=perm_imp["importance_std"]),
                text=[f"{v:.4f}" for v in perm_imp["importance_mean"]],
                textposition="auto",
                hovertemplate="<b>%{y}</b><br>Caída media en R²: %{x:.4f}<extra></extra>",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            template="plotly_white",
            height=740,
            barmode="group",
            title_text=(
                "<b>Modelado Econométrico de Efectos Fijos (FE-OLS HC3) y "
                "Machine Learning Causal (Double Machine Learning)</b>"
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(t=100, b=50, l=65, r=40),
        )
        return fig


class AnalyticalDashboardBuilder:
    """Constructor del archivo HTML autónomo multi-pestaña (Patrón Builder)."""

    def __init__(self, paths: DashboardPaths) -> None:
        self._paths = paths
        self._repo = DashboardDataRepository(paths.processed_dir)

    def build_standalone_html(self) -> Path:
        """Genera el archivo HTML interactivo completo con pestañas y KPIs."""
        data = self._repo.load_all()

        fig_exec = PlotlyChartFactory.build_tab1_executive_overview(data)
        fig_ofertados = PlotlyChartFactory.build_product_basket_dashboard(
            data, "Canasta Ofertada"
        )
        fig_demandados = PlotlyChartFactory.build_product_basket_dashboard(
            data, "Canasta Demandada (Referencia)"
        )
        fig_explorer = PlotlyChartFactory.build_product_drilldown_explorer(data)
        fig_stats = PlotlyChartFactory.build_tab4_statistical_inference_and_distfit(data)
        fig_models = PlotlyChartFactory.build_tab5_econometrics_and_causal_ml(data)

        panel = data["panel"]
        total_expo_ofertada = panel.loc[
            panel["tipo_canasta"] == "Canasta Ofertada", "fob_expo_real_musd"
        ].sum()
        total_impo_ap = panel["cif_impo_real_musd"].sum()

        html_exec = fig_exec.to_html(full_html=False, include_plotlyjs="cdn", div_id="plot_exec")
        html_ofertados = fig_ofertados.to_html(
            full_html=False, include_plotlyjs=False, div_id="plot_ofertados"
        )
        html_demandados = fig_demandados.to_html(
            full_html=False, include_plotlyjs=False, div_id="plot_demandados"
        )
        html_explorer = fig_explorer.to_html(
            full_html=False, include_plotlyjs=False, div_id="plot_explorer"
        )
        html_stats = fig_stats.to_html(
            full_html=False, include_plotlyjs=False, div_id="plot_stats"
        )
        html_models = fig_models.to_html(
            full_html=False, include_plotlyjs=False, div_id="plot_models"
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
    <style>
        :root {{
            --primary: #1b4f72;
            --secondary: #117864;
            --accent: #b9770e;
            --bg: #f4f6f9;
            --card: #ffffff;
            --text: #1c2833;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
        }}
        header {{
            background: linear-gradient(135deg, #1b4f72 0%, #117864 100%);
            color: white;
            padding: 22px 36px;
            box-shadow: 0 3px 12px rgba(0,0,0,0.15);
        }}
        header h1 {{
            margin: 0 0 6px 0;
            font-size: 24px;
            font-weight: 700;
        }}
        header p {{
            margin: 0;
            font-size: 14px;
            opacity: 0.92;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 14px;
            padding: 18px 36px 8px 36px;
        }}
        .kpi-card {{
            background: var(--card);
            border-radius: 10px;
            padding: 14px 18px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.06);
            border-left: 5px solid var(--primary);
        }}
        .kpi-card.green {{ border-left-color: var(--secondary); }}
        .kpi-card.amber {{ border-left-color: var(--accent); }}
        .kpi-title {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #566573;
            margin-bottom: 4px;
        }}
        .kpi-value {{
            font-size: 22px;
            font-weight: 700;
            color: var(--text);
        }}
        .kpi-sub {{
            font-size: 11.5px;
            color: #7f8c8d;
            margin-top: 3px;
        }}
        .tabs-bar {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            padding: 14px 36px 0 36px;
            border-bottom: 2px solid #d5dbdb;
        }}
        .tab-btn {{
            background: #e5e8e8;
            border: none;
            padding: 11px 18px;
            border-radius: 8px 8px 0 0;
            font-size: 13.5px;
            font-weight: 600;
            color: #2c3e50;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .tab-btn:hover {{
            background: #d4e6f1;
        }}
        .tab-btn.active {{
            background: var(--primary);
            color: white;
        }}
        .tab-content {{
            display: none;
            padding: 20px 36px 40px 36px;
        }}
        .tab-content.active {{
            display: block;
        }}
        .chart-card {{
            background: var(--card);
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 22px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
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
            padding: 10px 10px;
            text-align: left;
        }}
        table.stat-table td {{
            padding: 8px 10px;
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
        <p>Microdatos Oficiales DANE (EXPO + IMPO) | 16 Productos Clave (8 Ofertados y 8 Demandados) | Precios Reales Constantes (Base 2013=100) | Inferencia Paramétrica/No Paramétrica, Econometría FE-OLS y Double Machine Learning (DML)</p>
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
        <section id="tab_exec" class="tab-content active">
            <div class="chart-card">{html_explorer}</div>
            <div class="chart-card">{html_exec}</div>
        </section>

        <section id="tab_ofertados" class="tab-content">
            <div class="chart-card">{html_ofertados}</div>
        </section>

        <section id="tab_demandados" class="tab-content">
            <div class="chart-card">{html_demandados}</div>
        </section>

        <section id="tab_stats" class="tab-content">
            <div class="chart-card">{html_stats}</div>
            <div class="chart-card">
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

        <section id="tab_models" class="tab-content">
            <div class="chart-card">{html_models}</div>
        </section>
    </main>

    <script>
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
            window.dispatchEvent(new Event('resize'));
        }}
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
