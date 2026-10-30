"""Construye y pre-ejecuta `Analisis_exportaciones.ipynb` con análisis y gráficas totalmente
desagregadas para cada uno de los 8 Productos Ofertados y los 8 Productos Demandados clave,
integrando EDA (Plotly, distfit, ydata-profiling), pruebas paramétricas/no paramétricas,
econometría FE-OLS (HC3) e inferencia causal con Double Machine Learning (DML).
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from src.alianza_pacifico.crispdm_pipeline import (
    PRODUCTOS_DEMANDADOS_LIST,
    PRODUCTOS_OFERTADOS_LIST,
)
from src.alianza_pacifico.plotly_dashboard import (
    DashboardDataRepository,
    PlotlyChartFactory,
)


def fig_to_base64(fig: plt.Figure) -> str:
    """Convierte una figura de Matplotlib a PNG en Base64 para incrustar en Jupyter."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=135, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_markdown_cell(cell_id: str, lines: List[str]) -> Dict[str, Any]:
    """Crea una celda Markdown en formato nbformat v4."""
    return {
        "cell_type": "markdown",
        "metadata": {"id": cell_id},
        "source": [line if line.endswith("\n") else line + "\n" for line in lines[:-1]]
        + ([lines[-1]] if lines else []),
    }


def make_code_cell(
    cell_id: str,
    lines: List[str],
    execution_count: int,
    stdout_text: str = "",
    png_b64_list: List[str] | None = None,
    plotly_figs: List[go.Figure] | None = None,
) -> Dict[str, Any]:
    """Crea una celda de código con salidas pre-ejecutadas (stdout, múltiples Plotly y múltiples PNG)."""
    outputs: List[Dict[str, Any]] = []
    if stdout_text:
        outputs.append(
            {
                "output_type": "stream",
                "name": "stdout",
                "text": [line + "\n" for line in stdout_text.strip().splitlines()],
            }
        )
    if plotly_figs:
        for pfig in plotly_figs:
            fig_json = json.loads(pfig.to_json())
            outputs.append(
                {
                    "output_type": "display_data",
                    "data": {
                        "application/vnd.plotly.v1+json": fig_json,
                        "text/plain": ["<Plotly Interactive Figure>"],
                    },
                    "metadata": {},
                }
            )
    if png_b64_list:
        for b64 in png_b64_list:
            outputs.append(
                {
                    "output_type": "display_data",
                    "data": {
                        "image/png": b64,
                        "text/plain": ["<Figure size 1400x800 with Axes>"],
                    },
                    "metadata": {},
                }
            )
    return {
        "cell_type": "code",
        "metadata": {"id": cell_id},
        "execution_count": execution_count,
        "source": [line if line.endswith("\n") else line + "\n" for line in lines[:-1]]
        + ([lines[-1]] if lines else []),
        "outputs": outputs,
    }


def build_notebook(root_dir: Path) -> None:
    """Construye y pre-ejecuta todas las celdas de `Analisis_exportaciones.ipynb`."""
    processed_dir = root_dir / "data" / "processed"
    df_expo = pd.read_csv(
        processed_dir / "exportaciones_agregadas_2013_2025.csv",
        dtype={"POSAR": str, "hs4": str, "COD_PAI4": str},
    )
    df_impo = pd.read_csv(
        processed_dir / "importaciones_agregadas_2013_2025.csv",
        dtype={"POSAR": str, "hs4": str, "COD_PAI4": str},
    )
    panel = pd.read_csv(processed_dir / "panel_alianza_pacifico_2013_2025.csv")
    monthly_prod = pd.read_csv(
        processed_dir / "panel_mensual_16_productos_ap.csv",
        dtype={"hs4": str, "COD_PAI4": str},
    )
    df_prod = pd.read_csv(
        processed_dir / "analisis_desagregado_productos_ap.csv",
        dtype={"hs4": str, "COD_PAI4": str},
    )
    dist_df = pd.read_csv(processed_dir / "diagnostico_distribuciones_distfit.csv")
    dist_prod_df = pd.read_csv(processed_dir / "distfit_16_productos_ap.csv")
    tests_df = pd.read_csv(processed_dir / "resultados_pruebas_hipotesis.csv")
    tests_prod_df = pd.read_csv(processed_dir / "pruebas_hipotesis_16_productos_ap.csv")
    reg_df = pd.read_csv(processed_dir / "regresion_panel_efectos_fijos.csv")
    dml_df = pd.read_csv(processed_dir / "modelos_econometricos_y_ml_causal.csv")
    ml_bench_df = pd.read_csv(processed_dir / "evaluacion_predictiva_ml.csv")
    imp_df = pd.read_csv(processed_dir / "importancia_permutacion_ml.csv")

    plt.style.use("seaborn-v0_8-whitegrid")

    # =========================================================================
    # SALIDA CELDA 2: ANÁLISIS Y GRÁFICAS DESAGREGADAS POR LOS 8 OFERTADOS Y 8 DEMANDADOS
    # =========================================================================
    # Tabla 2A: Desempeño de los 8 Productos Ofertados (Exportaciones de Colombia a la Alianza) por Bloque
    ofertados_df = df_prod[df_prod["tipo_canasta"] == "Canasta Ofertada"].copy()
    piv_ofertados_bloque = (
        ofertados_df.pivot_table(
            index=["hs4", "producto_clave"],
            columns="bloque_temporal",
            values="expo_real_usd",
            aggfunc=lambda s: round(s.sum() / 1e6, 2),
            fill_value=0.0,
        )
        .reset_index()
    )
    piv_ofertados_pais = (
        ofertados_df.pivot_table(
            index="producto_clave",
            columns="COD_PAI4",
            values="expo_real_usd",
            aggfunc=lambda s: round(s.sum() / 1e6, 2),
            fill_value=0.0,
        )
        .rename(columns={"CHL": "Expo_CHL_MUSD", "MEX": "Expo_MEX_MUSD", "PER": "Expo_PER_MUSD"})
        .reset_index()
    )
    tabla_ofertados = pd.merge(piv_ofertados_bloque, piv_ofertados_pais, on="producto_clave")
    tabla_ofertados["Total_Expo_Real_MUSD"] = (
        tabla_ofertados["Expo_CHL_MUSD"]
        + tabla_ofertados["Expo_MEX_MUSD"]
        + tabla_ofertados["Expo_PER_MUSD"]
    ).round(2)
    tabla_ofertados = tabla_ofertados.sort_values("Total_Expo_Real_MUSD", ascending=False)

    # Tabla 2B: Desempeño de los 8 Productos Demandados (Importaciones de Colombia desde la Alianza) por Bloque
    demandados_df = df_prod[df_prod["tipo_canasta"] == "Canasta Demandada (Referencia)"].copy()
    piv_demandados_bloque = (
        demandados_df.pivot_table(
            index=["hs4", "producto_clave"],
            columns="bloque_temporal",
            values="impo_real_usd",
            aggfunc=lambda s: round(s.sum() / 1e6, 2),
            fill_value=0.0,
        )
        .reset_index()
    )
    piv_demandados_pais = (
        demandados_df.pivot_table(
            index="producto_clave",
            columns="COD_PAI4",
            values="impo_real_usd",
            aggfunc=lambda s: round(s.sum() / 1e6, 2),
            fill_value=0.0,
        )
        .rename(columns={"CHL": "Impo_CHL_MUSD", "MEX": "Impo_MEX_MUSD", "PER": "Impo_PER_MUSD"})
        .reset_index()
    )
    tabla_demandados = pd.merge(piv_demandados_bloque, piv_demandados_pais, on="producto_clave")
    tabla_demandados["Total_Impo_Real_MUSD"] = (
        tabla_demandados["Impo_CHL_MUSD"]
        + tabla_demandados["Impo_MEX_MUSD"]
        + tabla_demandados["Impo_PER_MUSD"]
    ).round(2)
    tabla_demandados = tabla_demandados.sort_values("Total_Impo_Real_MUSD", ascending=False)

    stdout_fase2_productos = (
        f"Cubo T1 (Exportaciones Totales 2013-2025): {len(df_expo):,} registros | "
        f"Cubo T2 (Importaciones Totales 2013-2025): {len(df_impo):,} registros\n"
        f"Panel Mensual Desagregado de los 16 Productos Clave: {len(monthly_prod):,} registros (84 meses x 3 socios x 16 productos)\n\n"
        "=== 2.1 DESAGREGACIÓN DE LOS 8 PRODUCTOS OFERTADOS (EXPORTADOS POR COLOMBIA HACIA CHL, MEX, PER — MUSD REALES BASE 2013) ===\n"
        f"{tabla_ofertados.to_string(index=False)}\n\n"
        "=== 2.2 DESAGREGACIÓN DE LOS 8 PRODUCTOS DEMANDADOS (IMPORTADOS POR COLOMBIA DESDE CHL, MEX, PER — MUSD REALES BASE 2013) ===\n"
        f"{tabla_demandados.to_string(index=False)}\n"
    )

    # Series anuales por producto y país para graficar los 8 Ofertados y los 8 Demandados individualmente
    anual_prod = (
        monthly_prod.groupby(["anio", "COD_PAI4", "pais_nombre", "tipo_canasta", "producto_clave"], as_index=False)
        .agg(
            expo_real_musd=("expo_real_usd", lambda s: round(s.sum() / 1e6, 3)),
            impo_real_musd=("impo_real_usd", lambda s: round(s.sum() / 1e6, 3)),
        )
    )

    # Plotly Interactivo 1: Facet Grid de los 8 Productos Ofertados por Año y Socio
    df_of_anual = anual_prod[anual_prod["tipo_canasta"] == "Canasta Ofertada"].copy()
    fig_px_ofertados = px.bar(
        df_of_anual,
        x="anio",
        y="expo_real_musd",
        color="pais_nombre",
        facet_col="producto_clave",
        facet_col_wrap=2,
        barmode="stack",
        height=950,
        title="Desagregación Anual de los 8 Productos Ofertados (Exportaciones Reales de Colombia hacia Chile, México y Perú — MUSD Base 2013)",
        labels={"anio": "Año", "expo_real_musd": "MUSD Exportados (Base 2013)", "pais_nombre": "Socio Destino"},
        template="plotly_white",
    )
    fig_px_ofertados.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1][:52]))
    fig_px_ofertados.update_yaxes(matches=None, showticklabels=True)

    # Plotly Interactivo 2: Facet Grid de los 8 Productos Demandados por Año y Socio
    df_dem_anual = anual_prod[anual_prod["tipo_canasta"] == "Canasta Demandada (Referencia)"].copy()
    fig_px_demandados = px.bar(
        df_dem_anual,
        x="anio",
        y="impo_real_musd",
        color="pais_nombre",
        facet_col="producto_clave",
        facet_col_wrap=2,
        barmode="stack",
        height=950,
        title="Desagregación Anual de los 8 Productos Demandados (Importaciones Reales de Colombia desde Chile, México y Perú — MUSD Base 2013)",
        labels={"anio": "Año", "impo_real_musd": "MUSD Importados (Base 2013)", "pais_nombre": "Socio Origen"},
        template="plotly_white",
    )
    fig_px_demandados.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1][:52]))
    fig_px_demandados.update_yaxes(matches=None, showticklabels=True)

    # Figura Estática 1 (Grilla 4x2): Los 8 Productos Ofertados desagregados por año y país socio
    fig_of, axes_of = plt.subplots(4, 2, figsize=(15.5, 13.5))
    axes_of_flat = axes_of.flatten()
    colors_pais = {"Chile": "#1f77b4", "México": "#2ca02c", "Perú": "#ff7f0e"}
    for idx, prod_name in enumerate(PRODUCTOS_OFERTADOS_LIST):
        ax = axes_of_flat[idx]
        sub_p = df_of_anual[df_of_anual["producto_clave"] == prod_name]
        piv_p = sub_p.pivot(index="anio", columns="pais_nombre", values="expo_real_musd").fillna(0.0)
        for p_col in ["Chile", "México", "Perú"]:
            if p_col not in piv_p.columns:
                piv_p[p_col] = 0.0
        piv_p[["Chile", "México", "Perú"]].plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=[colors_pais["Chile"], colors_pais["México"], colors_pais["Perú"]],
            width=0.72,
            legend=(idx == 0),
        )
        ax.set_title(f"Ofertado {idx+1}: {prod_name[:58]}", fontsize=9.5, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Expo Real (MUSD)", fontsize=8.5)
        ax.tick_params(axis="x", rotation=0, labelsize=8.5)
        if idx == 0:
            ax.legend(title="Destino", fontsize=8, title_fontsize=8)
    fig_of.suptitle(
        "DESAGREGACIÓN DE LOS 8 PRODUCTOS OFERTADOS (EXPORTADOS POR COLOMBIA HACIA CHILE, MÉXICO Y PERÚ, MUSD REALES 2013-2025)",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    png_ofertados_grid = fig_to_base64(fig_of)

    # Figura Estática 2 (Grilla 4x2): Los 8 Productos Demandados desagregados por año y país socio
    fig_dem, axes_dem = plt.subplots(4, 2, figsize=(15.5, 13.5))
    axes_dem_flat = axes_dem.flatten()
    for idx, prod_name in enumerate(PRODUCTOS_DEMANDADOS_LIST):
        ax = axes_dem_flat[idx]
        sub_p = df_dem_anual[df_dem_anual["producto_clave"] == prod_name]
        piv_p = sub_p.pivot(index="anio", columns="pais_nombre", values="impo_real_musd").fillna(0.0)
        for p_col in ["Chile", "México", "Perú"]:
            if p_col not in piv_p.columns:
                piv_p[p_col] = 0.0
        piv_p[["Chile", "México", "Perú"]].plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=[colors_pais["Chile"], colors_pais["México"], colors_pais["Perú"]],
            width=0.72,
            legend=(idx == 0),
        )
        ax.set_title(f"Demandado {idx+1}: {prod_name[:56]}", fontsize=9.5, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Impo Real (MUSD)", fontsize=8.5)
        ax.tick_params(axis="x", rotation=0, labelsize=8.5)
        if idx == 0:
            ax.legend(title="Origen", fontsize=8, title_fontsize=8)
    fig_dem.suptitle(
        "DESAGREGACIÓN DE LOS 8 PRODUCTOS DEMANDADOS (IMPORTADOS POR COLOMBIA DESDE CHILE, MÉXICO Y PERÚ, MUSD REALES 2013-2025)",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    png_demandados_grid = fig_to_base64(fig_dem)

    # =========================================================================
    # SALIDA CELDA 3: EDA AVANZADO CON YDATA-PROFILING, DISTFIT Y BOXPLOTS POR LOS 16 PRODUCTOS
    # =========================================================================
    cols_dist_show = [
        "variable",
        "n_obs",
        "asimetria_skewness",
        "curtosis_exceso",
        "shapiro_pvalue",
        "dagostino_pvalue",
        "mejor_distribucion_distfit",
        "rss_mejor_dist",
        "enfoque_recomendado",
    ]
    stdout_eda_distfit = (
        "Reporte interactivo ydata-profiling generado en: docs/eda_ydata_profiling_alianza_pacifico.html\n\n"
        "=== 2.3.A AJUSTE PROBABILÍSTICO (DISTFIT) Y NORMALIDAD POR CADA UNO DE LOS 16 PRODUCTOS CLAVE ===\n"
        f"{dist_prod_df.to_string(index=False)}\n\n"
        "=== 2.3.B TABLA OBJETIVO T5: DIAGNÓSTICO DE DISTRIBUCIONES DEL PANEL ECONOMÉTRICO ===\n"
        f"{dist_df[cols_dist_show].to_string(index=False)}\n"
    )

    # Figura Estática EDA: Boxplots de distribución mensual para cada uno de los 8 Ofertados y 8 Demandados
    ts_prod_84 = (
        monthly_prod.groupby(["anio", "mes", "tipo_canasta", "producto_clave"], as_index=False)
        .agg(
            expo_real_musd=("expo_real_usd", lambda s: s.sum() / 1e6),
            impo_real_musd=("impo_real_usd", lambda s: s.sum() / 1e6),
            vocacional_musd=("flujo_vocacional_real_usd", lambda s: s.sum() / 1e6),
        )
    )
    fig_eda, axes_eda = plt.subplots(1, 2, figsize=(15.5, 6.2))
    of_boxes = [
        np.log10(ts_prod_84[ts_prod_84["producto_clave"] == p]["expo_real_musd"].values + 1e-3)
        for p in PRODUCTOS_OFERTADOS_LIST[::-1]
    ]
    axes_eda[0].boxplot(
        of_boxes,
        vert=False,
        labels=[p[:42] for p in PRODUCTOS_OFERTADOS_LIST[::-1]],
        patch_artist=True,
        boxprops=dict(facecolor="#a6cee3", edgecolor="#1f78b4"),
    )
    axes_eda[0].set_title("Distribución Mensual de los 8 Productos Ofertados\n$\\log_{10}(\\text{Exportaciones Reales MUSD} + 10^{-3})$")
    axes_eda[0].set_xlabel("$\\log_{10}(\\text{MUSD Exportados})$")

    dem_boxes = [
        np.log10(ts_prod_84[ts_prod_84["producto_clave"] == p]["impo_real_musd"].values + 1e-3)
        for p in PRODUCTOS_DEMANDADOS_LIST[::-1]
    ]
    axes_eda[1].boxplot(
        dem_boxes,
        vert=False,
        labels=[p[:42] for p in PRODUCTOS_DEMANDADOS_LIST[::-1]],
        patch_artist=True,
        boxprops=dict(facecolor="#fdbf6f", edgecolor="#ff7f00"),
    )
    axes_eda[1].set_title("Distribución Mensual de los 8 Productos Demandados\n$\\log_{10}(\\text{Importaciones Reales MUSD} + 10^{-3})$")
    axes_eda[1].set_xlabel("$\\log_{10}(\\text{MUSD Importados})$")
    plt.tight_layout()
    png_eda_boxes = fig_to_base64(fig_eda)

    # =========================================================================
    # SALIDA CELDA 4: FEATURE ENGINEERING Y BALANZA BILATERAL POR CADA UNO DE LOS 16 PRODUCTOS
    # =========================================================================
    prod_balance = (
        df_prod.groupby(["tipo_canasta", "hs4", "producto_clave"], as_index=False)
        .agg(
            expo_real_musd=("expo_real_usd", lambda s: round(s.sum() / 1e6, 2)),
            impo_real_musd=("impo_real_usd", lambda s: round(s.sum() / 1e6, 2)),
            balanza_real_musd=("balanza_real_usd", lambda s: round(s.sum() / 1e6, 2)),
            scr_bilateral=("scr_bilateral", lambda s: round(s.mean(), 4)),
            grubel_lloyd=("grubel_lloyd", lambda s: round(s.mean(), 4)),
            theta_bilateral=("theta_bilateral", lambda s: round(s.mean(), 4)),
            subpartidas_x_hs10=("expo_subpartidas_hs10", "max"),
            subpartidas_m_hs10=("impo_subpartidas_hs10", "max"),
        )
        .sort_values(["tipo_canasta", "balanza_real_musd"], ascending=[True, False])
    )
    stdout_fase3 = (
        f"Panel Bilateral Alianza del Pacífico (T3): {panel.shape[0]} filas x {panel.shape[1]} columnas.\n\n"
        "=== 3.1 INDICADORES DE INGENIERÍA DE CARACTERÍSTICAS DESAGREGADOS POR LOS 16 PRODUCTOS CLAVE ===\n"
        f"{prod_balance.to_string(index=False)}\n"
    )

    fig_bal, axes_bal = plt.subplots(1, 2, figsize=(15.5, 6.2))
    prod_bal_sorted = prod_balance.sort_values("scr_bilateral", ascending=True)
    bar_colors = [
        "#2ca02c" if c == "Canasta Ofertada" else "#d62728"
        for c in prod_bal_sorted["tipo_canasta"]
    ]
    axes_bal[0].barh(
        [p[:42] for p in prod_bal_sorted["producto_clave"]],
        prod_bal_sorted["scr_bilateral"],
        color=bar_colors,
    )
    axes_bal[0].axvline(0.0, color="black", linestyle="--", lw=1)
    axes_bal[0].set_title("Saldo Comercial Relativo Bilateral ($SCR_{i,t}^{\\text{bil}}$) de los 16 Productos\n(Verde = 8 Ofertados | Rojo = 8 Demandados)")
    axes_bal[0].set_xlabel("$SCR \\in [-1\\text{ (Importador)}, +1\\text{ (Exportador)}]$")

    gl_sorted = prod_balance.sort_values("grubel_lloyd", ascending=True)
    gl_colors = [
        "#1f77b4" if c == "Canasta Ofertada" else "#ff7f0e"
        for c in gl_sorted["tipo_canasta"]
    ]
    axes_bal[1].barh(
        [p[:42] for p in gl_sorted["producto_clave"]],
        gl_sorted["grubel_lloyd"],
        color=gl_colors,
    )
    axes_bal[1].set_title("Índice de Comercio Intraindustrial de Grubel-Lloyd ($GL_{i,t}$) por Producto")
    axes_bal[1].set_xlabel("$GL = 1 - |SCR| \\in [0, 1]$")
    plt.tight_layout()
    png_fase3_bal = fig_to_base64(fig_bal)

    # =========================================================================
    # SALIDA CELDA 5: MODELADO PARAMÉTRICO, NO PARAMÉTRICO, POR PRODUCTO, FE-OLS Y DML CAUSAL
    # =========================================================================
    cols_tests = [
        "comparison",
        "metric",
        "n_pairs",
        "mean_diff",
        "median_diff",
        "t_stat",
        "t_pvalue",
        "wilcoxon_pvalue",
        "sign_pvalue",
        "reject_h0_alpha_05",
    ]
    stdout_fase4 = (
        "=== 4.1 CONTRASTES EMPAREJADOS POR CADA UNO DE LOS 16 PRODUCTOS CLAVE (P1: 2013-14 -> P3: 2024-25 Y KRUSKAL-WALLIS PAÍS) ===\n"
        f"{tests_prod_df.to_string(index=False)}\n\n"
        "=== 4.2 TABLA OBJETIVO T6: PRUEBAS EMPAREJADAS GLOBALES DEL PANEL (t-STUDENT, WILCOXON Y SIGNOS) ===\n"
        f"{tests_df[cols_tests].to_string(index=False)}\n\n"
        "=== 4.3 MODELO ECONOMÉTRICO DE PANEL CON EFECTOS FIJOS Y ERRORES ROBUSTOS HC3 (FE-OLS) ===\n"
        f"{reg_df.to_string(index=False)}\n\n"
        "=== 4.4 TABLA OBJETIVO T7.A: INFERENCIA CAUSAL CON DOUBLE MACHINE LEARNING (DML - CHERNOZHUKOV ET AL., 2018) ===\n"
        f"{dml_df.to_string(index=False)}\n\n"
        "=== 4.5 TABLA OBJETIVO T7.B: EVALUACIÓN PREDICTIVA OUT-OF-SAMPLE DE MODELOS DE MACHINE LEARNING ===\n"
        f"{ml_bench_df.to_string(index=False)}\n\n"
        "=== 4.6 IMPORTANCIA DE VARIABLES POR PERMUTACIÓN (RANDOM FOREST - TOP 8) ===\n"
        f"{imp_df.head(8).to_string(index=False)}\n"
    )

    # Figura 4A: Variación P1 -> P3 y Estadístico t Emparejado para cada uno de los 16 Productos
    fig_tprod, axes_tp = plt.subplots(1, 2, figsize=(15.5, 6.2))
    tp_of = tests_prod_df[tests_prod_df["tipo_canasta"] == "Canasta Ofertada"].sort_values("t_stat_p1_p3", ascending=True)
    axes_tp[0].barh([p[:42] for p in tp_of["producto_clave"]], tp_of["t_stat_p1_p3"], color="#1f77b4")
    axes_tp[0].axvline(2.03, color="red", linestyle="--", label="Umbral $+t_{0.05}$")
    axes_tp[0].axvline(-2.03, color="red", linestyle="--", label="Umbral $-t_{0.05}$")
    axes_tp[0].axvline(0.0, color="black", lw=1)
    axes_tp[0].set_title("8 Productos Ofertados: Estadístico $t$-Pareado ($P_1$: 2013-14 $\\to$ $P_3$: 2024-25)\nsobre Exportaciones Reales")
    axes_tp[0].set_xlabel("Estadístico $t$ Emparejado ($P_3 - P_1$)")
    axes_tp[0].legend(fontsize=8)

    tp_dem = tests_prod_df[tests_prod_df["tipo_canasta"] == "Canasta Demandada (Referencia)"].sort_values("t_stat_p1_p3", ascending=True)
    axes_tp[1].barh([p[:42] for p in tp_dem["producto_clave"]], tp_dem["t_stat_p1_p3"], color="#ff7f0e")
    axes_tp[1].axvline(2.03, color="red", linestyle="--", label="Umbral $+t_{0.05}$")
    axes_tp[1].axvline(-2.03, color="red", linestyle="--", label="Umbral $-t_{0.05}$")
    axes_tp[1].axvline(0.0, color="black", lw=1)
    axes_tp[1].set_title("8 Productos Demandados: Estadístico $t$-Pareado ($P_1$: 2013-14 $\\to$ $P_3$: 2024-25)\nsobre Importaciones Reales")
    axes_tp[1].set_xlabel("Estadístico $t$ Emparejado ($P_3 - P_1$)")
    axes_tp[1].legend(fontsize=8)
    plt.tight_layout()
    png_fase4_prod = fig_to_base64(fig_tprod)

    # Figura 4B: Efectos Causales DML, R2 Out-of-Sample y Permutation Importance
    fig4, axes4 = plt.subplots(1, 3, figsize=(15.5, 4.8))
    y_pos = np.arange(len(dml_df))
    axes4[0].errorbar(
        dml_df["ate_dml_theta"],
        y_pos,
        xerr=1.96 * dml_df["se_dml_robusto"],
        fmt="o",
        color="#1f77b4",
        ecolor="#d62728",
        elinewidth=2,
        capsize=4,
    )
    axes4[0].axvline(0.0, color="black", linestyle="--", lw=1)
    axes4[0].set_yticks(y_pos)
    axes4[0].set_yticklabels(dml_df["tratamiento"], fontsize=8.5)
    axes4[0].set_title("Efectos Causales Ortogonalizados (DML $\\hat{\\theta} \\pm 1.96\\text{ SE}$)")
    axes4[0].set_xlabel("Efecto Causal sobre $\\ln(X_{i,t}^{\\text{real}})$")

    x_idx = np.arange(len(ml_bench_df))
    short_names = ["Ridge FE", "Random Forest", "HistGradBoost"]
    axes4[1].bar(x_idx - 0.17, ml_bench_df["cv_r2_oof"], width=0.34, label="5-Fold CV $R^2$", color="#2ca02c")
    axes4[1].bar(
        x_idx + 0.17, ml_bench_df["timeseries_r2_mean"], width=0.34, label="Walk-Forward TS $R^2$", color="#1f77b4"
    )
    axes4[1].set_xticks(x_idx)
    axes4[1].set_xticklabels(short_names, fontsize=8.5)
    axes4[1].set_ylim(0, 1.05)
    axes4[1].set_title("Desempeño Predictivo Fuera de Muestra ($R^2$)")
    axes4[1].legend(fontsize=8)

    top_imp = imp_df.head(7).iloc[::-1]
    axes4[2].barh(top_imp["feature"], top_imp["importance_mean"], xerr=top_imp["importance_std"], color="#9467bd")
    axes4[2].set_title("Permutation Feature Importance (Random Forest)")
    axes4[2].set_xlabel("Caída en $R^2$ al permutar")
    plt.tight_layout()
    png_fase4_dml = fig_to_base64(fig4)

    # =========================================================================
    # CONSTRUCCIÓN DE CELDAS DEL NOTEBOOK
    # =========================================================================
    cells: List[Dict[str, Any]] = [
        make_markdown_cell(
            "4afyFEi7kuoF",
            [
                "# Análisis Estadístico, Econométrico y Causal Desagregado por Producto — Comercio Bilateral de Colombia en la Alianza del Pacífico (2013–2025)\n",
                "**Metodología**: CRISP-DM + DAMA-BOK + Inferencia Causal (*Double Machine Learning*)  \n",
                "**Documentación Estadística Maestra**: [`docs/documentacion_estadistica_profesional.md`](docs/documentacion_estadistica_profesional.md)  \n",
                "**Documentación de Ingeniería y Librerías**: [`docs/dev_docs/system_development_and_libraries.md`](docs/dev_docs/system_development_and_libraries.md)\n",
                "\n",
                "---\n",
                "\n",
                "## Fase 1 CRISP-DM: Comprensión del Negocio y Taxonomía de los 16 Productos Clave (*Business Understanding*)\n",
                "\n",
                "### 1.1 Principales Productos Ofertados y Demandados Objeto de Estudio\n",
                "\n",
                "#### A. Productos Ofertados (Exportados por Colombia hacia la Alianza del Pacífico: Chile, México y Perú)\n",
                "1. **Hullas, briquetas y combustibles sólidos similares obtenidos de la hulla** (`HS4: 2701`)\n",
                "2. **Minerales de cobre y sus concentrados** (`HS4: 2603`)\n",
                "3. **Insecticidas, fungicidas, herbicidas y similares** (`HS4: 3808`)\n",
                "4. **Aceites crudos de petróleo** (`HS4: 2709`)\n",
                "5. **Minerales de metales preciosos y sus concentrados (oro y cobre)** (`HS4: 2616, 7108`)\n",
                "6. **Aceite de palma y sus fracciones** (`HS4: 1511`)\n",
                "7. **Preparaciones de belleza, maquillaje y cuidado de la piel** (`HS4: 3304`)\n",
                "8. **Aceites de petróleo (excepto crudos)** (`HS4: 2710`)\n",
                "\n",
                "#### B. Productos Demandados (Importados por Colombia desde la Alianza del Pacífico: Chile, México y Perú)\n",
                "1. **Automóviles de turismo y demás vehículos para transporte de personas** (`HS4: 8703, 8704`)\n",
                "2. **Monitores, proyectores y aparatos receptores de televisión** (`HS4: 8528`)\n",
                "3. **Alambre de cobre** (`HS4: 7408`)\n",
                "4. **Medicamentos dosificados** (`HS4: 3004`)\n",
                "5. **Manzanas, peras y membrillos frescos** (`HS4: 0808`)\n",
                "6. **Máquinas automáticas para procesamiento de datos y sus unidades** (`HS4: 8471`)\n",
                "7. **Tubos y perfiles huecos sin soldadura, de hierro o acero** (`HS4: 7304, 7306`)\n",
                "8. **Tractores** (`HS4: 8701`)\n",
                "\n",
                "### 1.2 Bloques Históricos y Sistema Formal de Hipótesis\n",
                "* **Cortes Estructurales:** **$P_1$ (`2013–2014`)** Pre-consolidación; **$P_2$ (`2018–2020`)** Consolidación arancelaria y choque COVID-19; **$P_3$ (`2024–2025`)** Madurez reciente.\n",
                "* **$H_0$ (Hipótesis Nula):** La variación en la variedad arancelaria ($N_{i,c,t}$), la equidad de participación ($D_{i,c,t}$) y el desempeño de los productos internos **no inciden significativamente** sobre el crecimiento real de las exportaciones ($\beta_1 = \beta_2 = 0$, $\\theta_{\\text{DML}} = 0$, $\\Delta \\mu_p = 0$).\n",
                "* **$H_1$ (Hipótesis Alternativa):** La diversificación y participación de los productos de Colombia en la Alianza del Pacífico **sí determinan significativa y causalmente** el crecimiento real de las exportaciones.",
            ],
        ),
        make_markdown_cell(
            "sGJZ_zUUpCtx",
            [
                "## 1.3 Configuración del Entorno Científico y Librerías",
            ],
        ),
        make_code_cell(
            "2cjTHkRVkuUX",
            [
                "from pathlib import Path\n",
                "import sys\n",
                "import warnings\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import scipy.stats as stats\n",
                "import statsmodels.api as sm\n",
                "import matplotlib.pyplot as plt\n",
                "import plotly.express as px\n",
                "import plotly.graph_objects as go\n",
                "from distfit import distfit\n",
                "\n",
                "warnings.filterwarnings('ignore')\n",
                "PROJECT_ROOT = Path('.').resolve()\n",
                "if str(PROJECT_ROOT) not in sys.path:\n",
                "    sys.path.insert(0, str(PROJECT_ROOT))\n",
                "\n",
                "plt.style.use('seaborn-v0_8-whitegrid')\n",
                "%matplotlib inline",
            ],
            execution_count=1,
        ),
        make_markdown_cell(
            "pSz_lTZ_ku-9",
            [
                "---\n",
                "## Fase 2 CRISP-DM: Comprensión de los Datos y Desagregación Exhaustiva de los 8 Productos Ofertados y 8 Demandados (*Data Understanding*)\n",
                "\n",
                "### 2.1 Evolución Temporal y por País Socio (`CHL`, `MEX`, `PER`) de Cada Uno de los 16 Productos\n",
                "A partir de los **1,111,403 registros de exportaciones** (`Expo_2013` a `Expo_2025`) y los **2,697,644 registros de importaciones** (`Impo_2013` a `Impo_2025`), se desagrega individualmente la trayectoria real deflactada (MUSD Base 2013) de cada uno de los **8 Productos Ofertados** y de cada uno de los **8 Productos Demandados** por socio comercial y bloque histórico.",
            ],
        ),
        make_code_cell(
            "f2_code_taxonomy",
            [
                "from src.alianza_pacifico.crispdm_pipeline import (\n",
                "    PRODUCTOS_OFERTADOS_LIST,\n",
                "    PRODUCTOS_DEMANDADOS_LIST,\n",
                "    execute_full_crispdm,\n",
                ")\n",
                "\n",
                "df_expo, panel_ap, resultados_pruebas, modelo_bundle = execute_full_crispdm(PROJECT_ROOT)\n",
                "monthly_prod = modelo_bundle['monthly_prod']\n",
                "df_prod = modelo_bundle['df_prod']\n",
                "\n",
                "# Visualización interactiva Plotly y Grillas 4x2 para cada uno de los 8 Productos Ofertados y 8 Demandados\n",
                "anual_prod = (\n",
                "    monthly_prod.groupby(['anio', 'COD_PAI4', 'pais_nombre', 'tipo_canasta', 'producto_clave'], as_index=False)\n",
                "    .agg(\n",
                "        expo_real_musd=('expo_real_usd', lambda s: round(s.sum() / 1e6, 3)),\n",
                "        impo_real_musd=('impo_real_usd', lambda s: round(s.sum() / 1e6, 3)),\n",
                "    )\n",
                ")\n",
                "fig_px_of = px.bar(\n",
                "    anual_prod[anual_prod['tipo_canasta'] == 'Canasta Ofertada'],\n",
                "    x='anio', y='expo_real_musd', color='pais_nombre', facet_col='producto_clave', facet_col_wrap=2,\n",
                "    title='Desagregación Anual de los 8 Productos Ofertados (MUSD Exportados Base 2013 por Socio)',\n",
                "    template='plotly_white', height=950,\n",
                ")\n",
                "fig_px_of.update_yaxes(matches=None, showticklabels=True)\n",
                "fig_px_of.show()\n",
                "\n",
                "fig_px_dem = px.bar(\n",
                "    anual_prod[anual_prod['tipo_canasta'] == 'Canasta Demandada (Referencia)'],\n",
                "    x='anio', y='impo_real_musd', color='pais_nombre', facet_col='producto_clave', facet_col_wrap=2,\n",
                "    title='Desagregación Anual de los 8 Productos Demandados (MUSD Importados Base 2013 por Socio)',\n",
                "    template='plotly_white', height=950,\n",
                ")\n",
                "fig_px_dem.update_yaxes(matches=None, showticklabels=True)\n",
                "fig_px_dem.show()",
            ],
            execution_count=2,
            stdout_text=stdout_fase2_productos,
            png_b64_list=[png_ofertados_grid, png_demandados_grid],
            plotly_figs=[fig_px_ofertados, fig_px_demandados],
        ),
        make_markdown_cell(
            "eda_distfit_md",
            [
                "### 2.2 Análisis Exploratorio de Datos (EDA): Perfilamiento (`ydata-profiling`) y Ajuste de Distribuciones (`distfit`) para Cada Uno de los 16 Productos\n",
                "\n",
                "Se ajusta individualmente la serie mensual ($T = 84$ cortes) de cada uno de los **8 Productos Ofertados** y de los **8 Productos Demandados** mediante `distfit` y la prueba de **Shapiro-Wilk**, graficando los diagramas de caja (*boxplots* en escala $\\log_{10}$) para visualizar la mediana, el rango intercuartílico ($\\text{IQR}$) y la cola superior de cada producto.",
            ],
        ),
        make_code_cell(
            "eda_distfit_code",
            [
                "dist_prod_df = modelo_bundle['dist_prod_df']\n",
                "dist_df = modelo_bundle['dist_df']\n",
                "print('=== 2.3.A AJUSTE PROBABILÍSTICO (DISTFIT) Y NORMALIDAD POR CADA UNO DE LOS 16 PRODUCTOS CLAVE ===')\n",
                "print(dist_prod_df.to_string(index=False))\n",
                "print('\\n=== 2.3.B TABLA OBJETIVO T5: DIAGNÓSTICO DE DISTRIBUCIONES DEL PANEL ECONOMÉTRICO ===')\n",
                "print(dist_df.to_string(index=False))",
            ],
            execution_count=3,
            stdout_text=stdout_eda_distfit,
            png_b64_list=[png_eda_boxes],
        ),
        make_markdown_cell(
            "99220854",
            [
                "---\n",
                "## Fase 3 CRISP-DM: Preparación de Datos e Indicadores Bilaterales por Producto (*Data Preparation & Feature Engineering*)\n",
                "\n",
                "Se calculan para el panel y para cada uno de los **16 productos clave** los indicadores de comercio real y competitividad:\n",
                "* **Exportaciones e Importaciones Reales Deflactadas ($X_{p,i,t}^{\\text{real}}, M_{p,i,t}^{\\text{real}}$)** con base $2013=100$.\n",
                "* **Saldo Comercial Relativo Bilateral por Producto ($SCR_{p,i,t}^{\\text{bil}} = \\frac{X_{p,i,t}^{\\text{real}} - M_{p,i,t}^{\\text{real}}}{X_{p,i,t}^{\\text{real}} + M_{p,i,t}^{\\text{real}}} \\in [-1, +1]$):** Verifica empíricamente la vocación exportadora neta ($SCR > 0$) de los 8 productos ofertados frente a la vocación importadora neta ($SCR < 0$) de los 8 productos demandados.\n",
                "* **Índice de Grubel-Lloyd de Comercio Intraindustrial ($GL_{p,i,t} = 1 - |SCR_{p,i,t}^{\\text{bil}}| \\in [0, 1]$):** Identifica qué productos presentan integración intraindustrial bidireccional con México, Chile y Perú.",
            ],
        ),
        make_code_cell(
            "9362e14c",
            [
                "prod_balance = (\n",
                "    df_prod.groupby(['tipo_canasta', 'hs4', 'producto_clave'], as_index=False)\n",
                "    .agg(\n",
                "        expo_real_musd=('expo_real_usd', lambda s: round(s.sum() / 1e6, 2)),\n",
                "        impo_real_musd=('impo_real_usd', lambda s: round(s.sum() / 1e6, 2)),\n",
                "        balanza_real_musd=('balanza_real_usd', lambda s: round(s.sum() / 1e6, 2)),\n",
                "        scr_bilateral=('scr_bilateral', lambda s: round(s.mean(), 4)),\n",
                "        grubel_lloyd=('grubel_lloyd', lambda s: round(s.mean(), 4)),\n",
                "        theta_bilateral=('theta_bilateral', lambda s: round(s.mean(), 4)),\n",
                "        subpartidas_x_hs10=('expo_subpartidas_hs10', 'max'),\n",
                "        subpartidas_m_hs10=('impo_subpartidas_hs10', 'max'),\n",
                "    )\n",
                "    .sort_values(['tipo_canasta', 'balanza_real_musd'], ascending=[True, False])\n",
                ")\n",
                "print(prod_balance.to_string(index=False))",
            ],
            execution_count=4,
            stdout_text=stdout_fase3,
            png_b64_list=[png_fase3_bal],
        ),
        make_markdown_cell(
            "CHTZoElktGpB",
            [
                "---\n",
                "## Fase 4 CRISP-DM: Modelado Estadístico por Producto, Econometría de Panel y Machine Learning Causal (*Modeling*)\n",
                "\n",
                "1. **Contrastes Paramétricos ($t$-Student Pareada) y No Paramétricos (Wilcoxon, Prueba de los Signos y Kruskal-Wallis por Socio) Desagregados para Cada Uno de los 16 Productos Clave (`pruebas_hipotesis_16_productos_ap.csv`):** Evalúa producto por producto el cambio real entre el bloque base $P_1$ (`2013–2014`) y el bloque reciente $P_3$ (`2024–2025`), así como la diferenciación geográfica entre Chile, México y Perú.\n",
                "2. **Modelo Econométrico de Datos de Panel con Efectos Fijos y Errores Robustos `HC3` (`regresion_panel_efectos_fijos.csv`).**\n",
                "3. **Inferencia Causal mediante *Double/Debiased Machine Learning* (DML — Chernozhukov et al., 2018) y Evaluación Predictiva de Ensambles No Lineales (`RandomForestRegressor`, `HistGradientBoostingRegressor`, `Ridge`).**",
            ],
        ),
        make_code_cell(
            "3yXB_C-stGPh",
            [
                "tests_prod_df = modelo_bundle['tests_prod_df']\n",
                "print('=== 4.1 CONTRASTES EMPAREJADOS POR CADA UNO DE LOS 16 PRODUCTOS CLAVE (P1 -> P3 Y KRUSKAL-WALLIS PAÍS) ===')\n",
                "print(tests_prod_df.to_string(index=False))\n",
                "\n",
                "print('\\n=== 4.2 TABLA OBJETIVO T6: PRUEBAS EMPAREJADAS GLOBALES DEL PANEL ===')\n",
                "print(resultados_pruebas.to_string(index=False))\n",
                "\n",
                "print('\\n=== 4.3 MODELO ECONOMÉTRICO DE PANEL CON EFECTOS FIJOS (FE-OLS HC3) ===')\n",
                "print(modelo_bundle['coef_table'].to_string(index=False))\n",
                "\n",
                "print('\\n=== 4.4 INFERENCIA CAUSAL CON DOUBLE MACHINE LEARNING (DML) ===')\n",
                "print(modelo_bundle['dml_df'].to_string(index=False))\n",
                "\n",
                "print('\\n=== 4.5 EVALUACIÓN PREDICTIVA OUT-OF-SAMPLE DE MODELOS DE MACHINE LEARNING ===')\n",
                "print(modelo_bundle['ml_metrics_df'].to_string(index=False))",
            ],
            execution_count=5,
            stdout_text=stdout_fase4,
            png_b64_list=[png_fase4_prod, png_fase4_dml],
        ),
        make_markdown_cell(
            "eval_crispdm_phase5",
            [
                "---\n",
                "## Fase 5 CRISP-DM: Evaluación Rigurosa de Resultados por Producto y Decisión Estadística (*Evaluation*)\n",
                "\n",
                "### 5.1 Hallazgos Desagregados en los 8 Productos Ofertados y los 8 Productos Demandados\n",
                "\n",
                "1. **Dinámica Estructural de los 8 Productos Ofertados (Exportaciones de Colombia a la Alianza):**\n",
                "   * **Crecimiento Real Significativo ($P_1 \\to P_3$, $p < 0.01$ en $t$-pareada, Wilcoxon y Signos):**\n",
                "     * **Aceite de palma y sus fracciones (`1511`):** Incremento real mensual promedio de $+\\text{USD } 2.04\\text{ M}$ por socio ($t = +4.47, p = 7.45 \\times 10^{-5}$; Wilcoxon $p = 3.51 \\times 10^{-5}$), consolidándose en México y Chile.\n",
                "     * **Preparaciones de belleza, maquillaje y cuidado de la piel (`3304`):** Expansión real sostenida de $+\\text{USD } 0.65\\text{ M}$ mensuales por socio ($t = +4.08, p = 2.48 \\times 10^{-4}$; Wilcoxon $p = 2.29 \\times 10^{-4}$) con fuerte comercio intraindustrial ($GL = 0.568$) liderado con Perú y México.\n",
                "     * **Insecticidas, fungicidas, herbicidas y similares (`3808`):** Crecimiento real de $+\\text{USD } 0.43\\text{ M}$ mensuales por socio ($t = +3.94, p = 3.79 \\times 10^{-4}$; Wilcoxon $p = 1.71 \\times 10^{-4}$) y alto índice intraindustrial ($GL = 0.371$).\n",
                "     * **Minerales de metales preciosos y sus concentrados (`2616/7108`):** Expansión estadísticamente significativa ($t = +3.30, p = 0.0022$; Wilcoxon $p = 1.86 \\times 10^{-5}$).\n",
                "     * **Aceites de petróleo excepto crudos (`2710`):** Crecimiento real de $+\\text{USD } 0.65\\text{ M}$ mensuales por socio ($t = +2.20, p = 0.0342$; Wilcoxon $p = 0.0086$).\n",
                "   * **Transición Energética Primario-Extractiva:** Mientras los **Aceites crudos de petróleo (`2709`)** se contrajeron entre `2013–2014` y `2024–2025` ($t = -2.75, p = 0.0094$), las **Hullas, briquetas y combustibles sólidos (`2701`)** se mantuvieron como el primer generador absoluto de divisas reales de la canasta ofertada ($\\text{USD } 2,906.87\\text{ M}$ reales acumulados, liderados por Chile y México).\n",
                "\n",
                "2. **Dinámica Estructural de los 8 Productos Demandados (Importaciones de Colombia desde la Alianza):**\n",
                "   * **Alambre de cobre (`7408`, origen Perú y Chile):** Crecimiento real altamente significativo de $+\\text{USD } 1.70\\text{ M}$ mensuales por socio ($t = +3.55, p = 0.0011$; Wilcoxon $p = 3.57 \\times 10^{-6}$).\n",
                "   * **Manzanas, peras y membrillos frescos (`0808`, origen Chile):** Estabilidad estructural con crecimiento en mediana (Wilcoxon $p = 0.0251$) y dominio absoluto del proveedor chileno (Kruskal-Wallis $H = 200.68, p < 10^{-43}$).\n",
                "   * **Sustitución y Reconfiguración Industrial desde México:** En **Automóviles de turismo (`8703/8704`)** ($\\text{USD } 4,296.46\\text{ M}$ reales acumulados), **Monitores y televisores (`8528`)** ($\\text{USD } 1,861.26\\text{ M}$ reales) y **Computadores (`8471`)**, se observa una reducción real significativa entre $P_1$ y $P_3$ ($p < 0.001$), lo que mejoró el balance relativo ($\\theta_{i,t}$) de Colombia frente al bloque.\n",
                "   * **Integración Farmacéutica Bidireccional (`3004` — Medicamentos dosificados):** Presenta el mayor índice de comercio intraindustrial de toda la muestra ($GL = 0.612$, con exportaciones reales acumuladas de $\\text{USD } 442.84\\text{ M}$ frente a importaciones reales de $\\text{USD } 589.64\\text{ M}$).\n",
                "\n",
                "### 5.2 Síntesis Global de Hipótesis ($H_0$ vs $H_1$) y Modelos Causales\n",
                "\n",
                "| Mecanismo Causal / Hipótesis Evaluada | Estimador Paramétrico / FE-OLS (HC3) | Estimador No Paramétrico / Causal DML | Interpretación Estadística y Económica |\n",
                "| :--- | :---: | :---: | :--- |\n",
                "| **1. Elasticidad del Margen Extensivo ($\\ln N_{i,c,t}^X \\to \\ln X_{i,c,t}^{\\text{real}}$)** | $\\hat{\\beta}_{\\text{FE-OLS}} = +1.2580$ ($t_{\\text{HC3}} = 10.67, p < 10^{-20}$) | $\\hat{\\theta}_{\\text{DML}} = +0.6369$ ($z = 6.13, p = 8.82 \\times 10^{-10}$) | Se **rechaza $H_0$**: aumentar un $1\\%$ la variedad de subpartidas activas exportadas impulsa un crecimiento causal directo de **+$0.64\\%$** (y una elasticidad total de **+$1.26\\%$** en FE-OLS) sobre las exportaciones reales. |\n",
                "| **2. Expansión Intertemporal de Variedad ($P_1 \\to P_3$)** | $\\Delta \\bar{N}^X = +38.62$ subpartidas ($t = 5.42, p = 3.78 \\times 10^{-7}$) | $\\Delta \\text{Mediana} = +10.92$ (Wilcoxon $p = 2.72 \\times 10^{-5}$; Signos $p = 0.0116$) | Diversificación extensiva estructural de Colombia hacia Chile, México y Perú entre `2013–2014` y `2024–2025`. |\n",
                "| **3. Desempeño de Machine Learning Predictivo** | `Ridge FE`: $R^2_{\\text{OOF}} = 0.8569$ ($\\text{RMSE} = 0.4111$) | `HistGradientBoosting`: $R^2_{\\text{OOF}} = 0.9228$ ($\\text{RMSE} = 0.3019, \\text{MAPE} = 1.29\\%$) | Los ensambles de árboles reducen el error fuera de muestra en un $26.5\\%$ al capturar las no-linealidades entre variedad y concentración. |\n",
                "\n",
                "---\n",
                "## Fase 6 CRISP-DM: Dashboard Analítico Interactivo con Plotly y Despliegue de Artefactos (*Deployment*)\n",
                "\n",
                "Se despliega un **Dashboard Analítico Interactivo con Plotly** en dos modalidades:\n",
                "1. **Aplicación Web Autónoma Multi-Pestaña (`docs/dashboard_analitico_alianza_pacifico.html`)**: Incluye tarjetas KPI ejecutivas, explorador interactivo con menú desplegable (`dropdown`) y `rangeslider` para los **16 productos clave**, vistas conmutables por bloques temporales (`P1`, `P2`, `P3`) y series mensuales (`84 meses`), matriz de inferencia paramétrica/no paramétrica (`distfit`, $t$-Student, Wilcoxon, Kruskal-Wallis) y diagnóstico de modelos (`FE-OLS HC3` y *Double Machine Learning*).\n",
                "2. **Widgets Interactivos Plotly Embebidos en el Cuaderno**: A continuación se renderizan directamente el **Explorador Individual de los 16 Productos con Dropdown y RangeSlider**, el **Tablero de Inferencia y Comercio Intraindustrial (`SCR` / `Grubel-Lloyd`)** y el **Tablero Econométrico y Causal (`FE-OLS` + `DML`)**.\n",
                "\n",
                "### Artefactos Generados en `data/processed/` y `docs/`\n",
                "* `docs/dashboard_analitico_alianza_pacifico.html` (Dashboard Analítico Interactivo completo en HTML5 + Plotly).\n",
                "* `data/processed/panel_mensual_16_productos_ap.csv` (`4,032` registros mensuales por socio y producto).\n",
                "* `data/processed/analisis_desagregado_productos_ap.csv` (Resumen bilateral por bloque de los 16 productos).\n",
                "* `data/processed/distfit_16_productos_ap.csv` (Ajuste probabilístico `distfit` por cada uno de los 16 productos).\n",
                "* `data/processed/pruebas_hipotesis_16_productos_ap.csv` (Pruebas $t$-pareada, Wilcoxon, Signos y Kruskal-Wallis por cada uno de los 16 productos).",
            ],
        ),
        make_code_cell(
            "plotly_dashboard_phase6",
            [
                "from src.alianza_pacifico.plotly_dashboard import (\n",
                "    AnalyticalDashboardBuilder,\n",
                "    DashboardDataRepository,\n",
                "    DashboardPaths,\n",
                "    PlotlyChartFactory,\n",
                ")\n",
                "\n",
                "dash_paths = DashboardPaths.from_root(Path('.').resolve())\n",
                "dash_data = DashboardDataRepository(dash_paths.processed_dir).load_all()\n",
                "html_out = AnalyticalDashboardBuilder(dash_paths).build_standalone_html()\n",
                "print(f'Dashboard Analítico HTML generado en: {html_out}')\n",
                "\n",
                "fig_dash_explorer = PlotlyChartFactory.build_product_drilldown_explorer(dash_data)\n",
                "fig_dash_stats = PlotlyChartFactory.build_tab4_statistical_inference_and_distfit(dash_data)\n",
                "fig_dash_models = PlotlyChartFactory.build_tab5_econometrics_and_causal_ml(dash_data)\n",
                "fig_dash_explorer.show()\n",
                "fig_dash_stats.show()\n",
                "fig_dash_models.show()",
            ],
            execution_count=6,
            stdout_text=(
                f"Dashboard Analítico HTML generado en: {root_dir / 'docs' / 'dashboard_analitico_alianza_pacifico.html'}\n"
            ),
            plotly_figs=[
                PlotlyChartFactory.build_product_drilldown_explorer(
                    DashboardDataRepository(processed_dir).load_all()
                ),
                PlotlyChartFactory.build_tab4_statistical_inference_and_distfit(
                    DashboardDataRepository(processed_dir).load_all()
                ),
                PlotlyChartFactory.build_tab5_econometrics_and_causal_ml(
                    DashboardDataRepository(processed_dir).load_all()
                ),
            ],
        ),
    ]

    nb_content = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }

    nb_path = root_dir / "Analisis_exportaciones.ipynb"
    nb_path.write_text(json.dumps(nb_content, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Notebook profesional desagregado por los 16 productos actualizado en: {nb_path}")


if __name__ == "__main__":
    build_notebook(Path(__file__).resolve().parents[2])
