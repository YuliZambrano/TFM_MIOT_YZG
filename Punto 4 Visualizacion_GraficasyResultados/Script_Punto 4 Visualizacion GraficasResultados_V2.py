import pandas as pd
import numpy as np
import re
import ast
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html
from pathlib import Path
from datetime import datetime

# ============================================================
# 0) CONFIG
# ============================================================
FILE = "Punto3_nombrexxx archivo resultado punto 3.xlsx"  # <-- cambia PARA CADA PORTAL
OUT_HTML = "Punto4_nombrexxx archivo resultado punto 3.html"

PORTAL_COL = "portal"
plt.rcParams["figure.dpi"] = 120

EXPORT_DIR = Path("Punto4_HTML_por_grafica_12_ENERO_NUEVA VERSION")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

LEVELS = [
    ("bajo", 0, 30),
    ("medio", 30, 70),
    ("alto", 70, 100.0001),
]

GLOBAL_INDEX_NAME = "Índice Global de Madurez del Portal"

OPEN_FORMATS = {"CSV","JSON","GEOJSON","XML","RDF","TTL","TURTLE","N-TRIPLES","NT","JSON-LD","JSONLD"}

COMPONENT_COLORS = {
    "Abiertos": "#1f77b4",
    "Cerrados/Propietarios": "#ff7f0e",
    "Apertura total": "#1f77b4",
    "Restringida": "#d62728",
    "Vacío legal": "#7f7f7f",
}

DIMENSIONS = {
    "Trazabilidad Global": {"cols": {"Origen": "__calc__", "Temporal": "__calc__", "Única": "__calc__"}},
    "Interoperabilidad Técnica": {"cols": {"Formato abierto": "has_open_format", "DCAT/DCAT-AP": "portal_supports_dcat_dcatap"}},
    "Interoperabilidad Semántica": {"cols": {"Categoría definida": "category", "Vocabulario controlado": "uses_controlled_vocab", "Serialización semántica": "has_semantic_serialization"}},
    "Accesibilidad": {"cols": {"API REST": "portal_has_api_rest", "Licencia abierta": "license_open", "Acceso público OK": "public_access_ok", "URL descarga": "download_url_present"}},
    "Calidad de Metadatos": {"cols": {"Título": "title", "Descripción": "description", "Diccionario de datos": "has_data_dictionary"}},
}

WEIGHTS = {
    "Trazabilidad Global": 0.20,
    "Interoperabilidad Semántica": 0.20,
    "Interoperabilidad Técnica": 0.20,
    "Accesibilidad": 0.20,
    "Calidad de Metadatos": 0.20,
}

# ============================================================
# 1) LOAD
# ============================================================
if FILE.lower().endswith((".xlsx",".xls")):
    df = pd.read_excel(FILE)
else:
    df = pd.read_csv(FILE)

df.columns = [c.strip() for c in df.columns]

if PORTAL_COL not in df.columns:
    df[PORTAL_COL] = "Portal"

print("Cargado:", df.shape)
print("Portales:", df[PORTAL_COL].unique().tolist())
print("Columnas:", len(df.columns))

# ============================================================
# 2) HELPERS
# ============================================================
def _slugify(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s_-]+", "_", text)
    return text[:90] if len(text) > 90 else text

def save_plotly_figure_html(fig, filename_hint: str, out_dir: Path = EXPORT_DIR) -> str:
    safe = _slugify(filename_hint)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{safe}_{ts}.html"
    fig.write_html(
        str(out_path),
        full_html=True,
        include_plotlyjs=True,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": safe,
                "height": 900,
                "width": 1400,
                "scale": 2
            }
        }
    )
    return str(out_path)

def parse_list_cell(x):
    if isinstance(x, (list, tuple, set)):
        return list(x)
    if pd.isna(x):
        return []
    s = str(x).strip()
    if s == "":
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return v
    except:
        pass
    return [t.strip() for t in re.split(r"[;,|\s]+", s) if t.strip()]

def clean_text_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip()
    s2 = s2.replace({"nan":"", "NaN":"", "None":"", "N/A":"", "na":"", "null":""})
    return s2

def present_text(series: pd.Series) -> pd.Series:
    s = clean_text_series(series)
    bad = {"", "no definido", "no definida", "sin definir", "consultar", "consulte", "none"}
    return (~s.str.lower().isin(bad) & (s != "")).astype(int)

def present_binary(series: pd.Series) -> pd.Series:
    """
    0/1 estricto:
    - numérico: ==1 => 1
    - texto: "1" => 1
    - boolean: True => 1
    """
    s = series.copy()
    # boolean -> 1/0
    if s.dtype == "bool":
        return s.astype(int)
    # texto o mixed -> a numérico
    s = pd.to_numeric(s, errors="coerce")
    return (s == 1).astype(int)

def get_level(score_0_100: float) -> str:
    if pd.isna(score_0_100):
        return "no definido"
    for name, lo, hi in LEVELS:
        if lo <= score_0_100 < hi:
            return name
    return "no definido"

def pick_first_existing(candidates, df_cols):
    for c in candidates:
        if c in df_cols:
            return c
    return None

# ============================================================
# 3) Trazabilidad Temporal (agrupados + resumen)
# ============================================================
def grouped_age_percentages_modified(series, total):
    s = pd.to_numeric(series, errors="coerce")
    out = {
        "Grupo 1: 0–6 meses (Madurez)": ((s >= 0) & (s <= 6)).sum(),
        "Grupo 2: 6–12 meses": ((s > 6) & (s <= 12)).sum(),
        "Grupo 3: 1–2 años": ((s > 12) & (s <= 24)).sum(),
        "Grupo 4: ≥ 2 años": (s > 24).sum(),
        "Grupo 5: Vacío / sin definir": s.isna().sum()
    }
    dfb = pd.DataFrame({"Grupo": list(out.keys()), "Datasets": list(out.values())})
    dfb["Porcentaje"] = (dfb["Datasets"] / (total if total else 1) * 100).round(1)
    dfb["Etiqueta"] = dfb.apply(lambda r: f"{int(r.Datasets)} ({r.Porcentaje}%)", axis=1)
    return dfb

def update_frequency_group_new(series):
    s = series.astype(str).str.lower().str.strip()
    s = s.replace({"nan": "", "none": "", "nat": "", "null": ""})

    def classify(v: str):
        if v == "":
            return "Grupo 4: Vacío / sin definir"

        # Grupo 3: Nunca/irregular
        never_keys = ["never", "nunca", "sin periodicidad", "irregular", "sporadic", "eventual"]
        if any(k in v for k in never_keys):
            return "Grupo 3: Nunca"

        # Grupo 1: Alta/Media (se considera "maduro")
        g1_keys = [
            "instant", "realtime", "real time", "minut", "minute", "hora", "hour",
            "daily", "diari", "seman", "week", "monthly", "mensu", "quarter", "trimes",
            "semiannual", "semes", "cada 6", "6 meses", "biannual"
        ]
        if any(k in v for k in g1_keys):
            return "Grupo 1: Alta-Media (Horas/Instantanea/Diario)"

        # Grupo 2: Baja (anual y equivalentes)
        g2_keys = ["annual", "anual", "year", "yearly"]
        if any(k in v for k in g2_keys):
            return "Grupo 2: Baja (semestral/anual)"

        return "Grupo 3: Nunca"

    return s.apply(classify)

def grouped_update_frequency_new(series, total):
    grp = update_frequency_group_new(series)
    order = [
        "Grupo 1: Alta-Media (Horas/Instantanea/Diario)",
        "Grupo 2: Baja (semestral/anual)",
        "Grupo 3: Nunca",
        "Grupo 4: Vacío / sin definir"
    ]
    dfb = grp.value_counts().reindex(order, fill_value=0).reset_index()
    dfb.columns = ["Grupo", "Datasets"]
    dfb["Porcentaje"] = (dfb["Datasets"] / (total if total else 1) * 100).round(1)
    dfb["Etiqueta"] = dfb.apply(lambda r: f"{int(r.Datasets)} ({r.Porcentaje}%)", axis=1)
    return dfb

# IMPORTANTÍSIMO: aquí el nombre debe coincidir con el grupo exacto
UF_GROUP_1 = "Grupo 1: Alta-Media (Horas/Instantanea/Diario)"

def compute_temporal_summary_for_portal(g: pd.DataFrame) -> dict:
    """
    Temporal = promedio de:
    A) % datasets con age_months_modified en 0–6 meses (Grupo 1)
    B) % datasets cuya update_frequency_group pertenece al Grupo 1 (UF_GROUP_1)
    """
    n = len(g) if len(g) else 1

    if "age_months_modified" in g.columns:
        mod = pd.to_numeric(g["age_months_modified"], errors="coerce")
        a_cnt = int(((mod >= 0) & (mod <= 6)).sum())
        a_pct = (a_cnt / n) * 100
    else:
        a_cnt, a_pct = 0, 0.0

    # FIX: aquí SIEMPRE calculamos el grupo aunque no exista columna precalculada
    if "update_frequency" in g.columns:
        uf_grp = update_frequency_group_new(g["update_frequency"])
        b_cnt = int((uf_grp == UF_GROUP_1).sum())
        b_pct = (b_cnt / n) * 100
    elif "update_frequency_group" in g.columns:
        uf_grp = g["update_frequency_group"]
        b_cnt = int((uf_grp == UF_GROUP_1).sum())
        b_pct = (b_cnt / n) * 100
    else:
        b_cnt, b_pct = 0, 0.0

    score = float(np.mean([a_pct, b_pct]))
    return {
        "n": int(n),
        "modified_g1_cnt": int(a_cnt),
        "modified_g1_pct": float(round(a_pct, 1)),
        "update_g1_cnt": int(b_cnt),
        "update_g1_pct": float(round(b_pct, 1)),
        "score_temporal": float(round(score, 1)),
    }

# PRECALC: update_frequency_group (para tenerlo disponible también para depuración / otras vistas)
if "update_frequency" in df.columns:
    df["update_frequency_group"] = update_frequency_group_new(df["update_frequency"])

# ============================================================
# 4) FORMATS LISTS (para gráficas de técnica)
# ============================================================
if "open_formats_list" not in df.columns:
    df["open_formats_list"] = [[] for _ in range(len(df))]
if "non_open_formats_list" not in df.columns:
    df["non_open_formats_list"] = [[] for _ in range(len(df))]

df["open_formats_list_parsed"] = df["open_formats_list"].apply(parse_list_cell)
df["non_open_formats_list_parsed"] = df["non_open_formats_list"].apply(parse_list_cell)

# ============================================================
# 5) LICENSE BUCKET (gráfica legal)
# ============================================================
if "license" not in df.columns:
    df["license"] = np.nan
if "license_open" not in df.columns:
    df["license_open"] = np.nan

def license_bucket(row):
    lic = str(row.get("license","") if pd.notna(row.get("license",np.nan)) else "").lower().strip()
    open_flag = row.get("license_open", np.nan)

    if (
        lic == "" or "no definido" in lic or "sin definir" in lic
        or "consultar" in lic or "consulte" in lic
        or "permiso" in lic or "autoriz" in lic
    ):
        return "Vacío legal"

    if (
        "noncommercial" in lic or "no comercial" in lic or "no-comercial" in lic
        or re.search(r"\bby[-\s]?nc\b", lic)
        or "noderivatives" in lic or "sin deriv" in lic
        or re.search(r"\bby[-\s]?nd\b", lic)
    ):
        return "Restringida"

    if "avisolegal" in lic or "/aviso-legal" in lic or "aviso legal" in lic:
        return "Apertura total"

    if (
        "cc0" in lic
        or (
            (("creative commons" in lic) or re.search(r"\bcc\b", lic))
            and ("nc" not in lic and "nd" not in lic)
        )
    ):
        return "Apertura total"

    if pd.notna(open_flag) and int(open_flag) == 1:
        return "Apertura total"
    if pd.notna(open_flag) and int(open_flag) == 0:
        return "Restringida"

    return "Vacío legal"

df["license_bucket"] = df.apply(license_bucket, axis=1)

# ============================================================
# 6) PORTAL-LEVEL SCORES — METODOLOGÍA
# ============================================================
def is_defined_category(x) -> int:
    s = "" if pd.isna(x) else str(x).strip().lower()
    if s in ["", "nan", "none", "no definido", "no definida", "sin definir"]:
        return 0
    return 1

# FIX: detectar la columna de "URI OK" aunque venga con typo
URI_OK_COL = pick_first_existing(
    [
        "dataset_uri_access_ok",
        "datset_uri_access_ok",   # typo común
        "dataset_uri_ok",
        "dataset_uri_accessible",
        "dataset_uri_accessible_ok",
        "dataset_uri_http_ok",
        "dataset_uri_status_ok",
        "dataset_url_access_ok",
        "dataset_url_ok",
        "public_access_ok",       # fallback si el portal lo usa así
        "download_url_access_ok"  # último fallback
    ],
    df_cols=df.columns
)

print("\n[DEBUG] Columna URI OK detectada para trazabilidad única:", URI_OK_COL)

portal_results = {}

for portal, g in df.groupby(PORTAL_COL):
    g = g.copy()
    n = len(g) if len(g) else 1

    dim_scores = {}
    dim_comp_pct = {}
    dim_comp_count = {}

    # -----------------------------
    # 1) TRAZABILIDAD GLOBAL
    # Origen = publisher + identifier
    # Temporal = modified(0-6) + updatefreq(grupo1)
    # Única = (URI_OK_COL) + doi
    # -----------------------------
    pub_pct = (present_text(g["publisher"]).mean() * 100) if "publisher" in g.columns else 0.0
    id_pct  = (present_text(g["identifier"]).mean() * 100) if "identifier" in g.columns else 0.0
    score_origen = float(np.mean([pub_pct, id_pct]))

    tinfo = compute_temporal_summary_for_portal(g)
    score_temporal = float(tinfo["score_temporal"])

    # FIX 1: Única (URI OK) tomando columna detectada
    if URI_OK_COL and URI_OK_COL in g.columns:
        uri_pct = float(present_binary(g[URI_OK_COL]).mean() * 100)
        uri_cnt = int(present_binary(g[URI_OK_COL]).sum())
    else:
        uri_pct, uri_cnt = 0.0, 0

    doi_pct = (present_text(g["doi"]).mean() * 100) if "doi" in g.columns else 0.0
    score_unica = float(np.mean([uri_pct, doi_pct]))

    score_traz_global = float(np.mean([score_origen, score_temporal, score_unica]))
    dim_scores["Trazabilidad Global"] = score_traz_global

    dim_comp_pct["Trazabilidad Global"] = {
        "Origen": round(score_origen, 1),
        "Temporal": round(score_temporal, 1),
        "Única": round(score_unica, 1),
    }
    dim_comp_count["Trazabilidad Global"] = {
        "Origen": int(round((score_origen / 100) * n)),
        "Temporal": int(round((score_temporal / 100) * n)),
        "Única": int(round((score_unica / 100) * n)),
    }

    # -----------------------------
    # 2) INTEROP. TÉCNICA
    # -----------------------------
    comp_pcts, comp_counts = {}, {}
    for comp_label, col in DIMENSIONS["Interoperabilidad Técnica"]["cols"].items():
        if col in g.columns:
            pres = present_binary(g[col])
        else:
            pres = pd.Series([0]*n)
        comp_counts[comp_label] = int(pres.sum())
        comp_pcts[comp_label] = float((pres.mean() * 100) if n else 0.0)

    dim_scores["Interoperabilidad Técnica"] = float(np.mean(list(comp_pcts.values())))
    dim_comp_pct["Interoperabilidad Técnica"] = comp_pcts
    dim_comp_count["Interoperabilidad Técnica"] = comp_counts

    # -----------------------------
    # 3) INTEROP. SEMÁNTICA
    # -----------------------------
    comp_pcts, comp_counts = {}, {}
    for comp_label, col in DIMENSIONS["Interoperabilidad Semántica"]["cols"].items():
        if col not in g.columns:
            pres = pd.Series([0]*n)
        else:
            if col == "category":
                pres = g[col].apply(is_defined_category)
            else:
                pres = present_binary(g[col])
        comp_counts[comp_label] = int(pd.Series(pres).sum())
        comp_pcts[comp_label] = float((pd.Series(pres).mean() * 100) if n else 0.0)

    dim_scores["Interoperabilidad Semántica"] = float(np.mean(list(comp_pcts.values())))
    dim_comp_pct["Interoperabilidad Semántica"] = comp_pcts
    dim_comp_count["Interoperabilidad Semántica"] = comp_counts

    # -----------------------------
    # 4) ACCESIBILIDAD
    # -----------------------------
    comp_pcts, comp_counts = {}, {}
    for comp_label, col in DIMENSIONS["Accesibilidad"]["cols"].items():
        if col not in g.columns:
            pres = pd.Series([0]*n)
        else:
            pres = present_binary(g[col])
        comp_counts[comp_label] = int(pres.sum())
        comp_pcts[comp_label] = float((pres.mean() * 100) if n else 0.0)

    dim_scores["Accesibilidad"] = float(np.mean(list(comp_pcts.values())))
    dim_comp_pct["Accesibilidad"] = comp_pcts
    dim_comp_count["Accesibilidad"] = comp_counts

    # -----------------------------
    # 5) CALIDAD
    # -----------------------------
    comp_pcts, comp_counts = {}, {}
    for comp_label, col in DIMENSIONS["Calidad de Metadatos"]["cols"].items():
        if col not in g.columns:
            pres = pd.Series([0]*n)
        else:
            if col in ["title", "description"]:
                pres = present_text(g[col])
            else:
                pres = present_binary(g[col])
        comp_counts[comp_label] = int(pres.sum())
        comp_pcts[comp_label] = float((pres.mean() * 100) if n else 0.0)

    dim_scores["Calidad de Metadatos"] = float(np.mean(list(comp_pcts.values())))
    dim_comp_pct["Calidad de Metadatos"] = comp_pcts
    dim_comp_count["Calidad de Metadatos"] = comp_counts

    # -----------------------------
    # ÍNDICE GLOBAL
    # -----------------------------
    weighted_sum = 0.0
    weight_used = 0.0
    for dim_name, w in WEIGHTS.items():
        val = dim_scores.get(dim_name, np.nan)
        if pd.notna(val):
            weighted_sum += w * float(val)
            weight_used += w

    global_score = (weighted_sum / weight_used) if weight_used > 0 else np.nan
    global_level = get_level(global_score)

    portal_results[portal] = {
        "n": int(n),
        "dim_scores": dim_scores,
        "dim_comp_pct": dim_comp_pct,
        "dim_comp_count": dim_comp_count,
        "global_score": float(global_score),
        "global_level": global_level,
        "temporal_info": tinfo,
        "uri_ok_col": URI_OK_COL
    }

print("\nResumen portal-level:")
for p, r in portal_results.items():
    print(f"- {p}: N={r['n']}, {GLOBAL_INDEX_NAME}={r['global_score']:.1f} ({r['global_level']})")

# ============================================================
# 7) EXTRA: categoría
# ============================================================
def pick_first_existing_global(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

CATEGORY_COL = pick_first_existing_global(["category", "categoría", "categoria", "theme", "themes", "tematica", "temática", "tags"])
print("\nColumna categoría detectada:", CATEGORY_COL)

# ============================================================
# 8) PLOTLY HELPERS
# ============================================================
PLOTLY_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "height": 900, "width": 1400, "scale": 2}}

def plotly_polish(fig, height=520):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=70, r=30, t=90, b=70),
        title=dict(x=0.02, y=0.97),
        font=dict(size=14),
    )
    return fig

def plotly_caption(text):
    return f"<div class='cap'>{text}</div>"

def plotly_fig_block(fig, caption_html, export_name=None):
    if export_name is None:
        export_name = fig.layout.title.text if fig.layout.title and fig.layout.title.text else "figura"
    _ = save_plotly_figure_html(fig, export_name)
    fig_html = to_html(fig, full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG)
    return f"""
    <div class="card">
      {fig_html}
      {caption_html}
    </div>
    """

# ============================================================
# 9) GRÁFICOS GLOBALES (Formatos + Licencias)
# ============================================================
def plotly_base_figs():
    blocks = []

    # Clasificación de formatos (presencia)
    format_counts = []
    for f in sorted(list(OPEN_FORMATS)):
        cnt = int(df["open_formats_list_parsed"].apply(lambda lst: 1 if f in lst else 0).sum())
        if cnt > 0:
            format_counts.append({"Formato": f, "Datasets": cnt, "Tipo": "Abierto"})
    closed_cnt = int(df["non_open_formats_list_parsed"].apply(lambda lst: 1 if len(lst) > 0 else 0).sum())
    format_counts.append({"Formato": "Cerrados/Propietarios (agrupado)", "Datasets": closed_cnt, "Tipo": "Cerrado"})
    fmt_df = pd.DataFrame(format_counts).sort_values("Datasets", ascending=False)

    fig1 = px.bar(
        fmt_df, x="Formato", y="Datasets", color="Tipo",
        title="Interoperabilidad Técnica — Clasificación de formatos (datasets con presencia)",
        color_discrete_map={"Abierto": COMPONENT_COLORS["Abiertos"], "Cerrado": COMPONENT_COLORS["Cerrados/Propietarios"]},
        text="Datasets"
    )
    fig1.update_traces(textposition="outside", cliponaxis=False)
    fig1.update_layout(yaxis_title="Nº de datasets", xaxis_tickangle=-25)
    fig1 = plotly_polish(fig1, height=520)
    blocks.append(plotly_fig_block(
        fig1,
        plotly_caption("<b>Interpretación.</b> Conteo de datasets con presencia de formatos abiertos (por tipo) y datasets con al menos un formato cerrado/propietario (agrupado)."),
        export_name="global_clasificacion_formatos"
    ))

    # Distribución porcentual datasets con algún formato abierto/cerrado
    any_open = df["open_formats_list_parsed"].apply(lambda lst: 1 if len(lst) > 0 else 0)
    any_closed = df["non_open_formats_list_parsed"].apply(lambda lst: 1 if len(lst) > 0 else 0)
    den = len(df) if len(df) else 1
    pct_df = pd.DataFrame({
        "Clasificación": ["Datasets con formatos abiertos", "Datasets con formatos cerrados/propietarios"],
        "Porcentaje": [100*any_open.sum()/den, 100*any_closed.sum()/den]
    })
    fig2 = px.bar(
        pct_df, x="Clasificación", y="Porcentaje", color="Clasificación",
        title="Interoperabilidad Técnica — Distribución porcentual (abiertos vs cerrados)",
        text=pct_df["Porcentaje"].round(1)
    )
    fig2.update_traces(textposition="outside", cliponaxis=False)
    fig2.update_layout(yaxis_range=[0,105], yaxis_title="Porcentaje (%)")
    fig2 = plotly_polish(fig2, height=440)
    blocks.append(plotly_fig_block(
        fig2,
        plotly_caption("<b>Interpretación.</b> Proporción de datasets que reportan al menos un formato abierto y/o al menos un formato cerrado/propietario."),
        export_name="global_pct_formatos"
    ))

    # Licenciamiento
    lic_vc = df["license_bucket"].value_counts().reindex(["Apertura total","Restringida","Vacío legal"]).fillna(0).reset_index()
    lic_vc.columns = ["Categoría","Datasets"]
    fig3 = px.bar(
        lic_vc, x="Categoría", y="Datasets", color="Categoría",
        title="Accesibilidad (Dimensión legal) — Licenciamiento permitido",
        color_discrete_map={
            "Apertura total": COMPONENT_COLORS["Apertura total"],
            "Restringida": COMPONENT_COLORS["Restringida"],
            "Vacío legal": COMPONENT_COLORS["Vacío legal"],
        },
        text="Datasets"
    )
    fig3.update_traces(textposition="outside", cliponaxis=False)
    fig3.update_layout(yaxis_title="Nº de datasets")
    fig3 = plotly_polish(fig3, height=440)
    blocks.append(plotly_fig_block(
        fig3,
        plotly_caption("<b>Interpretación.</b> Apertura total: licencia abierta. Restringida: NC/ND u otras restricciones. Vacío legal: licencia ausente/no definida/consultar."),
        export_name="global_licenciamiento"
    ))

    return blocks

# ============================================================
# 10) GRÁFICOS POR PORTAL
# ============================================================
def plotly_portal_blocks(portal: str):
    r = portal_results[portal]
    n = r["n"]
    blocks = []

    # A) Índice Global
    score = r["global_score"]
    level = r["global_level"]
    color = "#2ca02c" if level == "alto" else "#ff7f0e" if level == "medio" else "#d62728"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[score], y=[portal], orientation="h", marker_color=color,
        text=[f"{score:.1f}/100 ({level})"], textposition="outside"
    ))
    fig.update_layout(
        title=f"{GLOBAL_INDEX_NAME} — {portal} (N={n})",
        xaxis=dict(range=[0,105], title="Índice Global (0–100)"),
        yaxis=dict(title=""),
        height=230
    )
    fig = plotly_polish(fig, height=240)
    blocks.append(plotly_fig_block(
        fig,
        plotly_caption("<b>Interpretación.</b> Índice global calculado como promedio ponderado de 5 componentes (20% cada uno). Cortes: Bajo 0–30, Medio 30–70, Alto 70–100."),
        export_name=f"{portal}_indice_global"
    ))

    # B) Componentes del índice (orden fijo)
    dim_scores = r["dim_scores"]
    component_order = ["Interoperabilidad Semántica", "Interoperabilidad Técnica", "Trazabilidad Global", "Accesibilidad", "Calidad de Metadatos"]
    t = pd.DataFrame({"Componente": list(dim_scores.keys()), "Score": list(dim_scores.values())})
    t["Componente"] = pd.Categorical(t["Componente"], categories=component_order, ordered=True)
    t = t[t["Componente"].notna()].sort_values("Componente")

    fig = px.bar(
        t, x="Componente", y="Score", text=t["Score"].round(1),
        title=f"Componentes que forman el Índice de Madurez — {portal}"
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis_range=[0,105], xaxis_tickangle=-25, yaxis_title="Score (%)")
    fig = plotly_polish(fig, height=520)
    blocks.append(plotly_fig_block(
        fig,
        plotly_caption("<b>Interpretación.</b> Los 5 componentes en orden fijo. Cada componente aporta 20% al Índice Global."),
        export_name=f"{portal}_componentes_indice"
    ))

    g = df[df[PORTAL_COL] == portal].copy()

    # ============================================================
    # 1) TRAZABILIDAD ORIGEN
    # ============================================================
    origen_rows = []
    for lab, col in [("Publisher", "publisher"), ("Identifier", "identifier")]:
        if col in g.columns:
            pres = present_text(g[col])
            cnt = int(pres.sum())
            pct = float(pres.mean() * 100) if n else 0.0
        else:
            cnt, pct = 0, 0.0
        origen_rows.append({"Variable": lab, "Datasets": cnt, "Porcentaje": round(pct, 1)})

    t_origen = pd.DataFrame(origen_rows)
    t_origen["Etiqueta"] = t_origen.apply(lambda rr: f"{int(rr['Datasets'])} ({rr['Porcentaje']}%)", axis=1)
    score_origen = float(np.mean(t_origen["Porcentaje"])) if len(t_origen) else 0.0

    fig_origen = px.bar(
        t_origen.sort_values("Datasets", ascending=True),
        x="Datasets", y="Variable", orientation="h", text="Etiqueta",
        title=f"Trazabilidad Origen (Score={score_origen:.1f}%) | {portal}",
        color="Variable",
        color_discrete_map={"Publisher": "#1f77b4", "Identifier": "#2ca02c"}
    )
    fig_origen.update_traces(textposition="outside", cliponaxis=False)
    fig_origen.update_layout(showlegend=False, xaxis_title="Nº de datasets", yaxis_title="")
    fig_origen = plotly_polish(fig_origen, height=420)
    blocks.append(plotly_fig_block(
        fig_origen,
        plotly_caption("<b>Interpretación.</b> Score = promedio simple del % de presencia de publisher e identifier."),
        export_name=f"{portal}_trazabilidad_origen"
    ))

    # ============================================================
    # 2) TRAZABILIDAD GLOBAL — RESUMEN
    # ============================================================
    tg = r["dim_comp_pct"]["Trazabilidad Global"]
    tg_df = pd.DataFrame({"Componente": list(tg.keys()), "Porcentaje": list(tg.values())}).sort_values("Porcentaje", ascending=True)
    tg_df["Etiqueta"] = tg_df["Porcentaje"].round(1).astype(str) + "%"

    figTG = px.bar(
        tg_df, x="Porcentaje", y="Componente", orientation="h", text="Etiqueta",
        title=f"Trazabilidad Global (Score={r['dim_scores']['Trazabilidad Global']:.1f}%) | {portal}",
        labels={"Porcentaje": "% de datasets"}
    )
    figTG.update_layout(xaxis_range=[0,105], showlegend=False)
    figTG.update_traces(textposition="outside", cliponaxis=False)
    figTG.add_vline(x=r["dim_scores"]["Trazabilidad Global"], line_width=3, line_dash="dash")
    figTG = plotly_polish(figTG, height=420)
    blocks.append(plotly_fig_block(
        figTG,
        plotly_caption("<b>Interpretación.</b> Trazabilidad Global = promedio simple de Origen, Temporal y Única. La línea vertical indica el score final."),
        export_name=f"{portal}_trazabilidad_global"
    ))

    # ============================================================
    # 3) TRAZABILIDAD TEMPORAL (RESUMEN + 2 GRÁFICAS)
    # ============================================================
    tinfo = r.get("temporal_info", compute_temporal_summary_for_portal(g))

    tmp = pd.DataFrame({
        "Variable": ["Antigüedad modified: 0–6m (Maduro)", f"Frecuencia actualización: {UF_GROUP_1}"],
        "Porcentaje": [tinfo["modified_g1_pct"], tinfo["update_g1_pct"]],
    })
    tmp["Etiqueta"] = tmp["Porcentaje"].round(1).astype(str) + "%"

    figT = px.bar(
        tmp, x="Porcentaje", y="Variable", orientation="h", text="Etiqueta",
        title=f"Resumen Trazabilidad Temporal (Score={tinfo['score_temporal']}%) | {portal}",
        labels={"Porcentaje": "% sobre el total de datasets"}
    )
    figT.update_layout(xaxis_range=[0,105], showlegend=False)
    figT.update_traces(textposition="outside", cliponaxis=False)
    figT = plotly_polish(figT, height=440)
    blocks.append(plotly_fig_block(
        figT,
        plotly_caption("<b>Interpretación.</b> Trazabilidad Temporal = promedio de (1) % con modified en 0–6 meses y (2) % con update_frequency clasificada en Grupo 1."),
        export_name=f"{portal}_trazabilidad_temporal_resumen"
    ))

    if "age_months_modified" in g.columns:
        tmpm = grouped_age_percentages_modified(g["age_months_modified"], len(g))
        figM = px.bar(
            tmpm, x="Grupo", y="Porcentaje", text="Etiqueta",
            title=f"Trazabilidad Temporal — Antigüedad desde última actualización (modified) | {portal}",
            labels={"Porcentaje": "% datasets"}
        )
        figM.update_layout(yaxis_range=[0,105], xaxis_tickangle=-20)
        figM.update_traces(textposition="outside", cliponaxis=False)
        figM = plotly_polish(figM, height=520)
        blocks.append(plotly_fig_block(
            figM,
            plotly_caption("<b>Interpretación.</b> Grupo 1 (0–6 meses) alimenta el cálculo del score temporal."),
            export_name=f"{portal}_temporal_modified_grupos"
        ))

    if "update_frequency" in g.columns:
        tmpu = grouped_update_frequency_new(g["update_frequency"], len(g))
        figU = px.bar(
            tmpu, x="Grupo", y="Porcentaje", text="Etiqueta",
            title=f"Trazabilidad Temporal — Distribución de frecuencia de actualización | {portal}",
            labels={"Porcentaje": "% datasets"}
        )
        figU.update_layout(yaxis_range=[0,105], xaxis_tickangle=-20)
        figU.update_traces(textposition="outside", cliponaxis=False)
        figU = plotly_polish(figU, height=520)
        blocks.append(plotly_fig_block(
            figU,
            plotly_caption(
                "<b>Interpretación.</b> Agrupación de update_frequency:"
                "<br><b>Grupo 1:</b> instantánea/minutos/horas/diaria/semanal/mensual/trimestral/semestral."
                "<br><b>Grupo 2:</b> anual."
                "<br><b>Grupo 3:</b> nunca/irregular/eventual."
                "<br><b>Grupo 4:</b> vacío/no definido."
            ),
            export_name=f"{portal}_temporal_updatefreq_grupos"
        ))

    # ============================================================
    # 4) TRAZABILIDAD ÚNICA (URI_OK + DOI) — FIX REAL
    # ============================================================
    unica_rows = []

    uri_col = r.get("uri_ok_col", URI_OK_COL)
    if uri_col and uri_col in g.columns:
        pres_uri = present_binary(g[uri_col])
        cnt_uri = int(pres_uri.sum())
        pct_uri = float(pres_uri.mean() * 100)
    else:
        cnt_uri, pct_uri = 0, 0.0

    # DOI
    if "doi" in g.columns:
        pres_doi = present_text(g["doi"])
        cnt_doi = int(pres_doi.sum())
        pct_doi = float(pres_doi.mean() * 100)
    else:
        cnt_doi, pct_doi = 0, 0.0

    unica_rows.append({"Variable": f"URI OK ({uri_col if uri_col else 'no detectada'})", "Datasets": cnt_uri, "Porcentaje": round(pct_uri, 1)})
    unica_rows.append({"Variable": "DOI", "Datasets": cnt_doi, "Porcentaje": round(pct_doi, 1)})

    t_unica = pd.DataFrame(unica_rows)
    t_unica["Etiqueta"] = t_unica.apply(lambda rr: f"{int(rr['Datasets'])} ({rr['Porcentaje']}%)", axis=1)
    score_unica = float(np.mean(t_unica["Porcentaje"])) if len(t_unica) else 0.0

    fig_unica = px.bar(
        t_unica.sort_values("Datasets", ascending=True),
        x="Datasets", y="Variable", orientation="h", text="Etiqueta",
        title=f"Trazabilidad Única (Score={score_unica:.1f}%) | {portal}",
    )
    fig_unica.update_traces(textposition="outside", cliponaxis=False)
    fig_unica.update_layout(showlegend=False, xaxis_title="Nº de datasets", yaxis_title="")
    fig_unica = plotly_polish(fig_unica, height=440)
    blocks.append(plotly_fig_block(
        fig_unica,
        plotly_caption(
            "<b>Interpretación.</b> Trazabilidad Única = promedio simple de (1) presencia de URI accesible/OK (campo binario=1) y (2) DOI presente."
            f"<br><b>Debug:</b> columna URI OK usada: <b>{uri_col}</b>."
        ),
        export_name=f"{portal}_trazabilidad_unica"
    ))

    # ============================================================
    # 5) Barras por dimensión (Interoperabilidad/Accesibilidad/Calidad)
    # ============================================================
    for dim_name in ["Interoperabilidad Técnica", "Interoperabilidad Semántica", "Accesibilidad", "Calidad de Metadatos"]:
        comp_pct = r["dim_comp_pct"][dim_name]
        comp_cnt = r["dim_comp_count"][dim_name]
        dim_score = r["dim_scores"][dim_name]
        dim_level = get_level(dim_score)

        tdim = pd.DataFrame({
            "Variable": list(comp_pct.keys()),
            "Porcentaje": [float(comp_pct[k]) for k in comp_pct.keys()],
            "Conteo": [int(comp_cnt[k]) for k in comp_pct.keys()]
        }).sort_values("Porcentaje", ascending=True)

        tdim["Etiqueta"] = tdim.apply(lambda rr: f"{rr['Porcentaje']:.1f}% ({int(rr['Conteo'])})", axis=1)

        figD = px.bar(
            tdim, x="Porcentaje", y="Variable", orientation="h", text="Etiqueta",
            title=f"{dim_name} (Score={dim_score:.1f}%) | {portal}",
            labels={"Porcentaje": "Presencia/Cumplimiento en datasets (%)"}
        )
        figD.update_layout(xaxis_range=[0,105], showlegend=False)
        figD.update_traces(textposition="outside", cliponaxis=False)
        figD.add_vline(x=dim_score, line_width=3)
        figD = plotly_polish(figD, height=520)

        blocks.append(plotly_fig_block(
            figD,
            plotly_caption(f"<b>Interpretación.</b> Score = promedio simple de variables. Línea vertical: <b>{dim_score:.1f}/100</b> (<b>{dim_level}</b>)."),
            export_name=f"{portal}_{_slugify(dim_name)}"
        ))

    # ============================================================
    # 6 Categorías
    # ============================================================
    if CATEGORY_COL and CATEGORY_COL in g.columns:
        s = g[CATEGORY_COL].astype(str).str.strip().replace({"nan":"", "NaN":"", "None":""})
        s = s.replace("", "No definido")
        vc = s.value_counts(dropna=False).reset_index()
        vc.columns = ["Categoría", "Datasets"]
        vc = vc.sort_values("Datasets", ascending=False)

        figC = px.bar(
            vc, x="Categoría", y="Datasets", text="Datasets",
            title=f"Distribución de categorías por Portal | {portal}"
        )
        figC.update_traces(textposition="outside", cliponaxis=False)
        figC.update_layout(height=700, xaxis_title="Categorías", yaxis_title="Nº de datasets", xaxis_tickangle=-45)
        figC = plotly_polish(figC, height=740)

        blocks.append(plotly_fig_block(
            figC,
            plotly_caption("<b>Interpretación.</b> Conteo total de datasets por categoría. Incluye <i>No definido</i> si el campo está vacío."),
            export_name=f"{portal}_categorias"
        ))

    return blocks

# ============================================================
# 11) HTML WRAPPER
# ============================================================
def build_html(blocks):
    return f"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8"/>
  <title>Grafico – Madurez Open Data</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 22px; background: #fafafa; }}
    h1 {{ margin-bottom: 6px; }}
    h2 {{ margin-top: 26px; margin-bottom: 10px; }}
    .sub {{ color: #555; margin-bottom: 18px; }}
    .card {{
      background: white;
      border: 1px solid #ddd;
      border-radius: 14px;
      padding: 12px;
      margin-bottom: 16px;
      box-shadow: 0 1px 6px rgba(0,0,0,0.04);
    }}
    .cap {{
      margin-top: 10px;
      font-size: 13.5px;
      color: #222;
      line-height: 1.35;
      background: #f6f6f6;
      border: 1px solid #eee;
      border-radius: 12px;
      padding: 10px 12px;
    }}
    .note {{
      font-size: 13px;
      color: #444;
      background: #fffbe6;
      border: 1px solid #f2e6a2;
      border-radius: 12px;
      padding: 10px 12px;
      margin: 10px 0 18px 0;
    }}
  </style>
</head>
<body>
  <h1>Grafico – Madurez del ecosistema de datos abiertos</h1>
  <div class="sub">
    Interactivo (Plotly): usa el icono de cámara en cada gráfico para descargar la imagen.
  </div>

  <div class="note">
    <b>Metodología.</b> Índice global = promedio ponderado de 5 componentes con pesos iguales (20% c/u):
    Trazabilidad Global, Interoperabilidad Semántica, Interoperabilidad Técnica, Accesibilidad y Calidad de Metadatos.
    Cortes: Bajo 0–30, Medio 30–70, Alto 70–100.
  </div>

  {''.join(blocks)}
</body>
</html>
"""

# ============================================================
# 12) EJECUCIÓN PRINCIPAL
# ============================================================
final_blocks = []
final_blocks.append("<h1>Grafico Global (Formatos y Licencias)</h1>")
final_blocks += plotly_base_figs()

for portal in portal_results.keys():
    final_blocks.append(f"<h2>Portal: {portal}</h2>")
    final_blocks += plotly_portal_blocks(portal)

html_final = build_html(final_blocks)
Path(OUT_HTML).write_text(html_final, encoding="utf-8")
print(f"\n Archivo generado exitosamente: {OUT_HTML}")
print(f"HTMLs individuales por gráfica en: {EXPORT_DIR.resolve()}")