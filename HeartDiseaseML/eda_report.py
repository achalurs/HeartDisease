# edareport.py
# ------------------------------------------------------------
# Helpers to build a SINGLE-FILE EDA HTML report and a PDF
# that the Streamlit app can generate and download.
#
# Exposes:
#   build_eda_report_html(df) -> (html_string, ordered_images)
#   build_pdf_from_images(ordered_images, title) -> pdf_bytes
# ------------------------------------------------------------

import io
import base64
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader


# ---------- internal helpers ----------
def _to_snake(name: str) -> str:
    return name.strip().lower().replace(" ", "_")

def _fig_to_b64(fig) -> str:
    """Matplotlib figure -> data URI (base64 PNG)"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{b64}"

def _html_img(b64, alt="", width="720px"):
    return f'<img src="{b64}" alt="{alt}" style="max-width:{width}; width:100%; height:auto; border:1px solid #eee; border-radius:8px;" />'


# ---------- public: HTML builder ----------
def build_eda_report_html(df: pd.DataFrame):
    """
    Returns:
      html_string: str
      ordered_images: List[(title, data_uri_string)]
    """
    df = df.copy()
    df.columns = [_to_snake(c) for c in df.columns]

    target = "target"
    numeric = [c for c in ["age", "resting_bp_s", "cholesterol", "max_heart_rate", "oldpeak"] if c in df.columns]
    categ  = [c for c in ["sex", "chest_pain_type", "fasting_blood_sugar", "resting_ecg", "exercise_angina", "st_slope"] if c in df.columns]

    images = {}
    ordered = []  # (title, key) preserve order for PDF

    # Target distribution
    fig = plt.figure(figsize=(5.5, 4))
    ax = sns.countplot(x=target, data=df)
    ax.bar_label(ax.containers[0])
    plt.title("Target Distribution (0 = No Disease, 1 = Disease)")
    plt.xlabel("target"); plt.ylabel("count")
    images["target"] = _fig_to_b64(fig)
    ordered.append(("Target Distribution", "target"))

    # Correlation
    corr_html = "<p><em>Not enough numeric columns.</em></p>"
    if numeric:
        cols = numeric + [target]
        corr = df[cols].corr(method="pearson")
        fig = plt.figure(figsize=(7, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", square=True, cmap="coolwarm", cbar=True)
        plt.title("Correlation Heatmap (numeric + target)")
        images["corr"] = _fig_to_b64(fig)
        ordered.append(("Correlation Heatmap", "corr"))

        tc = corr[target].drop(target).sort_values(key=lambda s: s.abs(), ascending=False)
        corr_html = tc.to_frame("corr_with_target").to_html(border=0, classes="tbl")

    # Numeric plots
    for col in numeric:
        fig = plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), bins=30, kde=True)
        plt.title(f"Distribution: {col}")
        plt.xlabel(col); plt.ylabel("count")
        images[f"{col}_hist"] = _fig_to_b64(fig)
        ordered.append((f"{col} • Histogram", f"{col}_hist"))

        fig = plt.figure(figsize=(6, 4))
        sns.boxplot(x=target, y=col, data=df, showfliers=False)
        plt.title(f"{col} by {target}")
        plt.xlabel("target (0/1)"); plt.ylabel(col)
        images[f"{col}_box"] = _fig_to_b64(fig)
        ordered.append((f"{col} • By Target (Boxplot)", f"{col}_box"))

    # Categorical plots
    for col in categ:
        fig = plt.figure(figsize=(6, 4))
        ax = sns.countplot(x=col, data=df)
        ax.bar_label(ax.containers[0])
        plt.title(f"Counts: {col}")
        plt.xlabel(col); plt.ylabel("count")
        images[f"{col}_counts"] = _fig_to_b64(fig)
        ordered.append((f"{col} • Counts", f"{col}_counts"))

        ct = df.groupby([col, target]).size().reset_index(name="count")
        total = ct.groupby(col)["count"].transform("sum")
        ct["percent"] = ct["count"] / total
        pv = ct.pivot(index=col, columns=target, values="percent").fillna(0.0).sort_index(axis=1)

        fig = plt.figure(figsize=(7, 4))
        pv.plot(kind="bar", stacked=True, ax=plt.gca())
        plt.title(f"{col} – % distribution by target")
        plt.xlabel(col); plt.ylabel("percent"); plt.legend(title="target", loc="upper right")
        images[f"{col}_stacked"] = _fig_to_b64(fig)
        ordered.append((f"{col} • % by Target (Stacked)", f"{col}_stacked"))

    # Tables
    if numeric:
        summary_num = df[numeric + [target]].describe().T
        num_tbl = summary_num.to_html(border=0, classes="tbl")
    else:
        num_tbl = "<em>No numeric columns.</em>"

    rows = []
    for col in categ:
        vc = df[col].value_counts(dropna=False, normalize=False)
        vcp = df[col].value_counts(dropna=False, normalize=True)
        for k in vc.index:
            rows.append({"column": col, "value": k, "count": int(vc[k]), "percent": float(vcp[k])})
    cat_tbl = pd.DataFrame(rows).to_html(index=False, border=0, classes="tbl") if rows else "<em>No categorical columns.</em>"

    # HTML
    HTML = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Heart Disease EDA Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, "Noto Sans", sans-serif; margin: 24px; color: #222; }}
  h1 {{ margin-top: 0; }}
  .meta {{ color:#666; font-size: 0.95rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
  .card {{ padding: 14px; border: 1px solid #eee; border-radius: 10px; background:#fff; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }}
  .tbl {{ border-collapse: collapse; width: 100%; font-size: 0.95rem; }}
  .tbl th, .tbl td {{ padding: 8px 10px; border-bottom: 1px solid #f0f0f0; text-align: left; }}
  .muted {{ color:#8a8a8a; }}
</style>
</head>
<body>

<h1>Heart Disease — EDA Report</h1>
<div class="meta">
  Generated from <code>dataset.csv</code> • Created: {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>

<h2>Target Distribution</h2>
<div class="card">{_html_img(images["target"], "target distribution")}</div>

<h2>Correlation (Numeric + Target)</h2>
<div class="grid">
  <div class="card">{_html_img(images["corr"], "correlation heatmap") if "corr" in images else "<em class='muted'>Not enough numeric columns.</em>"}</div>
  <div class="card"><div class="muted">Correlation with target</div>{corr_html}</div>
</div>

<h2>Numeric Features</h2>
<div class="grid">
"""
    for col in numeric:
        HTML += f"""
  <div class="card"><div class="muted">Histogram + KDE</div><strong>{col}</strong>{_html_img(images[f"{col}_hist"], f"{col} hist")}</div>
  <div class="card"><div class="muted">By target (boxplot)</div><strong>{col}</strong>{_html_img(images[f"{col}_box"], f"{col} by target")}</div>
"""
    HTML += """
</div>

<h2>Categorical Features</h2>
<div class="grid">
"""
    for col in categ:
        HTML += f"""
  <div class="card"><div class="muted">Counts</div><strong>{col}</strong>{_html_img(images[f"{col}_counts"], f"{col} counts")}</div>
  <div class="card"><div class="muted">% by target (stacked)</div><strong>{col}</strong>{_html_img(images[f"{col}_stacked"], f"{col} stacked")}</div>
"""
    HTML += f"""
</div>

<h2>Summary Tables</h2>
<div class="grid">
  <div class="card"><div class="muted">Numeric</div>{num_tbl}</div>
  <div class="card"><div class="muted">Categorical</div>{cat_tbl}</div>
</div>

<hr style="margin:28px 0 16px; border:none; border-top:1px solid #eee;">
<div class="muted">⚠️ Educational purposes only — not medical advice.</div>

</body>
</html>
"""
    ordered_images = [(title, images[key]) for (title, key) in ordered if key in images]
    return HTML, ordered_images


# ---------- public: PDF builder ----------
def build_pdf_from_images(ordered_images, title="Heart Disease EDA Report") -> bytes:
    """
    Convert a list of (title, data_uri) images to a multi-page PDF.
    Returns PDF bytes.
    """
    from reportlab.lib.pagesizes import A4
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    margin = 36

    # Title page
    c.setFont("Helvetica-Bold", 20)
    c.drawString(margin, height - margin - 10, title)
    c.setFont("Helvetica", 11)
    c.drawString(margin, height - margin - 34, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    c.drawString(margin, height - margin - 50, "Source: dataset.csv")
    c.showPage()

    for sec_title, data_uri in ordered_images:
        # decode base64 data-uri
        b64 = data_uri.split(",")[-1]
        img_bytes = base64.b64decode(b64)
        img_reader = ImageReader(io.BytesIO(img_bytes))
        iw, ih = img_reader.getSize()

        # fit image within margins
        max_w = width - 2*margin
        max_h = height - 2*margin - 24
        scale = min(max_w / iw, max_h / ih)
        w, h = iw * scale, ih * scale

        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, height - margin - 8, sec_title)

        c.drawImage(img_reader,
                    margin,
                    height - margin - 18 - h,
                    width=w, height=h,
                    preserveAspectRatio=True, mask='auto')
        c.showPage()

    c.save()
    buf.seek(0)
    return buf.getvalue()
