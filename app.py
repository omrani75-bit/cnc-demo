import io
import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

st.set_page_config(page_title="Greenwich CNC Optimisation Demo", layout="wide")

APP_MODE = st.sidebar.radio("Mode", ["Try sample data", "Upload CSV", "About / Book diagnostic"])

REQUIRED_COLS = [
    "process_type",
    "material",
    "spindle_rpm",
    "feed_mm_min",
    "depth_mm",
    "tool_id",
    "coolant_on",
    "batch_size",
    "scrap"
]

def validate(df: pd.DataFrame):
    return [c for c in REQUIRED_COLS if c not in df.columns]

def demo_predict_scrap_probability(df: pd.DataFrame) -> np.ndarray:
    rpm = df["spindle_rpm"].astype(float).clip(1)
    feed = df["feed_mm_min"].astype(float).clip(1)
    depth = df["depth_mm"].astype(float).clip(0.01)
    coolant = df["coolant_on"].astype(float).clip(0, 1)

    score = (feed / (rpm * 0.8)) + (depth * 0.6) - (coolant * 0.3)
    prob = 1 / (1 + np.exp(-3 * (score - np.median(score))))
    return prob.clip(0.01, 0.99)

def recommend_parameters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["scrap_prob"] = demo_predict_scrap_probability(df)

    out["rec_feed_mm_min"] = np.where(out["scrap_prob"] > 0.7, out["feed_mm_min"] * 0.9, out["feed_mm_min"])
    out["rec_depth_mm"] = np.where(out["scrap_prob"] > 0.7, out["depth_mm"] * 0.9, out["depth_mm"])
    out["note"] = np.where(out["scrap_prob"] > 0.7, "High risk: reduce feed/depth ~10%", "OK")
    return out[["scrap_prob", "rec_feed_mm_min", "rec_depth_mm", "note"]]

def make_pdf_report(df: pd.DataFrame, rec: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 50, "Greenwich Strategy — CNC Optimisation Demo Report")

    c.setFont("Helvetica", 10)
    c.drawString(40, height - 80, f"Rows analysed: {len(df)}")

    avg_risk = float(rec["scrap_prob"].mean())
    c.drawString(40, height - 100, f"Average predicted scrap risk: {avg_risk:.2%}")

    c.drawString(40, height - 130, "Top recommendations (first 10 rows):")
    y = height - 150
    c.setFont("Helvetica", 8)
    for i in range(min(10, len(rec))):
        line = (
            f"Row {i+1}: risk={rec.loc[i,'scrap_prob']:.2%}, "
            f"feed={rec.loc[i,'rec_feed_mm_min']:.0f}, depth={rec.loc[i,'rec_depth_mm']:.2f} — "
            f"{rec.loc[i,'note']}"
        )
        c.drawString(40, y, line)
        y -= 12
        if y < 80:
            c.showPage()
            y = height - 60

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()

st.title("Greenwich CNC Optimisation — Demo (Scrap Risk + Parameter Recommendations)")

if APP_MODE == "Try sample data":
    os.makedirs("sample_data", exist_ok=True)
    sample = st.selectbox("Choose sample dataset", ["cnc_turning_demo.csv", "cnc_milling_demo.csv"])

    path = os.path.join("sample_data", sample)
    if not os.path.exists(path):
        demo = pd.DataFrame({
            "process_type": ["turning"]*20,
            "material": ["AL6061"]*20,
            "spindle_rpm": np.random.randint(1500, 4500, 20),
            "feed_mm_min": np.random.randint(150, 900, 20),
            "depth_mm": np.round(np.random.uniform(0.2, 2.0, 20), 2),
            "tool_id": np.random.choice(["T01","T02","T03"], 20),
            "coolant_on": np.random.choice([0,1], 20),
            "batch_size": np.random.randint(5, 200, 20),
            "scrap": np.random.choice([0,1], 20, p=[0.9,0.1]),
        })
        demo.to_csv(path, index=False)

    df = pd.read_csv(path)

elif APP_MODE == "Upload CSV":
    st.write("Your CSV must contain these columns:")
    st.code(", ".join(REQUIRED_COLS))
    up = st.file_uploader("Upload CSV", type=["csv"])
    if not up:
        st.stop()
    df = pd.read_csv(up)

else:
    st.subheader("What this demo shows")
    st.write("- Predict scrap risk from machining parameters\n- Recommend parameter adjustments\n- Generate a PDF report")
    st.subheader("Book a diagnostic")
    st.write("Replace this text with your real booking link / email.")
    st.stop()

missing = validate(df)
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input data (preview)")
    st.dataframe(df.head(20), use_container_width=True)

rec = recommend_parameters(df)

with col2:
    st.subheader("Risk overview")
    fig = plt.figure()
    plt.hist(rec["scrap_prob"], bins=20)
    plt.xlabel("Predicted scrap probability")
    plt.ylabel("Count")
    st.pyplot(fig, clear_figure=True)

st.subheader("Recommendations")
st.dataframe(rec.join(df, how="left").head(50), use_container_width=True)

st.subheader("ROI quick calculator")
material_cost = st.number_input("Annual material cost (£)", min_value=0.0, value=250000.0, step=10000.0)
improve_pp = st.number_input("Target scrap reduction (percentage points)", min_value=0.0, max_value=50.0, value=1.5, step=0.5)
savings = material_cost * (improve_pp / 100.0)
st.metric("Estimated annual savings (£)", f"{savings:,.0f}")

pdf = make_pdf_report(df, rec)
st.download_button("Download PDF report", data=pdf, file_name="Greenwich_CNC_Demo_Report.pdf", mime="application/pdf")
