import streamlit as st
import cv2
import numpy as np
import collections
import time
from tremor_analyzer import (
    calculate_tremor_frequency,
    calculate_amplitude,
    calculate_severity,
    normalize_amplitude,
    get_hand_size,
    generate_report
)
from handtrackingmodule import handDetector

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Hand Tremor Analyzer",
    page_icon="🖐",
    layout="wide"
)

st.title("🖐 Hand Tremor Analyzer")
st.markdown("*Computer Vision based Parkinson's tremor detection*")

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
baseline     = st.sidebar.slider("Baseline Noise (px)", 5, 50, 20)
patient_name = st.sidebar.text_input("Patient Name", "Patient")

# ── Session state ─────────────────────────────────────────────
if "freq_hist" not in st.session_state:
    st.session_state.freq_hist = []
if "amp_hist" not in st.session_state:
    st.session_state.amp_hist  = []
if "sev_hist" not in st.session_state:
    st.session_state.sev_hist  = []

# ── Camera toggle ─────────────────────────────────────────────
run = st.checkbox("▶️ Start Camera", value=False)

# ── ALL placeholders created ONCE here outside loop ──────────
frame_placeholder = st.empty()

st.markdown("### 📊 Live Metrics")
col1, col2, col3, col4 = st.columns(4)

# Create exactly ONE metric per column — never recreated
freq_placeholder = col1.empty()
amp_placeholder  = col2.empty()
sev_placeholder  = col3.empty()
stat_placeholder = col4.empty()

# Initialize with default values
freq_placeholder.metric("🔁 Frequency", "0 Hz")
amp_placeholder.metric( "📏 Amplitude", "0 px")
sev_placeholder.metric( "⚠️ Severity",  "0/4")
stat_placeholder.metric("📋 Status",    "No Hand")

st.markdown("### 📈 History")
chart_placeholder = st.empty()

st.markdown("---")
report_placeholder = st.empty()

# ── Main loop ─────────────────────────────────────────────────
if run:
    cap              = cv2.VideoCapture(0)
    detector         = handDetector(detectionCon=0.7)
    position_buffer  = collections.deque(maxlen=30)
    hand_size_buffer = collections.deque(maxlen=30)

    st.session_state.freq_hist = []
    st.session_state.amp_hist  = []
    st.session_state.sev_hist  = []

    frame_count = 0

    while run:
        success, img = cap.read()
        if not success:
            st.error("❌ Camera not found.")
            break

        img    = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        frequency      = 0
        amplitude      = 0
        severity_score = 0
        severity_label = "No Hand"

        if len(lmList) != 0:
            TRACKED    = [0, 4, 8, 12, 16, 20]
            coords     = [(lmList[i][1], lmList[i][2])
                          for i in TRACKED if i < len(lmList)]
            centroid_y = np.mean([c[1] for c in coords])
            position_buffer.append(centroid_y)

            hand_size = get_hand_size(lmList)
            if hand_size:
                hand_size_buffer.append(hand_size)

            if len(position_buffer) >= 15:
                avg_hs       = np.mean(hand_size_buffer) if hand_size_buffer else 150
                frequency, _ = calculate_tremor_frequency(list(position_buffer))
                raw_amp      = calculate_amplitude(list(position_buffer))
                amplitude    = normalize_amplitude(raw_amp, avg_hs)
                severity_score, severity_label, _ = calculate_severity(
                    frequency, amplitude, baseline)

                st.session_state.freq_hist.append(frequency)
                st.session_state.amp_hist.append(amplitude)
                st.session_state.sev_hist.append(severity_score)

        # ── Update frame ──────────────────────────────────────
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(img_rgb, use_column_width=True)

        # ── Update metrics (same placeholders, just .metric() again) ──
        freq_placeholder.metric("🔁 Frequency", f"{frequency} Hz")
        amp_placeholder.metric( "📏 Amplitude", f"{amplitude} px")
        sev_placeholder.metric( "⚠️ Severity",  f"{severity_score}/4")
        stat_placeholder.metric("📋 Status",    severity_label)

        # ── Update chart every 15 frames ──────────────────────
        frame_count += 1
        if frame_count % 15 == 0 and len(st.session_state.amp_hist) > 1:
            import pandas as pd
            chart_data = pd.DataFrame({
                "Amplitude (px)":  st.session_state.amp_hist[-100:],
                "Frequency (Hz)":  st.session_state.freq_hist[-100:],
            })
            chart_placeholder.line_chart(chart_data)

        time.sleep(0.03)

    cap.release()

# ── Report section ────────────────────────────────────────────
if len(st.session_state.amp_hist) > 0:
    st.markdown("---")
    st.markdown("### 💾 Session Summary")

    s1, s2, s3 = st.columns(3)
    s1.metric("Avg Frequency",
              f"{round(np.mean(st.session_state.freq_hist), 2)} Hz")
    s2.metric("Avg Amplitude",
              f"{round(np.mean(st.session_state.amp_hist), 2)} px")
    s3.metric("Peak Severity",
              f"{max(st.session_state.sev_hist)}/4")

    if st.button("📄 Generate PDF Report"):
        path = generate_report(
            st.session_state.freq_hist,
            st.session_state.amp_hist,
            st.session_state.sev_hist,
            baseline,
            patient_name
        )
        with open(path, "rb") as f:
            st.download_button(
                label="📥 Download Report",
                data=f,
                file_name=path,
                mime="application/pdf"
            )
        st.success(f"✅ Report ready!")
