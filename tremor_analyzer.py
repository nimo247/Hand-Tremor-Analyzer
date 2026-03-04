import cv2
import time
import numpy as np
import collections
import os
import datetime
from scipy.fft import fft, fftfreq
from handtrackingmodule import handDetector
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import matplotlib.pyplot as plt


# ============================================================
# STEP 1 — PDF Report Generator
# ============================================================
def generate_report(frequency_history, amplitude_history,
                    severity_history, baseline_noise, patient_name="Patient"):

    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename   = f"tremor_report_{timestamp}.pdf"
    graph_path = f"graph_{timestamp}.png"

    # --- Save graph as image ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    ax1.plot(amplitude_history, color='blue', linewidth=1.5)
    ax1.set_title('Tremor Amplitude Over Time')
    ax1.set_ylabel('Amplitude (normalized px)')
    ax1.set_xlabel('Frames')
    ax1.axhline(y=baseline_noise, color='red',
                linestyle='--', label='Baseline')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(frequency_history, color='orange', linewidth=1.5)
    ax2.set_title('Tremor Frequency Over Time')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Frames')
    ax2.axhline(y=4, color='red', linestyle='--',
                label='Parkinson threshold (4Hz)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

    # --- Build PDF ---
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Header bar
    c.setFillColor(colors.HexColor('#1a1a2e'))
    c.rect(0, height - 80, width, 80, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(40, height - 50, "Hand Tremor Analysis Report")
    c.setFont("Helvetica", 12)
    c.drawString(40, height - 70,
                 f"Generated: {datetime.datetime.now().strftime('%B %d, %Y %H:%M')}")

    # Patient name
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 120, f"Patient: {patient_name}")

    # Summary stats
    avg_freq = round(np.mean(frequency_history), 2) if frequency_history else 0
    avg_amp  = round(np.mean(amplitude_history), 2) if amplitude_history else 0
    max_sev  = max(severity_history) if severity_history else 0

    severity_labels = {0: "No Tremor", 1: "Minimal",
                       2: "Mild",      3: "Moderate", 4: "Severe"}

    box_data = [
        ("Avg Frequency",  f"{avg_freq} Hz"),
        ("Avg Amplitude",  f"{avg_amp} px"),
        ("Peak Severity",  severity_labels[max_sev]),
        ("Baseline Noise", f"{round(baseline_noise, 1)} px"),
    ]

    box_y = height - 180
    box_w = (width - 80) / 4

    for i, (label, value) in enumerate(box_data):
        bx = 40 + i * box_w
        c.setFillColor(colors.HexColor('#f0f0f0'))
        c.rect(bx, box_y - 50, box_w - 10, 60, fill=True, stroke=False)
        c.setFillColor(colors.HexColor('#333333'))
        c.setFont("Helvetica", 9)
        c.drawString(bx + 8, box_y + 2, label)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(bx + 8, box_y - 25, value)

    # Graph image
    c.drawImage(graph_path, 40, height - 530, width=520, height=280)

    # Recommendation
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(colors.black)
    c.drawString(40, height - 560, "Recommendation:")
    c.setFont("Helvetica", 11)

    if max_sev == 0:
        rec = "No tremor detected. Hand movement appears normal."
    elif max_sev == 1:
        rec = "Minimal tremor detected. Monitor periodically."
    elif max_sev == 2:
        rec = "Mild tremor detected. Consider consulting a neurologist."
    elif max_sev == 3:
        rec = "Moderate tremor detected. Medical consultation strongly recommended."
    else:
        rec = "Severe tremor detected. Immediate medical attention recommended."

    c.drawString(40, height - 580, rec)

    # Disclaimer
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(colors.grey)
    c.drawString(40, 40,
                 "This report is for informational purposes only "
                 "and does not constitute medical advice.")

    c.save()
    os.remove(graph_path)

    print(f"✅ Report saved: {filename}")
    return filename


# ============================================================
# CORE FUNCTIONS
# ============================================================
def calculate_tremor_frequency(positions, fps=10):
    if len(positions) < 15:
        return 0, 0
    signal = np.array(positions)
    signal = signal - np.mean(signal)
    n = len(signal)
    frequencies = fftfreq(n, d=1 / fps)
    fft_values  = np.abs(fft(signal))
    pos_idx     = np.where((frequencies > 1) & (frequencies < 12))
    rel_freqs   = frequencies[pos_idx]
    rel_fft     = fft_values[pos_idx]
    if len(rel_fft) == 0:
        return 0, 0
    dom_idx    = np.argmax(rel_fft)
    return round(abs(rel_freqs[dom_idx]), 2), round(rel_fft[dom_idx], 2)


def calculate_amplitude(positions):
    if len(positions) < 10:
        return 0
    signal = np.array(positions)
    return round(float(np.max(signal) - np.min(signal)), 2)


def get_hand_size(lmList):
    if len(lmList) < 10:
        return None
    x1, y1 = lmList[0][1], lmList[0][2]
    x2, y2 = lmList[9][1], lmList[9][2]
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def normalize_amplitude(amplitude, hand_size, reference_size=150):
    if hand_size is None or hand_size == 0:
        return amplitude
    return round(amplitude * (reference_size / hand_size), 2)


def calculate_severity(frequency, amplitude, baseline_noise):
    adj = max(0, amplitude - baseline_noise)
    if adj < 10:
        return 0, "No Tremor", (0, 255, 0)
    elif adj < 25 and frequency < 3:
        return 1, "Minimal",   (0, 255, 0)
    elif adj < 50 and frequency < 5:
        return 2, "Mild",      (0, 255, 255)
    elif adj < 80 and frequency < 7:
        return 3, "Moderate",  (0, 165, 255)
    else:
        return 4, "Severe",    (0, 0, 255)


# ============================================================
# STEP 2 — Live Graph drawn inside OpenCV window
# ============================================================
def draw_live_graph(img, history, label, color, x, y, w, h, max_val=100):
    cv2.rectangle(img, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 100), 1)
    cv2.putText(img, label, (x + 5, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    if len(history) >= 2:
        data = list(history)[-w:]
        for i in range(1, len(data)):
            v1 = int(np.clip(data[i - 1] / max_val, 0, 1) * (h - 20))
            v2 = int(np.clip(data[i]     / max_val, 0, 1) * (h - 20))
            p1 = (x + int((i - 1) * w / len(data)), y + h - 5 - v1)
            p2 = (x + int(i       * w / len(data)), y + h - 5 - v2)
            cv2.line(img, p1, p2, color, 1)
        cv2.putText(img, str(round(history[-1], 1)),
                    (x + w - 50, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return img


# ============================================================
# UI Panel
# ============================================================
def draw_ui(img, frequency, amplitude, severity_score,
            severity_label, color, baseline_noise, hand_size=None):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (320, 265), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    cv2.putText(img, f"Frequency:  {frequency} Hz",            (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Amplitude:  {amplitude} px",             (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Baseline:   {round(baseline_noise,1)} px",(10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    hs_text = f"{round(hand_size,1)} px" if hand_size else "N/A"
    cv2.putText(img, f"Hand Size:  {hs_text}",                  (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(img, f"Severity:   {severity_score}/4",         (10, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(img, f"Status:     {severity_label}",           (10, 228),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img


# ============================================================
# Calibration
# ============================================================
def calibrate_baseline(cap, detector, seconds=3):
    baseline_positions = []
    start_time = time.time()

    while time.time() - start_time < seconds:
        success, img = cap.read()
        if not success:
            continue

        img    = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        remaining = int(seconds - (time.time() - start_time)) + 1
        overlay   = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        cv2.putText(img, "CALIBRATING - Hold hand still",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(img, f"Starting in {remaining} seconds...",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Hand Tremor Analyzer", img)
        cv2.waitKey(1)

        if len(lmList) != 0:
            TRACKED = [0, 4, 8, 12, 16, 20]
            coords  = [(lmList[i][1], lmList[i][2])
                       for i in TRACKED if i < len(lmList)]
            baseline_positions.append(np.mean([c[1] for c in coords]))

    if len(baseline_positions) > 10:
        noise = (np.percentile(baseline_positions, 90) -
                 np.percentile(baseline_positions, 10))
        noise = min(noise, 30)
        print(f"✅ Calibration complete! Baseline noise: {noise}px")
        return noise

    print("⚠️  Hand not detected during calibration. Using default.")
    return 20


# ============================================================
# MAIN LOOP
# ============================================================
def main():
    cap              = cv2.VideoCapture(0)
    detector         = handDetector(detectionCon=0.7)
    position_buffer  = collections.deque(maxlen=30)
    hand_size_buffer = collections.deque(maxlen=30)
    pTime            = 0

    frequency_history = []
    amplitude_history = []
    severity_history  = []

    GRAPH_W = 300
    GRAPH_H = 100

    # Calibrate first
    baseline_noise = calibrate_baseline(cap, detector, seconds=3)

    while True:
        success, img = cap.read()
        if not success:
            break

        img    = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        frequency      = 0
        amplitude      = 0
        severity_score = 0
        severity_label = "No Hand"
        color          = (255, 255, 255)

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
                severity_score, severity_label, color = calculate_severity(
                    frequency, amplitude, baseline_noise)

                frequency_history.append(frequency)
                amplitude_history.append(amplitude)
                severity_history.append(severity_score)

        # FPS
        cTime = time.time()
        fps   = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
        pTime = cTime

        # Draw UI panel
        avg_hs_display = np.mean(hand_size_buffer) if hand_size_buffer else None
        img = draw_ui(img, frequency, amplitude, severity_score,
                      severity_label, color, baseline_noise, avg_hs_display)

        cv2.putText(img, f"FPS: {int(fps)}", (10, 255),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Draw live graphs at bottom
        h_img = img.shape[0]
        img = draw_live_graph(img, amplitude_history,
                              "Amplitude", (100, 100, 255),
                              0, h_img - (GRAPH_H * 2 + 10),
                              GRAPH_W, GRAPH_H, max_val=120)

        img = draw_live_graph(img, frequency_history,
                              "Frequency Hz", (100, 200, 255),
                              0, h_img - GRAPH_H,
                              GRAPH_W, GRAPH_H, max_val=12)

        # Bottom hint bar
        cv2.putText(img, "S=Save Report | R=Recalibrate | Q=Quit",
                    (10, h_img - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        cv2.imshow("Hand Tremor Analyzer", img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('r'):
            position_buffer.clear()
            hand_size_buffer.clear()
            frequency_history.clear()
            amplitude_history.clear()
            severity_history.clear()
            baseline_noise = calibrate_baseline(cap, detector, seconds=3)

        elif key == ord('s'):
            if len(frequency_history) > 0:
                patient_name = input("Enter patient name: ")
                generate_report(frequency_history, amplitude_history,
                                severity_history, baseline_noise, patient_name)
            else:
                print("⚠️  No data yet. Keep hand in frame first.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
