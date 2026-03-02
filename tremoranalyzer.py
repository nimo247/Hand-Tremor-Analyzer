import cv2
import time
import numpy as np
import collections
from scipy.fft import fft, fftfreq
from handtrackingmodule import handDetector


# Tremor Frequency Function
def calculate_tremor_frequency(positions, fps=10):
    if len(positions) < 15:
        return 0, 0
    signal = np.array(positions)
    signal = signal - np.mean(signal)
    n = len(signal)
    frequencies = fftfreq(n, d=1/fps)
    fft_values = np.abs(fft(signal))
    positive_freq_idx = np.where((frequencies > 1) & (frequencies < 12))
    relevant_freqs = frequencies[positive_freq_idx]
    relevant_fft = fft_values[positive_freq_idx]
    if len(relevant_fft) == 0:
        return 0, 0
    dominant_idx = np.argmax(relevant_fft)
    dominant_freq = relevant_freqs[dominant_idx]
    dominant_power = relevant_fft[dominant_idx]
    return round(abs(dominant_freq), 2), round(dominant_power, 2)


# Amplitude Function
def calculate_amplitude(positions):
    if len(positions) < 10:
        return 0
    signal = np.array(positions)
    amplitude = np.max(signal) - np.min(signal)
    return round(amplitude, 2)

def get_hand_size(lmList):
    # Distance between wrist (0) and middle finger base (9)
    if len(lmList) < 10:
        return None
    x1, y1 = lmList[0][1], lmList[0][2]
    x2, y2 = lmList[9][1], lmList[9][2]
    hand_size = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return hand_size

def normalize_amplitude(amplitude, hand_size, reference_size=150):
    # Converts amplitude to distance-independent score
    # reference_size = typical hand size when at normal distance
    if hand_size is None or hand_size == 0:
        return amplitude
    scale_factor = reference_size / hand_size
    return round(amplitude * scale_factor, 2)


# Calibration Function
def calibrate_baseline(cap, detector, seconds=3):
    baseline_positions = []
    start_time = time.time()

    while time.time() - start_time < seconds:
        success, img = cap.read()
        if not success:
            continue

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        remaining = int(seconds - (time.time() - start_time)) + 1

        # Dark overlay for calibration screen
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        cv2.putText(img, "CALIBRATING - Hold hand still",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(img, f"Starting in {remaining} seconds...",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Hand Tremor Analyzer", img)
        cv2.waitKey(1)

        if len(lmList) != 0:
            TRACKED_POINTS = [0, 4, 8, 12, 16, 20]
            tracked_coords = [(lmList[i][1], lmList[i][2]) for i in TRACKED_POINTS if i < len(lmList)]
            centroid_y = np.mean([c[1] for c in tracked_coords])
            baseline_positions.append(centroid_y)

    if len(baseline_positions) > 10:
        # Use 90th percentile instead of max-min to ignore outlier movements
        baseline_noise = np.percentile(baseline_positions, 90) - np.percentile(baseline_positions, 10)
        baseline_noise = min(baseline_noise, 30)  # Cap at 30px so real tremors aren't hidden
        print(f"✅ Calibration complete! Baseline noise: {baseline_noise}px")
        return baseline_noise

    print("⚠️ Hand not detected during calibration. Using default baseline.")
    return 20  # safe default


# Severity Score Function (now uses baseline)
def calculate_severity(frequency, amplitude, baseline_noise):
    # Subtract natural hand noise from amplitude
    adjusted_amplitude = max(0, amplitude - baseline_noise)

    if adjusted_amplitude < 10:
        return 0, "No Tremor", (0, 255, 0)
    elif adjusted_amplitude < 25 and frequency < 3:
        return 1, "Minimal", (0, 255, 0)
    elif adjusted_amplitude < 50 and frequency < 5:
        return 2, "Mild", (0, 255, 255)
    elif adjusted_amplitude < 80 and frequency < 7:
        return 3, "Moderate", (0, 165, 255)
    else:
        return 4, "Severe", (0, 0, 255)


# UI Display Function
def draw_ui(img, frequency, amplitude, severity_score, severity_label, color, baseline_noise, hand_size=None):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (320, 260), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    cv2.putText(img, f"Frequency:  {frequency} Hz", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Amplitude:  {amplitude} px", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Baseline:   {round(baseline_noise, 1)} px", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    hand_size_text = f"{round(hand_size, 1)} px" if hand_size else "N/A"
    cv2.putText(img, f"Hand Size:  {hand_size_text}", (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(img, f"Severity:   {severity_score}/4", (10, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(img, f"Status:     {severity_label}", (10, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img


# Main Loop
def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector(detectionCon=0.7)
    position_buffer = collections.deque(maxlen=30)  # 30 frames at 10fps = 3 seconds
    hand_size_buffer = collections.deque(maxlen=30)  # track hand size over time
    pTime = 0

    # ✅ Calibrate baseline before starting
    baseline_noise = calibrate_baseline(cap, detector, seconds=3)

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        frequency = amplitude = severity_score = 0
        severity_label = "No Hand"
        color = (255, 255, 255)

        if len(lmList) != 0:
            # Track ALL fingertips + wrist for complete hand movement detection
            # Landmark IDs: 0=wrist, 4=thumb, 8=index, 12=middle, 16=ring, 20=pinky
            TRACKED_POINTS = [0, 4, 8, 12, 16, 20]

            # Calculate centroid (center point) of all tracked landmarks
            tracked_coords = [(lmList[i][1], lmList[i][2]) for i in TRACKED_POINTS if i < len(lmList)]
            centroid_y = np.mean([c[1] for c in tracked_coords])
            position_buffer.append(centroid_y)

            # Track hand size for distance normalization
            hand_size = get_hand_size(lmList)
            if hand_size:
                hand_size_buffer.append(hand_size)

            if len(position_buffer) >= 15:
                avg_hand_size = np.mean(hand_size_buffer) if hand_size_buffer else 150
                frequency, power = calculate_tremor_frequency(list(position_buffer))
                raw_amplitude = calculate_amplitude(list(position_buffer))
                amplitude = normalize_amplitude(raw_amplitude, avg_hand_size)
                severity_score, severity_label, color = calculate_severity(
                    frequency, amplitude, baseline_noise
                )

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
        pTime = cTime

        # ✅ Pass baseline_noise to UI so it shows on screen
        img = draw_ui(img, frequency, amplitude, severity_score,
                      severity_label, color, baseline_noise,
                      hand_size=np.mean(hand_size_buffer) if hand_size_buffer else None)

        cv2.putText(img, f"FPS: {int(fps)}", (10, 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Recalibrate hint
        cv2.putText(img, "Press R to recalibrate | Q to quit",
                    (10, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("Hand Tremor Analyzer", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            position_buffer.clear()
            hand_size_buffer.clear()
            baseline_noise = calibrate_baseline(cap, detector, seconds=3)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
