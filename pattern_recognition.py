import cv2
import numpy as np
import mss
import tensorflow as tf

# Define screen capture region (Modify for your screen)
monitor = {"top": 200, "left": 100, "width": 1200, "height": 600}
sct = mss.mss()

def capture_screen():
    """ Captures a screenshot of the trading platform """
    screenshot = sct.grab(monitor)
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def detect_candlestick_patterns(img):
    """ Detects candlestick patterns based on shape analysis """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 2 * w:  # Long candles (Hammer, Shooting Star)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            print(f"Long candlestick detected at {x}, {y} -> Height: {h}, Width: {w}")

    return img

while True:
    img = capture_screen()
    processed_img = detect_candlestick_patterns(img)

    cv2.imshow("Pattern Detector", processed_img)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
