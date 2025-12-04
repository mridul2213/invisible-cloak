import cv2
import numpy as np
import time
CAM = 0
BG_FRAMES = 25
PRE_BLUR = 5        
MORPH_K = 5         
SOFT_BLEND = 21   
MIN_AREA = 1500     


def capture_background(cap, n=BG_FRAMES, wait=2.0):
    print(f"[INFO] Capturing background in {wait}s... please leave frame.")
    time.sleep(wait)
    acc, c = None, 0
    for _ in range(n):
        ret, f = cap.read()
        if not ret: continue
        f = np.flip(f, 1)
        acc = f.astype(np.float32) if acc is None else acc + f.astype(np.float32)
        c += 1
    if c == 0:
        raise RuntimeError("No background frames captured.")
    bg = (acc / c).astype(np.uint8)
    print(f"[INFO] Background captured ({c} frames).")
    return bg

def get_presets():
    return {
        'red': {
            'name':'Red',
            'm1_low': np.array([0,120,70]), 'm1_high': np.array([10,255,255]),
            'm2_low': np.array([170,120,70]), 'm2_high': np.array([180,255,255])
        },
        'blue': {
            'name':'Blue',
            'm1_low': np.array([90,100,100]), 'm1_high': np.array([130,255,255]),
            'm2_low': None, 'm2_high': None
        }
    }

def build_mask(hsv, p):
    hsv_proc = cv2.GaussianBlur(hsv, (PRE_BLUR, PRE_BLUR), 0) if PRE_BLUR>1 else hsv
    m1 = cv2.inRange(hsv_proc, p['m1_low'], p['m1_high'])
    if p['m2_low'] is not None:
        m2 = cv2.inRange(hsv_proc, p['m2_low'], p['m2_high'])
        m = cv2.bitwise_or(m1, m2)
    else:
        m = m1
    return m

def refine_mask(mask):
    k = np.ones((MORPH_K, MORPH_K), np.uint8)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.dilate(m, k, iterations=1)
    # keep largest contour(s) above area
    cnts, _ = cv2.findContours(m.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    for c in cnts:
        if cv2.contourArea(c) >= MIN_AREA:
            cv2.drawContours(out, [c], -1, 255, -1)
    return out

def soft_blend(frame, bg, mask):
    k = SOFT_BLEND if SOFT_BLEND%2==1 else SOFT_BLEND+1
    alpha = cv2.GaussianBlur(mask, (k,k), 0).astype(np.float32)/255.0
    alpha3 = cv2.merge([alpha,alpha,alpha])
    composed = (frame.astype(np.float32)*(1-alpha3) + bg.astype(np.float32)*alpha3).astype(np.uint8)
    return composed

def create_trackbars(win):
    cv2.namedWindow(win)
    cv2.createTrackbar('h1', win, 0, 179, lambda x:None)
    cv2.createTrackbar('s1', win, 120, 255, lambda x:None)
    cv2.createTrackbar('v1', win, 70, 255, lambda x:None)
    cv2.createTrackbar('h2', win, 10, 179, lambda x:None)

def read_trackbars(win):
    h1 = cv2.getTrackbarPos('h1', win)
    s1 = cv2.getTrackbarPos('s1', win)
    v1 = cv2.getTrackbarPos('v1', win)
    h2 = cv2.getTrackbarPos('h2', win)
    low = np.array([h1, s1, v1])
    high = np.array([h2, 255, 255])
    return low, high

def main():
    presets = get_presets()
    choice = input("Cloak color? (r)ed / (b)lue: ").strip().lower()
    p = presets['red'] if choice=='r' else presets['blue']

    cap = cv2.VideoCapture(CAM)
    if not cap.isOpened():
        print("Cannot open camera.")
        return
    # warmup
    for _ in range(5): cap.read()

    background = capture_background(cap)
    track_win = "Tune"
    use_track = False

    print("Controls: q=quit | b=recapture bg | t=toggle trackbars")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = np.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if use_track:
            low, high = read_trackbars(track_win)
            temp = {'m1_low':low, 'm1_high':high, 'm2_low':None, 'm2_high':None}
            mask = build_mask(hsv, temp)
        else:
            mask = build_mask(hsv, p)

        mask = refine_mask(mask)
        out = soft_blend(frame, background, mask)

        cv2.putText(out, f"Cloak: {p['name']}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        cv2.imshow("Invisible Cloak (small)", out)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        if k == ord('b'):
            try:
                background = capture_background(cap)
            except Exception as e:
                print("BG recapture failed:", e)
        if k == ord('t'):
            use_track = not use_track
            if use_track:
                create_trackbars(track_win)
            else:
                cv2.destroyWindow(track_win)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
raise