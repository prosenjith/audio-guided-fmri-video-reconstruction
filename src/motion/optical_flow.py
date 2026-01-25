import cv2
import torch

def compute_optical_flow(video_path, resize=(64, 64)):
    cap = cv2.VideoCapture(video_path)
    ret, prev = cap.read()
    if not ret:
        return None

    prev = cv2.resize(prev, resize)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    feats = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        feats.append([mag.mean(), ang.mean()])
        prev_gray = gray

    cap.release()
    return torch.tensor(feats, dtype=torch.float32)
