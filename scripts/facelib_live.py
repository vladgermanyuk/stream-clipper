"""Face boxes with OpenCV YuNet (models/face_detection_yunet.onnx). Not thread-safe: one call at a time."""
import os, threading, cv2
M = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "face_detection_yunet.onnx")
_lock = threading.Lock()


def faces_boxes(img, thr=0.8):
    h, w = img.shape[:2]
    with _lock:
        d = cv2.FaceDetectorYN.create(M, "", (w, h), thr, 0.3, 50)
        _, f = d.detect(img)
    return [] if f is None else [(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in f]
