import os
import urllib.request
import cv2
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
data_directory = os.path.join(PROJECT_DIR, "Audio_Visual_Data")
images_path = os.path.join(PROJECT_DIR, "image_dataset")

resolution_target = (100, 60)
_face_detector = None


def _get_face_detector():
    global _face_detector
    if _face_detector is not None:
        return _face_detector
    model_path = os.path.join(PROJECT_DIR, "face_detection_yunet_2023mar.onnx")
    if not os.path.exists(model_path):
        url = ("https://github.com/opencv/opencv_zoo/raw/refs/heads/main/"
               "models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
        print("Downloading face detection model (227 KB)...")
        urllib.request.urlretrieve(url, model_path)
    _face_detector = cv2.FaceDetectorYN.create(model_path, "", (320, 320))
    return _face_detector


def _crop_face(frame, margin=0.3):
    h, w = frame.shape[:2]
    detector = _get_face_detector()
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None or len(faces) == 0:
        return None
    x, y, fw, fh = faces[0][:4].astype(int)
    x = max(0, x - int(fw * margin))
    y = max(0, y - int(fh * margin))
    fw = min(frame.shape[1] - x, fw + int(2 * fw * margin))
    fh = min(frame.shape[0] - y, fh + int(2 * fh * margin))
    return frame[y:y + fh, x:x + fw]


def images_from_video(datapath, num_images, crop_face=False):
    os.makedirs(images_path, exist_ok=True)
    folders = [f for f in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, f))]
    folders.sort()

    for folder in folders:
        src_dir = os.path.join(datapath, folder)
        dst_dir = os.path.join(images_path, folder)
        os.makedirs(dst_dir, exist_ok=True)
        video_clips = [f for f in os.listdir(src_dir) if f.endswith('.mp4')]
        video_clips.sort()

        for video_clip in video_clips:
            clip_dst = os.path.join(dst_dir, video_clip)
            os.makedirs(clip_dst, exist_ok=True)

            cap = cv2.VideoCapture(os.path.join(src_dir, video_clip))
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if length < num_images:
                cap.release()
                continue
            indices = set(np.linspace(0, length - 1, num_images, dtype=int))
            frame_idx = 0
            count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx in indices:
                    if crop_face:
                        face = _crop_face(frame)
                        if face is None:
                            frame_idx += 1
                            continue
                        frame = face
                    frame_resized = cv2.resize(frame, resolution_target[::-1], interpolation=cv2.INTER_LINEAR)
                    filename = os.path.join(clip_dst, f"{video_clip}{count + 1}.jpg")
                    cv2.imwrite(filename, frame_resized)
                    count += 1
                frame_idx += 1

            cap.release()
            print(f"Done: {video_clip} ({count} frames)")


if __name__ == '__main__':
    images_from_video(data_directory, 29, crop_face=True)
