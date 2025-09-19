import cv2, os, glob

def extract_all(videos_dir="data/UCF101-mini", out_dir="data/ucf101_frames", frame_rate=5):
    os.makedirs(out_dir, exist_ok=True)
    videos = glob.glob(os.path.join(videos_dir, "*/*.avi"))
    for vid in videos:
        class_name = os.path.basename(os.path.dirname(vid))
        video_name = os.path.splitext(os.path.basename(vid))[0]
        save_dir = os.path.join(out_dir, class_name, video_name)
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(vid)
        idx, saved = 0, 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if idx % frame_rate == 0:
                cv2.imwrite(os.path.join(save_dir, f"{saved:05d}.jpg"), frame)
                saved += 1
            idx += 1
        cap.release()
        print(f"Extracted {saved} frames from {vid}")

if __name__ == "__main__":
    extract_all()
