# Deep Vision Project 🚀

This project studies **YOLO architectures** and **video classification** with a reproducible, ready-to-run pipeline.  
Everything runs on **mini datasets** so results are visible in minutes.

---

## 🔹 Setup
```bash
git clone <your-repo-url>
cd deep-vision-project
pip install -r requirements.txt
bash get_datasets.sh
🔹 Run YOLO Training

python yolo/train.py
Trains YOLOv8n on COCO128.

Results in runs/detect/train/.

🔹 Run Video Classification
Extract frames:

python video/extract_frames.py
Train CNN+LSTM:


python video/train_cnn_lstm.py
🔹 Results
YOLOv8n fine-tuned on COCO128 → sample detections saved in yolo/results/.

Video CNN+LSTM on UCF101-mini → accuracy & confusion matrix in video/results/.

🔹 Docs
See docs/ for:

Architecture diagrams

SOTA comparison table

Short report (PDF)