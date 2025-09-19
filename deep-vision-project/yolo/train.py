from ultralytics import YOLO

def main():
    # Use pretrained YOLOv8n and fine-tune on COCO128
    model = YOLO("yolov8n.pt")
    model.train(
        data="yolo/data.yaml",
        epochs=3,
        imgsz=640,
        batch=8
    )
    model.val()
    model.predict(source="data/coco128/images/train2017", save=True)

if __name__ == "__main__":
    main()
