#!/bin/bash
set -e
mkdir -p data && cd data

# Mini COCO128
echo "Downloading COCO128 (small subset)..."
wget -c https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip
unzip -n coco128.zip

# Mini UCF101 (10 classes from Kaggle mirror)
echo "Downloading UCF101 Mini..."
wget -c https://github.com/OverLordGoldDragon/UCF101-mini/releases/download/v1.0/UCF101-mini.zip
unzip -n UCF101-mini.zip

cd ..
echo "✅ Mini datasets ready in ./data/"
