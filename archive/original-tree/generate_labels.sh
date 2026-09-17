#!/bin/bash

echo "🚀 Starting label generation pipeline..."
echo ""

# Step 1: Map colors
if [ ! -f "seg_color_map.json" ]; then
    echo "📌 Step 1: Mapping segmentation colors to classes"
    echo "Please click on each object class in order: DRONE, BIRD, PLANE"
    python3 03_map_seg_colors.py --seg_dir dataset/seg/front_center --out seg_color_map.json

    if [ $? -ne 0 ]; then
        echo "❌ Error: Color mapping failed"
        exit 1
    fi
else
    echo "✅ Color map already exists (seg_color_map.json)"
fi

echo ""

# Step 2: Generate bounding boxes
echo "📦 Step 2: Generating bounding boxes from segmentation masks"
python3 04_build_bboxes.py \
    --dataset dataset \
    --cameras front_center bottom_center back_center \
    --seg_map seg_color_map.json \
    --min_area 50 \
    --format both

if [ $? -ne 0 ]; then
    echo "❌ Error: Bounding box generation failed"
    exit 1
fi

echo ""

# Step 3: Generate preview videos
echo "🎬 Step 3: Generating preview videos"
python3 05_preview_overlay.py \
    --dataset dataset \
    --cameras front_center bottom_center back_center \
    --out_dir previews \
    --fps 10

if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Preview generation had issues (non-critical)"
fi

echo ""
echo "✅ Label generation complete!"
echo ""
echo "📊 Output files:"
echo "  - YOLO labels: dataset/labels/<camera>/*.txt"
echo "  - COCO annotations: dataset/annotations_coco.json"
echo "  - Preview videos: previews/<camera>_preview.mp4"