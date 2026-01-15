import os
import json
import threading
import time
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import torch
import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from brush import *
from uuid import uuid4
import xml.etree.ElementTree as ET
import cv2
from urllib.parse import unquote



# -----------------------------------------------------------------------------
def extract_labels(label_config_xml):
    root = ET.fromstring(label_config_xml)
    return [tag.attrib["value"] for tag in root.iter("Label")]

# -----------------------------------------------------------------------------
def mask_to_polygons(mask, tolerance=2.0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    h, w = mask.shape

    for cnt in contours:
        if len(cnt) < 6:
            continue
        approx = cv2.approxPolyDP(cnt, tolerance, True)
        poly = [[round(100.0 * p[0][0] / w, 2), round(100.0 * p[0][1] / h, 2)] for p in approx]
        if len(poly) >= 3:
            polygons.append(poly)

    return polygons

# -----------------------------------------------------------------------------
LABEL_STUDIO_UPLOAD_DIR = "/home/amkumar/.local/share/label-studio/media/upload"
MAX_QUEUE_WAIT = 30   # seconds

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

print("Loading SAM3 model...")
model = build_sam3_image_model(bpe_path=bpe_path).cuda().eval()
processor = Sam3Processor(model, confidence_threshold=0.5)
print("SAM3 model loaded!")

app = Flask(__name__)
request_lock = threading.Lock()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/setup", methods=["POST"])
def setup():
    return jsonify({
        "model_name": "sam3",
        "model_version": "1.0",
        "description": "Local SAM3 backend",
        "labels": ["Swimming Pool", "Driveway"]
    })

# -----------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if not request_lock.acquire(timeout=MAX_QUEUE_WAIT):
        return jsonify({"error": "Server busy. Try again later."}), 429

    try:
        payload = request.json
        print("\n===== LS PAYLOAD =====")
        print(json.dumps(payload, indent=2))
        print("=====================\n")

        task = payload["tasks"][0]
        image_url = task["data"]["image"]

        if image_url.startswith("/data/upload"):
            image_path = image_url.replace("/data/upload", LABEL_STUDIO_UPLOAD_DIR)

        elif image_url.startswith("/data/local-files/?d="):
            rel_path = unquote(image_url.split("?d=")[1])

            # handle wrongly-generated absolute path
            if rel_path.startswith("mnt/"):
                image_path = "/" + rel_path
            else:
                image_path = os.path.join("/mnt/harddrive/amkumar", rel_path)

        else:
            image_path = image_url


        if not os.path.exists(image_path):
            return jsonify({"error": f"Image not found: {image_path}"}), 400

        in_image = Image.open(image_path).convert("RGB")
        width, height = in_image.size

        label_config = payload.get("label_config", "")
        labels = extract_labels(label_config)
        is_polygon = "<PolygonLabels" in label_config

        results = []

        for label in labels:
            state = processor.set_image(in_image)
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(state=state, prompt=label)
            masks = state.get("masks", [])
            if masks is None or len(masks) == 0:
                continue

            for m in masks:
                try:
                    m = m.squeeze(0).to(torch.uint8).cpu().numpy()
                    if m.sum() == 0:
                        continue
                    m = m * 255

                    if is_polygon:
                        for poly in mask_to_polygons(m):
                            results.append({
                                "id": str(uuid4())[:4],
                                "type": "polygonlabels",
                                "from_name": "label",
                                "to_name": "image",
                                "value": {
                                    "points": poly,
                                    "polygonlabels": [label]
                                }
                            })
                    else:
                        results.append({
                            "id": str(uuid4())[:4],
                            "type": "brushlabels",
                            "from_name": "tag",
                            "to_name": "image",
                            "value": {
                                "format": "rle",
                                "rle": mask2rle(m),
                                "brushlabels": [label]
                            }
                        })
                except Exception as e:
                    print("MASK FAILURE:", e)
                    continue

        if not results:
            return jsonify({"results": []})

        return jsonify({
            "results": [{
                "result": results,
                "model_version": "sam3",
                "score": 0.99
            }]
        })
    finally:
        request_lock.release()

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004, debug=False)
