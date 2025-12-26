import os
import json
import threading
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

def extract_labels(label_config_xml):
    root = ET.fromstring(label_config_xml)
    labels = []
    for tag in root.iter("Label"):
        labels.append(tag.attrib["value"])
    return labels


# -----------------------------------------------------------------------------
# LABEL STUDIO MEDIA DIRECTORY  (IMPORTANT)
# -----------------------------------------------------------------------------
LABEL_STUDIO_UPLOAD_DIR = "/home/amkumar/.local/share/label-studio/media/upload"

# -----------------------------------------------------------------------------
# GPU Setup
# -----------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

print("Loading SAM3 model...")
model = build_sam3_image_model(bpe_path=bpe_path).cuda().eval()
processor = Sam3Processor(model, confidence_threshold=0.5)
print("SAM3 model loaded!")

# -----------------------------------------------------------------------------
# Flask App + Single Request Lock
# -----------------------------------------------------------------------------
app = Flask(__name__)
request_lock = threading.Lock()

# -----------------------------------------------------------------------------
# Endpoints required by Label Studio
# -----------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/setup", methods=["POST"])
def setup():
    return jsonify({
        "model_name": "sam3",
        "model_version": "1.0",
        "description": "Local SAM3 backend",
        "labels": ["Swiming Pool", "Driveway"]
    })

from PIL import Image  # make sure this is at the top
import io
import base64
import numpy as np

# -----------------------------------------------------------------------------
# /predict
# -----------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if not request_lock.acquire(blocking=False):
        return jsonify({"error": "Server busy"}), 429

    try:
        payload = request.json
        print("\n===== LS PAYLOAD =====")
        print(json.dumps(payload, indent=2))
        print("=====================\n")

        task = payload["tasks"][0]
        image_url = task["data"]["image"]
        if image_url.startswith("/data/upload"):
            image_path = image_url.replace("/data/upload", LABEL_STUDIO_UPLOAD_DIR)
        else:
            image_path = image_url

        if not os.path.exists(image_path):
            return jsonify({"error": f"Image not found: {image_path}"}), 400

        in_image = Image.open(image_path).convert("RGB")
        width, height = in_image.size

        # ------------------ GET SELECTED LABEL ------------------
        label_config = payload.get("label_config")
        labels = extract_labels(label_config)

        print("SAM3 LABEL PROMPTS:", labels)

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
                    rle = mask2rle(m)
                    results.append({ "id": str(uuid4())[:4], 
                                    "type": "brushlabels", 
                                    "from_name": "tag", 
                                    "to_name": "image", 
                                    "value": { "format": "rle", "rle": rle, "brushlabels": [label] } 
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
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004, debug=False)
