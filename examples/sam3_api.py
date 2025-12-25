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

        # /data/upload/2/filename.jpg  -> real path
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

        inference_state = processor.set_image(in_image)
        processor.reset_all_prompts(inference_state)

        inference_state = processor.set_text_prompt(state=inference_state, prompt=labels[0])
        print(inference_state.keys())

        masks = inference_state["masks"]

        if len(masks) == 0:
            print("NOT MASK DETECTED")
            return jsonify({"result": []})
        else:
            print("success", len(masks))
            print("shape", masks[0].shape)

        from pycocotools import mask as mask_utils
        
        # mask: shape (1, 800, 800), dtype=bool, device=cuda
        mask = inference_state['masks'][0].squeeze(0)          # (800, 800)
        mask = mask.to(torch.uint8)     # bool -> uint8
        mask = mask.cpu().numpy()       # to numpy
        mask = mask * 255
        rle = mask2rle(mask)

        result = {
            "result": [
                {
                    "id": "sam3_mask",
                    "type": "brushlabels",
                    "from_name": "tag",
                    "to_name": "image",
                    "value": {
                    "format": "rle",
                    "rle": rle,
                    "brushlabels": ["Swiming Pool"]
                    }
                }
            ],
            "score": 0.99,
            "model_version": "sam3-v1"
        }

        #print("result", result)

        return jsonify({"results": [result]})

    finally:
        request_lock.release()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004, debug=False)
