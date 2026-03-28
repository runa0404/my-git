from flask import Flask, request, jsonify
import clip
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

app = Flask(__name__)

torch.set_num_threads(2)

device = "cuda" if torch.cuda.is_available() else "cpu" # or "cuda" if available
model, preprocess = clip.load("ViT-B/32", device=device)

def normalize_note(image):
    w, h = image.size
    new_w = 400
    new_h = int((h / w) * new_w)
    return image.resize((new_w, new_h))


def center_crop(image):
    w, h = image.size
    crop_w = int(w * 0.9)
    crop_h = int(h * 0.9)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return image.crop((left, top, left + crop_w, top + crop_h))


def mask_serial(image):
    img = np.array(image)
    h, w, _ = img.shape

    img[int(h*0.75):h, 0:int(w*0.35)] = 200
    img[0:int(h*0.25), int(w*0.65):w] = 200

    return Image.fromarray(img)
	
def generate_variants(image, is_coin=True):
    variants = []

    angles = [0, 90, 180, 270] if is_coin else [0, 180]

    for angle in angles:
        rotated = image.rotate(angle, expand=True)

        variants.append(rotated)

        enhancer = ImageEnhance.Brightness(rotated)
        variants.append(enhancer.enhance(1.1))
        variants.append(enhancer.enhance(0.9))

    return variants


@app.route('/clip', methods=['POST'])
def get_embedding():

    file = request.files['image']
    base_image = Image.open(file.stream).convert("RGB")

    # simple coin detection
    w, h = base_image.size
    is_coin = 0.9 <= (w / h) <= 1.1

    if not is_coin:
        base_image = normalize_note(base_image)
        base_image = center_crop(base_image)
        base_image = mask_serial(base_image)

    # generate variants
    variants = generate_variants(base_image, is_coin)

    # ?? BATCH PROCESSING
    image_tensors = torch.stack([
        preprocess(img) for img in variants
    ]).to(device)

    with torch.no_grad():
        features = model.encode_image(image_tensors)

    features = features.cpu().numpy()

    # normalize all at once
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / norms

    embeddings = features.tolist()

    return jsonify({"embeddings": embeddings})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)