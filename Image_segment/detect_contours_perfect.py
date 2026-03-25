import torch
from transformers import pipeline
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import cv2
import numpy as np
from collections import defaultdict

def run_universal_segmentation(img_path, output_path, candidate_labels):
    # 1. Initialize Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load OWL-ViT Tracker (Zero-Shot Text Localization)
    print("Loading OWL-ViT (Semantic Locator)...")
    detector = pipeline(
        model="google/owlvit-large-patch14",
        task="zero-shot-object-detection",
        device=0 if device.type == "cuda" else -1
    )

    # 3. Load SAM (Pixel-Perfect Segmentation)
    print("Loading SAM (vit_h)...")
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    # 4. Load Image
    image_pillow = Image.open(img_path).convert("RGB")
    image_cv2 = cv2.imread(img_path)
    h, w = image_cv2.shape[:2]

    # 5. Semantic Box Detection
    print(f"Searching image for prompts: {candidate_labels}...")
    predictions = detector(image_pillow, candidate_labels=candidate_labels)
    
    # Filter predictions (OWL-ViT scores can be low, 0.05 is a safe threshold to start)
    valid_predictions = [p for p in predictions if p["score"] >= 0.05]
    print(f"Found {len(valid_predictions)} matching objects.")

    if not valid_predictions:
        print("No objects found matching the prompts. Exiting.")
        cv2.imwrite(output_path, image_cv2)
        return

    # 6. SAM High-Fidelity Segmentation (Separated by Label)
    print("Generating pixel-perfect masks via SAM...")
    sam_predictor.set_image(np.array(image_pillow))
    
    label_masks = defaultdict(lambda: np.zeros((h, w), dtype=np.uint8))
    
    for pred in valid_predictions:
        label = pred["label"]
        box = pred["box"]
        # Convert box dict to numpy array [x_min, y_min, x_max, y_max]
        input_box = np.array([box["xmin"], box["ymin"], box["xmax"], box["ymax"]])
        
        # Guide SAM with the semantic bounding box
        masks, _, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        
        # Merge this object's mask into its specific label mask
        label_masks[label] = np.logical_or(label_masks[label], masks[0]).astype(np.uint8) * 255

    # 7. Rendering Professional Outline
    print("Rendering final professional outline for distinct objects...")
    
    result_img = image_cv2.copy()
    scale = 4
    overlay_up = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
    raw_contour_drawn = False
    kernel = np.ones((5, 5), np.uint8)

    for label, mask in label_masks.items():
        print(f"Drawing contours for: {label}")
        # Slight closing to bridge microscopic gaps
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Extract structural contour for this specific object
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > (w * h * 0.005): # Filter tiny insignificant noise
                # Gentle geometric smoothing for professional aesthetic
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.001 * peri, True)
                
                cnt_up = (approx * scale).astype(np.int32)
                # Draw all distinct requested objects in purely RED (0, 0, 255) as required
                cv2.drawContours(overlay_up, [cnt_up], -1, (0, 0, 255), 2 * scale, lineType=cv2.LINE_AA)
                raw_contour_drawn = True

    if raw_contour_drawn:
        overlay_up = cv2.GaussianBlur(overlay_up, (3, 3), 0)
        overlay = cv2.resize(overlay_up, (w, h), interpolation=cv2.INTER_AREA)
        alpha = np.max(overlay, axis=2).astype(float) / 255.0
        alpha = np.stack([alpha, alpha, alpha], axis=2)
        result_img = (result_img.astype(float) * (1 - alpha) + overlay.astype(float) * alpha).astype(np.uint8)

    # Save output
    cv2.imwrite(output_path, result_img)
    print(f"Successfully processed image and universally identified objects. Saved to {output_path}")


if __name__ == "__main__":
    input_file = "pizza.jpg"
    out_file = "result.jpg"
    labels = ["lasagna", "aluminum foil"]
    run_universal_segmentation(input_file, out_file, labels)

