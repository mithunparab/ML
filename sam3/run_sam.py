import os
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import base64

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

MODEL = None
PROCESSOR = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIDENCE_THRESHOLD = 0.45

def initialize_model():
    global MODEL, PROCESSOR

    if MODEL is not None:
        return
    
    if DEVICE == "cuda":
        torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    MODEL = build_sam3_image_model().to(DEVICE)
    PROCESSOR = Sam3Processor(MODEL, confidence_threshold=0.3)

def from_sam(sam_result: dict) -> tuple:
    xyxy = sam_result["boxes"].to(torch.float32).cpu().numpy()
    confidence = sam_result["scores"].to(torch.float32).cpu().numpy()

    mask = sam_result["masks"].to(torch.bool)
    mask = mask.reshape(mask.shape[0], mask.shape[2], mask.shape[3]).cpu().numpy()

    return xyxy, confidence, mask

def process_image(image_data: str, prompt: str, output_path: str, is_base64: bool = False):
    initialize_model()

    if is_base64:
        print("Decoding Base64 image data...")
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            print(f"Error decoding Base64 image: {e}")
            return None
    else:
        print(f"Downloading image from URL: {image_data}")
        try:
            response = requests.get(image_data)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            print(f"Error downloading image from URL: {e}")
            return None

    image_np = np.array(image)

    inference_state = PROCESSOR.set_image(image)
    inference_state = PROCESSOR.set_text_prompt(state=inference_state, prompt=prompt)
    
    xyxy, confidence, masks = from_sam(sam_result=inference_state)

    confidence_mask = confidence > CONFIDENCE_THRESHOLD
    xyxy = xyxy[confidence_mask]
    masks = masks[confidence_mask]

    print(f"Detected {len(masks)} objects with confidence > {CONFIDENCE_THRESHOLD} for prompt: {prompt}")

    if len(masks) > 0:
        combined_mask = np.logical_or.reduce(masks)

        x_min = int(np.min(xyxy[:, 0]))
        y_min = int(np.min(xyxy[:, 1]))
        x_max = int(np.max(xyxy[:, 2]))
        y_max = int(np.max(xyxy[:, 3]))
        
        H, W, _ = image_np.shape
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(W, x_max)
        y_max = min(H, y_max)
        
        cropped_mask = combined_mask[y_min:y_max, x_min:x_max] 
        cropped_original_np = image_np[y_min:y_max, x_min:x_max]
        
        alpha_channel = (cropped_mask * 255).astype(np.uint8)
        rgba_image_np = np.dstack((cropped_original_np, alpha_channel))
        
        transparent_image = Image.fromarray(rgba_image_np, 'RGBA')

        # Save to buffer and encode to Base64 for API return
        buffer = BytesIO()
        transparent_image.save(buffer, format="PNG")
        output_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Save to disk for local runpod setup compatibility
        transparent_image.save(output_path)
        print(f"Saved local file to {output_path}")

        return output_base64

    else:
        return None