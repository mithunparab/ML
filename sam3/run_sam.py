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
    if MODEL is not None: return
    
    if DEVICE == "cuda":
        torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    print(f"Loading SAM3 Model to {DEVICE}...")
    MODEL = build_sam3_image_model().to(DEVICE)
    PROCESSOR = Sam3Processor(MODEL, confidence_threshold=0.3)
    print("Model loaded.")

def process_image(data_source: str, prompt: str = None, box_prompts: list = None, box_labels: list = None, output_path: str = "", is_base64: bool = False):
    initialize_model()

    if is_base64:
        try:
            image_bytes = base64.b64decode(data_source)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Image Decode Error: {e}")
    else:
        try:
            response = requests.get(data_source)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
             raise ValueError(f"Image Download Error: {e}")

    print(f"Setting image: {image.size}")
    inference_state = PROCESSOR.set_image(image)
    
    orig_w, orig_h = image.size

    
    if prompt:
        print(f"Applying text prompt: {prompt}")
        inference_state = PROCESSOR.set_text_prompt(state=inference_state, prompt=prompt)

    if box_prompts and len(box_prompts) > 0:
        print(f"Applying {len(box_prompts)} box prompts...")
        
        if not box_labels:
            box_labels = [1] * len(box_prompts)

        for i, box in enumerate(box_prompts):
            x1, y1, x2, y2 = box
            label_int = box_labels[i]
            is_positive = (label_int == 1)

            if x2 <= 1.0 and y2 <= 1.0:
                x1 *= orig_w
                x2 *= orig_w
                y1 *= orig_h
                y2 *= orig_h

            w_px = x2 - x1
            h_px = y2 - y1
            
            cx_px = x1 + (w_px / 2.0)
            cy_px = y1 + (h_px / 2.0)

            cx_norm = cx_px / orig_w
            cy_norm = cy_px / orig_h
            w_norm = w_px / orig_w
            h_norm = h_px / orig_h
            
            geometric_box = [cx_norm, cy_norm, w_norm, h_norm]
            print(f"Adding Geometric Prompt: {geometric_box}, Positive: {is_positive}")

            inference_state = PROCESSOR.add_geometric_prompt(
                box=geometric_box,
                label=is_positive,
                state=inference_state
            )

    if "masks" not in inference_state or inference_state["masks"] is None:
        return None, None

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.to(torch.float32).cpu().numpy()
        return x

    xyxy_res = to_numpy(inference_state["boxes"])
    scores = to_numpy(inference_state["scores"])
    
    masks_res = inference_state["masks"]
    if isinstance(masks_res, torch.Tensor):
        masks_res = masks_res.to(torch.bool)
        if masks_res.ndim == 4:
            masks_res = masks_res.reshape(masks_res.shape[0] * masks_res.shape[1], masks_res.shape[2], masks_res.shape[3])
        masks_res = masks_res.cpu().numpy()

    mask_filter = scores > CONFIDENCE_THRESHOLD
    masks = masks_res[mask_filter]
    xyxy = xyxy_res[mask_filter]

    print(f"Detected {len(masks)} objects after threshold.")

    if len(masks) > 0:
        if masks.ndim == 3:
             combined_mask = np.logical_or.reduce(masks)
        else:
             combined_mask = masks

        image_np = np.array(image)

        full_mask_uint8 = (combined_mask * 255).astype(np.uint8)
        full_mask_image = Image.fromarray(full_mask_uint8, 'L')
        buffer_mask = BytesIO()
        full_mask_image.save(buffer_mask, format="PNG")
        mask_base64 = base64.b64encode(buffer_mask.getvalue()).decode('utf-8')

        if len(xyxy) > 0 and len(xyxy.shape) > 1:
            x_min = int(np.min(xyxy[:, 0]))
            y_min = int(np.min(xyxy[:, 1]))
            x_max = int(np.max(xyxy[:, 2]))
            y_max = int(np.max(xyxy[:, 3]))
        else:
            rows = np.any(combined_mask, axis=1)
            cols = np.any(combined_mask, axis=0)
            if not np.any(rows): return None, None
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

        H, W, _ = image_np.shape
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(W, x_max), min(H, y_max)
        
        cropped_mask = combined_mask[y_min:y_max, x_min:x_max] 
        cropped_original_np = image_np[y_min:y_max, x_min:x_max]
        
        alpha_channel = (cropped_mask * 255).astype(np.uint8)
        if alpha_channel.shape[:2] != cropped_original_np.shape[:2]:
            alpha_pil = Image.fromarray(alpha_channel).resize((cropped_original_np.shape[1], cropped_original_np.shape[0]))
            alpha_channel = np.array(alpha_pil)

        rgba_image_np = np.dstack((cropped_original_np, alpha_channel))
        transparent_image = Image.fromarray(rgba_image_np, 'RGBA')

        buffer_img = BytesIO()
        transparent_image.save(buffer_img, format="PNG")
        output_base64 = base64.b64encode(buffer_img.getvalue()).decode('utf-8')

        return output_base64, mask_base64

    return None, None