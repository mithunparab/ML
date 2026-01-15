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
            print(f"Error decoding Base64 image: {e}")
            return None, None
    else:
        try:
            response = requests.get(data_source)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            print(f"Error downloading image from URL: {e}")
            return None, None

    image_np = np.array(image)
    masks = []
    xyxy = []

    if box_prompts and len(box_prompts) > 0:
        print(f"Processing with {len(box_prompts)} input boxes...")
        
        formatted_boxes = [box_prompts] 
        
        if box_labels:
             formatted_labels = [box_labels]
        else:
             formatted_labels = [[1] * len(box_prompts)]

        inputs = PROCESSOR(
            images=image,
            input_boxes=formatted_boxes,
            input_boxes_labels=formatted_labels,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = MODEL(**inputs)

        results = PROCESSOR.post_process_instance_segmentation(
            outputs,
            threshold=CONFIDENCE_THRESHOLD,
            mask_threshold=0.5, 
            target_sizes=inputs.get("original_sizes").tolist()
        )[0] 


        if "masks" in results:
            masks_tensor = results["masks"]
            if isinstance(masks_tensor, torch.Tensor):
                 masks = masks_tensor.cpu().numpy()
            else:
                 masks = masks_tensor
            
            # Ensure boolean
            masks = masks > 0.5 

        if "boxes" in results:
            xyxy = results["boxes"].cpu().numpy() if isinstance(results["boxes"], torch.Tensor) else results["boxes"]

    elif prompt:
        print(f"Processing with text prompt: {prompt}")
        inference_state = PROCESSOR.set_image(image)
        inference_state = PROCESSOR.set_text_prompt(state=inference_state, prompt=prompt)
        
        xyxy_res = inference_state["boxes"].to(torch.float32).cpu().numpy()
        confidence_res = inference_state["scores"].to(torch.float32).cpu().numpy()
        masks_res = inference_state["masks"].to(torch.bool)
        masks_res = masks_res.reshape(masks_res.shape[0], masks_res.shape[2], masks_res.shape[3]).cpu().numpy()

        conf_mask = confidence_res > CONFIDENCE_THRESHOLD
        xyxy = xyxy_res[conf_mask]
        masks = masks_res[conf_mask]

    else:
        print("No prompts provided.")
        return None, None

    print(f"Detected {len(masks)} mask objects.")

    if len(masks) > 0:
        if masks.ndim == 3:
             combined_mask = np.logical_or.reduce(masks)
        else:
             combined_mask = masks 

        full_mask_uint8 = (combined_mask * 255).astype(np.uint8)
        full_mask_image = Image.fromarray(full_mask_uint8, 'L')
        
        buffer_mask = BytesIO()
        full_mask_image.save(buffer_mask, format="PNG")
        mask_base64 = base64.b64encode(buffer_mask.getvalue()).decode('utf-8')

        if len(xyxy) > 0:
            x_min = int(np.min(xyxy[:, 0]))
            y_min = int(np.min(xyxy[:, 1]))
            x_max = int(np.max(xyxy[:, 2]))
            y_max = int(np.max(xyxy[:, 3]))
        else:
            rows = np.any(combined_mask, axis=1)
            cols = np.any(combined_mask, axis=0)
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

    else:
        return None, None