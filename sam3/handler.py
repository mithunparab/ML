import runpod
import os
import time
import run_sam
import json

run_sam.initialize_model()

OUTPUT_DIR = "/workspace/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def handler(job):
    job_input = job['input']
    
    image_data = job_input.get('image_data') 
    image_url = job_input.get('image_url')
    text_prompt = job_input.get('prompt')
    input_boxes = job_input.get('input_boxes')
    input_labels = job_input.get('input_labels')
    
    data_source = image_data if image_data else image_url
    is_base64 = True if image_data else False

    if not data_source:
        return {"error": "Missing 'image_data' or 'image_url'."}
    
    start_time = time.time()
    
    try:
        result_base64, mask_base64 = run_sam.process_image(
            data_source=data_source, 
            prompt=text_prompt,
            box_prompts=input_boxes,
            box_labels=input_labels,
            output_path="", 
            is_base64=is_base64
        )
    except Exception as e:
        return {"error": str(e)}

    end_time = time.time()

    if result_base64:
        return {
            "output_image_base64": result_base64, 
            "output_mask_base64": mask_base64,    
            "message": "Success",
            "inference_time": f"{end_time - start_time:.2f}s"
        }
    else:
        return {
            "output_image_base64": None,
            "output_mask_base64": None,
            "message": "No objects found.",
            "inference_time": f"{end_time - start_time:.2f}s"
        }

if __name__ == '__main__':
    print("Starting Handler...")
    runpod.serverless.start({"handler": handler})