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
    
    is_base64 = False
    data_source = None
    
    if image_data:
        data_source = image_data
        is_base64 = True
    elif image_url:
        data_source = image_url
    else:
        return {"error": "Missing 'image_data' (Base64) or 'image_url' in input."}
    
    if not text_prompt and not input_boxes:
         return {"error": "Missing prompt data. Provide 'prompt' (text) and/or 'input_boxes'."}

    output_filename = f"{job['id']}.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    start_time = time.time()
    
    try:
        result_base64 = run_sam.process_image(
            data_source=data_source, 
            prompt=text_prompt,
            box_prompts=input_boxes,
            box_labels=input_labels,
            output_path=output_path, 
            is_base64=is_base64
        )
    except Exception as e:
        return {"error": str(e)}

    end_time = time.time()
    
    print(f"Inference time: {end_time - start_time:.2f} seconds")

    if result_base64:
        return {
            "output_image_base64": result_base64,
            "message": "Segmentation and extraction complete.",
            "inference_time": f"{end_time - start_time:.2f}s"
        }
    else:
        return {
            "output_image_base64": None,
            "message": "No objects found with the given prompts and confidence threshold.",
            "inference_time": f"{end_time - start_time:.2f}s"
        }

if __name__ == '__main__':
    print("RunPod Handler starting...")
    print(f"Model initialized. Saving outputs to {OUTPUT_DIR}. Ready to accept jobs.")
    runpod.serverless.start({"handler": handler})