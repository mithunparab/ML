import runpod
import os
import time
import run_sam

run_sam.initialize_model()

OUTPUT_DIR = "/workspace/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def handler(job):
    job_input = job['input']
    
    # Check if image_data or image_url is provided
    image_data = job_input.get('image_data') 
    image_url = job_input.get('image_url')
    prompt = job_input.get('prompt', 'car')
    
    is_base64 = False
    
    if image_data:
        # Assume image_data is Base64 if present
        data_source = image_data
        is_base64 = True
    elif image_url:
        # Fallback to image_url if image_data is not present
        data_source = image_url
    else:
        return {"error": "Missing 'image_data' (Base64) or 'image_url' in input."}
    
    output_filename = f"{job['id']}.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    start_time = time.time()
    result_base64 = run_sam.process_image(data_source, prompt, output_path, is_base64=is_base64)
    end_time = time.time()
    
    print(f"Inference time: {end_time - start_time:.2f} seconds")

    if result_base64:
        return {
            "output_image_base64": result_base64,
            "message": "Segmentation and extraction complete. Image saved to output_image_base64.",
            "inference_time": f"{end_time - start_time:.2f}s"
        }
    else:
        return {
            "output_image_base64": None,
            "message": "No objects found with the given prompt and confidence threshold.",
            "inference_time": f"{end_time - start_time:.2f}s"
        }

if __name__ == '__main__':
    print("RunPod Handler starting...")
    print(f"Model initialized. Saving outputs to {OUTPUT_DIR}. Ready to accept jobs.")
    runpod.serverless.start({"handler": handler})