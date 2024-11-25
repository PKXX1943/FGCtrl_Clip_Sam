import os
import json
import io
import base64
import numpy as np
from PIL import Image
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class CellAnalyzer:
    sys_prompt = """You are a medical imaging doctor. I will present different cell tissue slices, please answer the related questions. Keep your answers concise and no more than 30 words."""

    def __init__(self, openai_api_key):
        self.api_key = openai_api_key
        self.url = "https://twapi.openai-hk.com/v1/chat/completions"  
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def encode_pil_image_to_base64(self, pil_image):
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")  # Save as PNG format
        return str(base64.b64encode(buffered.getvalue()).decode('utf-8'))

    def resize_image_if_needed(self, img):
        max_dimension = 196
        width, height = img.size
        if width > max_dimension or height > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int((max_dimension / width) * height)
            else:
                new_height = max_dimension
                new_width = int((max_dimension / height) * width)
            img = img.resize((new_width, new_height))
        return img

    def analyze_cells(self, img: Image, mask: Image = None):
        # Resize images if needed
        img = self.resize_image_if_needed(img)

        base64_image = self.encode_pil_image_to_base64(img)
        results = {"caption_all": None, "caption_cell": None, "caption_bg": None}

        if mask:
            mask = self.resize_image_if_needed(mask)  # Resize mask if needed
            # Convert mask to grayscale and apply
            mask = mask.convert("L")  # Convert to grayscale
            mask_array = np.array(mask)
            mask_region = np.where(mask_array > 0, 1, 0)

            # Extract the mask region
            img_array = np.array(img)
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array, img_array, img_array], axis=-1)
            masked_image = Image.fromarray(np.uint8(img_array * mask_region[:,:,np.newaxis]))  # Apply mask to the image
            unmasked_image = Image.fromarray(np.uint8(img_array * (1 - mask_region[:,:,None])))  # Image without mask

            # Convert masked and unmasked images to base64 for GPT analysis
            base64_masked_image = self.encode_pil_image_to_base64(masked_image)
            base64_unmasked_image = self.encode_pil_image_to_base64(unmasked_image)

            # 1. Analyze the full image
            full_user_request = f"""This is a complete cell imaging image. Please tell me what cell it might be and in which part of the human body. use only percise words in one line."""
            results["caption_all"] = self.call_gpt_api(full_user_request, base64_image)

            # 2. Analyze the masked region (cells)
            masked_user_request = f"""This image only contains cells. Please describe the color, shape, number, and distribution of the cells. use only percise words in one line."""
            results["caption_cell"] = self.call_gpt_api(masked_user_request, base64_masked_image)

            # 3. Analyze the unmasked region (background)
            unmasked_user_request = f"""This is a slice with masked cells. Please describe the background color and texture. use only percise words in one line."""
            results["caption_bg"] = self.call_gpt_api(unmasked_user_request, base64_unmasked_image)

        else:
            # If no mask, only analyze the full image and ask three questions
            # Full image: "What cell might this be and in which part of the body?"
            full_user_request = f"""This is a complete cell imaging image. Please answer: \n1.what cell it might be and in which part of the human body.\n2.Describe the color, shape, number, and distribution of the cells.\n3.Focus on the background where there is not any cell and describe the background color and texture.\nUse only percise words, no indexes. One line for each answer and make sure there are 3 lines."""
            all_answers = self.call_gpt_api(full_user_request, base64_image).strip().split('\n')
            results["caption_all"] = all_answers[0]
            results['caption_cell'] = all_answers[1]
            results['caption_bg'] = all_answers[2]
            
        return results

    def call_gpt_api(self, user_request, base64_image):
        try:
            data = {
                "max_tokens": 1200,
                "model": "gpt-4o-mini",
                "temperature": 0.8,
                "top_p": 1,
                "presence_penalty": 1,
                "messages": [
                    {"role": "system", "content": CellAnalyzer.sys_prompt},
                    {"role": "user", "content": f"[Image: data:image/png;base64,{base64_image}]"},
                    {"role": "user", "content": user_request},
                    # {
                    #     "role": "system",
                    #     "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible."
                    # },
                    # {
                    #     "role": "user",
                    #     "content": "你是chatGPT多少？"
                    # }
                ]
            }

            # Send POST request to OpenAI API
            response = requests.post(self.url, headers=self.headers, data=json.dumps(data).encode('utf-8'))

            # Get response
            result = response.content.decode("utf-8")
            response_json = json.loads(result)

            # Extract the analysis result from the response
            analysis_result = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
            return analysis_result

        except Exception as e:
            print(f"Error while requesting OpenAI API: {e}")
            return None

def process_image_data(image_path, mask_path, analyzer):
    try:
        img = Image.open(image_path)
        mask = Image.open(mask_path) if os.path.exists(mask_path) else None
        analysis_result = analyzer.analyze_cells(img, mask)
    except Exception as e:
        print(f"Error while processing image: {e}")
        analysis_result = {}
    result = {
        'image_path': image_path,
        'mask_path': mask_path if os.path.exists(mask_path) else None,
        'analysis_result': analysis_result
    }
    return result

def process_batch(batch, analyzer, pbar):
    results = []
    for image_path, mask_path in batch:
        result = process_image_data(image_path, mask_path, analyzer)
        if result is not None:
            results.append(result)
        pbar.update(1)
    return results

def save_results(results, output_file):
    # os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

def chunk_data(data, num_threads):
    # Divide the data into chunks based on the number of threads
    chunk_size = len(data) // num_threads
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    # If there is a remainder, add it to the last chunk
    if len(data) % num_threads != 0:
        chunks[-1].extend(data[len(chunks)*chunk_size:])
    return chunks

def main():
    # Set image and mask folder paths
    image_folder = 'data/Training-labeled/images'
    mask_folder = 'data/Training-labeled/labels'
    output_file = 'data/annotations/training_labeled_detailed.json'

    # Initialize the analyzer
    analyzer = CellAnalyzer(openai_api_key="hk-rheyva100004661330ee0aeabe882b1448211555c790a6e9")  # Add actual API key

    # Get all image and mask paths
    image_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder)])
    mask_paths = sorted([os.path.join(mask_folder, f) for f in os.listdir(mask_folder)])

    # Create data pairs
    data = [(image_paths[i], mask_paths[i]) for i in range(len(image_paths))]

    # Get number of threads (default is 4)
    num_threads = 16   # You can adjust this value

    # Create batches based on the number of threads
    batches = chunk_data(data, num_threads)

    # Use ThreadPoolExecutor to process batches in parallel with tqdm progress bar
    all_results = []
    pbar = tqdm(total=len(data), desc="Processing images")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for batch in batches:
            futures.append(executor.submit(process_batch, batch, analyzer, pbar))

        for future in futures:
            all_results.extend(future.result())

    # Save all results to the output JSON file
    save_results(all_results, output_file)

if __name__ == "__main__":
    main()
