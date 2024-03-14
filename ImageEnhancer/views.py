from django.shortcuts import render

def index(request):
    return render(request, 'index.html')

from django.shortcuts import render

def index(request):
    return render(request, 'index.html')

def citations_page(request):
    return render(request, 'citations.html')

# views.py
from django.shortcuts import render
from django.core.files.storage import default_storage
from PIL import Image
from diffusers import LDMSuperResolutionPipeline
import torch
import os

import os
from django.shortcuts import render
from django.core.files.storage import default_storage
from PIL import Image
from diffusers import LDMSuperResolutionPipeline
import torch

def enhance_image(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        file_path = default_storage.save(uploaded_image.name, uploaded_image)
        input_image_path = default_storage.path(file_path)
        enhanced_image_path = enhance_image_with_model(input_image_path)

        # Construct the relative path to the enhanced image within the 'media' subdirectory
        enhanced_image_relative_path = os.path.relpath(enhanced_image_path, 'media')

        # Delete the uploaded image from the 'media' directory
        os.remove(input_image_path)

        return render(request, 'enhance_image.html', {'enhanced_image_path': enhanced_image_relative_path})
    else:
        return render(request, 'enhance_image.html')

def enhance_image_with_model(input_image_path):
    # Load the input image
    low_res_img = Image.open(input_image_path).convert("RGB")
    original_width, original_height = low_res_img.size

    # Resize the low resolution image
    low_res_img = low_res_img.resize((128, 128))

    # Load the enhancer model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "CompVis/ldm-super-resolution-4x-openimages"
    pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to(device)

    # Run the enhancer model
    upscaled_image = pipeline(low_res_img, num_inference_steps=200, eta=1).images[0]

    # Resize the enhanced image to original dimensions
    upscaled_image = upscaled_image.resize((original_width, original_height), Image.BICUBIC)

    # Save the enhanced image with a unique filename within the 'media' subdirectory
    enhanced_image_name = 'enhanced_image.png'
    enhanced_image_path = os.path.join(os.path.dirname(input_image_path), enhanced_image_name)
    upscaled_image.save(enhanced_image_path)
    return enhanced_image_path

def enhance_image_compress(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        file_path = default_storage.save(uploaded_image.name, uploaded_image)
        input_image_path = default_storage.path(file_path)
        enhanced_image_path = enhance_image_with_model(input_image_path)

        # Construct the relative path to the enhanced image within the 'media' subdirectory
        enhanced_image_relative_path = os.path.relpath(enhanced_image_path, 'media')

        # Delete the uploaded image from the 'media' directory
        os.remove(input_image_path)

        return render(request, 'enhance_image_compress.html', {'enhanced_image_path': enhanced_image_relative_path})
    else:
        return render(request, 'enhance_image_compress.html')
# views.py

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from PIL import Image, ImageDraw, ImageFont
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch

def object_detection(request):
    if request.method == 'POST' and request.FILES['image']:
        # Handle image upload
        uploaded_image = request.FILES['image']
        fs = FileSystemStorage()
        image_path = fs.save(uploaded_image.name, uploaded_image)
        image_url = fs.url(image_path)

        # Load the model and processor
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

        # Process the uploaded image
        image = Image.open(str(settings.BASE_DIR) + image_url)
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(image)
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x_min, y_min, x_max, y_max = box.tolist()
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

        # Save the processed image
        processed_image_path = str(settings.BASE_DIR) + '/media/processed_image.jpg'
        image.save(processed_image_path)

        return render(request, 'object_detection_result.html', {'processed_image_url': '/media/processed_image.jpg'})

    return render(request, 'object_detection.html')

from django.shortcuts import render
from diffusers import StableDiffusionXLPipeline
import torch


from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from PIL import Image
from diffusers import StableDiffusionXLPipeline
import torch

# views.py

from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from PIL import Image
from diffusers import StableDiffusionXLPipeline
import torch

# Load the image generation model
pipe = StableDiffusionXLPipeline.from_pretrained("segmind/SSD-1B", torch_dtype=torch.float32, use_safetensors=True, variant="fp16")

def generate_image(request):
    if request.method == 'POST':
        prompt = request.POST.get('prompt')#
        neg_prompt = "ugly, blurry, poor quality" # Negative prompt here

        # Generate the image using the model
        image = pipe(prompt=prompt, negative_prompt=neg_prompt, num_inference_steps=10).images[0]

        # Save the generated image to a temporary file
        image_path = str(settings.BASE_DIR) + '/media/generated_image.png'
        image.save(image_path)

        # Render the HTML template with the generated image path
        return render(request, 'image_generator.html', {'generated_image': '/media/generated_image.png'})

    return render(request, 'image_generator.html')

from django.shortcuts import render
from django.http import HttpResponse
import cv2
import numpy as np
from django.http import HttpResponse
from django.shortcuts import render

from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
import os
import cv2

def resize_image(request):
    if request.method == 'POST':
        # Get the uploaded image and dimensions from the form
        uploaded_image = request.FILES['image']
        target_width = int(request.POST['width'])
        target_height = int(request.POST['height'])

        # Save the uploaded image temporarily
        temp_image_path = os.path.join(str(settings.MEDIA_ROOT), 'temp_image.jpg')
        with default_storage.open(temp_image_path, 'wb+') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)

        # Read the uploaded image
        img = cv2.imread(temp_image_path)

        # Resize the image
        resized_img = cv2.resize(img, (target_width, target_height))

        # Save the resized image
        resized_image_path = os.path.join(str(settings.MEDIA_ROOT), 'resized_image.jpg')
        cv2.imwrite(resized_image_path, resized_img)

        # Delete the temporary image
        default_storage.delete(temp_image_path)

        # Pass the resized image to the template
        resized_image_url = os.path.join(str(settings.MEDIA_URL), 'resized_image1.jpg')
        return render(request, 'resize_image.html', {'resized_image': '/media/resized_image.jpg'})

    return render(request, 'resize_image.html')

from django.shortcuts import render
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import os

def classify_image(request):
    if request.method == 'POST' and request.FILES['image']:
        # Get the uploaded image
        image = request.FILES['image']
        image_path = os.path.join('media', image.name)
        with open(image_path, 'wb') as f:
            for chunk in image.chunks():
                f.write(chunk)

        # Load the input image
        image = Image.open(image_path)
        # Convert RGBA image to RGB mode if necessary
        if image.mode == "RGBA":
            image = image.convert("RGB")

        # Load the model and processor
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

        # Process the input image
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Convert outputs (bounding boxes and class logits) to COCO API
        # Let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        # Create a drawing context
        draw = ImageDraw.Draw(image)

        # Draw bounding boxes around detected objects
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            # Extract box coordinates
            x_min, y_min, x_max, y_max = box.tolist()

            # Draw the box
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

            # Add label and confidence score
            label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}"
            draw.text((x_min, y_min), label_text, fill="blue")

        # Save the output image
        output_image_path = os.path.join(str(settings.MEDIA_ROOT), 'output_image.jpg')
        image.save(output_image_path)

        return render(request, 'classify_image.html', {'output_image': '/media/output_image.jpg'})

    return render(request, 'classify_image.html')

# views.py

from django.shortcuts import render
from transformers import DPTImageProcessor, DPTForDepthEstimation
from PIL import Image
import numpy as np
import torch

def depth_analysis(request):
    if request.method == 'POST' and request.FILES['image']:
        # Load the input image
        image = Image.open(request.FILES['image'])
        
        # Initialize the model and processor
        processor = DPTImageProcessor.from_pretrained("Intel/dpt-beit-large-512")
        model = DPTForDepthEstimation.from_pretrained("Intel/dpt-beit-large-512")
        
        # Prepare image for the model
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        # Visualize the prediction
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth_map = Image.fromarray(formatted)

        # Save the generated depth map image
        depth_map_path = os.path.join(str(settings.MEDIA_ROOT), "predicted_depth_map.jpg")
        depth_map.save(depth_map_path)

        return render(request, 'depth_analysis.html', {'depth_map_path': '/media/predicted_depth_map.jpg'})
    else:
        return render(request, 'depth_analysis.html')

# views.py
from django.shortcuts import render
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

def process_image_questioning(request):
    if request.method == 'POST' and request.FILES['image'] and request.POST['prompt']:
        # Load the processor and model
        processor = AutoProcessor.from_pretrained("microsoft/git-base-textvqa")
        model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-textvqa")

        # Process the input image
        uploaded_image = request.FILES['image']
        image = Image.open(uploaded_image).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values

        # Process the text prompt
        prompt = request.POST['prompt']
        input_ids = processor(text=prompt, add_special_tokens=False).input_ids
        input_ids = [processor.tokenizer.cls_token_id] + input_ids
        input_ids = torch.tensor(input_ids).unsqueeze(0)

        # Generate output
        generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
        output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return render(request, 'image_questioning.html', {'output': output})

    return render(request, 'image_questioning.html')

from django.shortcuts import render
from django.http import HttpResponse
import torch
from diffusers import ShapEImg2ImgPipeline
from diffusers.utils import export_to_gif, load_image

from django.shortcuts import render
from django.http import HttpResponse
import PIL
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

from PIL import Image, ImageOps
import os
from django.conf import settings
from django.shortcuts import render
import PIL
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

def download_image(image):
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

def resize_image2(image, size):
    return image.resize(size, Image.LANCZOS)

def redesigner(request):
    if request.method == 'POST':
        # Process the form data
        input_image = request.FILES.get('image')
        prompt = request.POST.get('prompt')

        # Load the input image
        image = PIL.Image.open(input_image)

        # Resize the input image to 512x512
        image = resize_image2(image, (512, 512))

        # Download image
        image = download_image(image)

        # Initialize the model
        model_id = "timbrooks/instruct-pix2pix"
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float32, safety_checker=None)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

        # Process the input image and prompt
        images = pipe(prompt, image=image, num_inference_steps=50, image_guidance_scale=1).images

        # Resize the generated image back to original dimensions
        original_size = (image.width, image.height)
        output_image = resize_image2(images[0], original_size)

        # Save the generated image
        output_image_path = os.path.join(str(settings.MEDIA_ROOT), "./generated_image627.jpg")
        output_image.save(output_image_path)

        # Pass the generated image path to the template
        context = {'output_image_path': '/media/generated_image627.jpg'}
        return render(request, 'redesigner.html', context)
    else:
        return render(request, 'redesigner.html')


# views.py
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from PIL import Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline

def cartoonizer_view(request):
    if request.method == 'POST' and request.FILES['image']:
        # Get the uploaded image
        uploaded_image = request.FILES['image']
        
        # Save the uploaded image to a temporary location
        file_name = default_storage.save('temp_image.jpg', ContentFile(uploaded_image.read()))

        # Load the input image from the temporary location
        input_image = Image.open(default_storage.path(file_name))
        input_image = input_image.convert("RGB")

        # Initialize the model
        model_id = "instruction-tuning-sd/cartoonizer"
        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=torch.float32, use_auth_token=True
        )

        # Apply the cartoonization process
        output_image = pipeline("Cartoonize the following image", image=input_image, num_inference_steps=10).images[0]

        # Save the output image to a temporary location
        output_file_path = default_storage.path('output_image.png')
        output_image.save(output_file_path)

        # Pass the paths to the template
        context = {
            'input_image_path': default_storage.url(file_name),
            'output_image_path': default_storage.url('output_image.png')
        }

        return render(request, 'cartoonizer.html', context)

    return render(request, 'cartoonizer.html')

from django.shortcuts import render
from django.conf import settings
from super_image import EdsrModel, ImageLoader
from PIL import Image
import os
import tempfile

# views.py
from django.shortcuts import render
from django.http import HttpResponse
import av
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

# Initialize the model and processor
processor = AutoProcessor.from_pretrained("microsoft/git-base-vatex")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-vatex")
np.random.seed(45)

def read_video_pyav(file_path, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        file_path (str): Path to the input video file.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container = av.open(file_path)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def video_captioning_view(request):
    generated_captions = None
    if request.method == 'POST':
        video_file = request.FILES['video']
        text_prompt = request.POST.get('text_prompt', '')
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            # Write the uploaded video content to the temporary file
            temp_video.write(video_file.read())
            temp_video_path = temp_video.name
        
            num_frames = model.config.num_image_with_embedding
            container = av.open(temp_video_path)
            indices = sample_frame_indices(
                clip_len=num_frames, frame_sample_rate=4, seg_len=container.streams.video[0].frames
            )
            frames = read_video_pyav(temp_video_path, indices)
            pixel_values = processor(images=list(frames), return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
            generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # The temporary file will be automatically deleted when the 'with' block ends

    return render(request, 'video_captioning.html', {'generated_captions': generated_captions})

from django.shortcuts import render
from django.conf import settings
from PIL import Image
import os

def reduce_image_size(image_path, output_path, max_size=500):
    """Reduces the size of an image while maintaining its aspect ratio.

    Args:
        image_path: The path to the input image.
        output_path: The path to the output image.
        max_size: The maximum size of the output image in pixels.
    """
    # Open the input image
    image = Image.open(image_path)

    # Check if the input image is not in RGB mode
    if image.mode != "RGB":
        # Convert the image to RGB mode
        image = image.convert("RGB")

    # Get the image dimensions
    width, height = image.size

    # Determine the scaling factor
    if width > height:
        scaling_factor = max_size / width
    else:
        scaling_factor = max_size / height

    # Resize the image
    resized_image = image.resize((int(width * scaling_factor), int(height * scaling_factor)), Image.LANCZOS)

    # Save the resized image
    resized_image.save(output_path)

def compress_image(request):
    if request.method == 'POST':
        input_image = request.FILES['image']
        input_path = os.path.join(settings.MEDIA_ROOT, input_image.name)
        output_path = os.path.join(str(settings.MEDIA_ROOT), 'compressed_image.jpg')
        
        with open(input_path, 'wb') as f:
            for chunk in input_image.chunks():
                f.write(chunk)

        reduce_image_size(input_path, output_path)
        compressed_image_url = os.path.join(str(settings.MEDIA_URL), 'compressed_image.jpg')
        return render(request, 'image_upload.html', {'compressed_image_url': '/media/compressed_image.jpg'})
    else:
        return render(request, 'image_upload.html')

# views.py
from django.shortcuts import render
from django.http import JsonResponse
import moviepy.editor as mp

def reduce_video_size(input_path, output_path, target_bitrate):
    """
    Reduces the size of a video by adjusting the resolution, frame rate, and codec.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the reduced-size video file.
        target_bitrate (int): Target bitrate for the output video (in kbps).
    """
    # Load the original video
    video_clip = mp.VideoFileClip(input_path)

    # Reduce resolution by half
    resized_clip = video_clip.resize(height=video_clip.size[1]//2, width=video_clip.size[0]//2)

    # Reduce the bitrate of the video
    reduced_clip = resized_clip.subclip().\
        write_videofile(output_path, codec='libx264', bitrate=f"{target_bitrate}k")

    # Close the video clips
    video_clip.close()
    resized_clip.close()

    return reduced_clip

def video_compressor(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        target_bitrate = int(request.POST.get('bitrate'))

        # Save the uploaded video file
        input_path = os.path.join(str(settings.MEDIA_ROOT), 'uploaded_video.mp4')
        with open(input_path, 'wb') as f:
            for chunk in video_file.chunks():
                f.write(chunk)

        # Compress the video
        output_path = os.path.join(str(settings.MEDIA_ROOT), 'compressed_video.mp4')
        reduce_video_size(input_path, output_path, target_bitrate)

        # Return the compressed video path
        return JsonResponse({'compressed_video_path': '/media/compressed_video.mp4'})

    return render(request, 'video_compressor.html')

from django.shortcuts import render
from django.http import JsonResponse
from moviepy.editor import VideoFileClip, clips_array

def change_video_fps(input_path, output_path, fps):
    """
    Changes the FPS of the input video and saves the output video.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the output video file.
        fps (int): Desired frames per second for the output video.
    """
    # Load the input video
    video_clip = VideoFileClip(input_path)

    # Set the desired FPS for the output video
    video_clip = video_clip.set_fps(fps)

    # Write the output video to the specified path
    video_clip.write_videofile(output_path, codec='libx264')

    # Close the video clip
    video_clip.close()

def video_fps_changer(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        fps = int(request.POST.get('fps'))

        # Save the uploaded video file
        input_path = os.path.join(str(settings.MEDIA_ROOT), 'uploaded_video.mp4')
        with open(input_path, 'wb') as f:
            for chunk in video_file.chunks():
                f.write(chunk)

        # Process the video
        output_path = os.path.join(str(settings.MEDIA_ROOT), 'output_video.mp4')
        change_video_fps(input_path, output_path, fps)

        # Return the processed video path
        output_video_path = os.path.join(str(settings.MEDIA_URL), 'output_video.mp4')
        return JsonResponse({'output_video_path': '/media/output_video.mp4'})

    return render(request, 'video_fps_changer.html')

from django.shortcuts import render
from PIL import Image

from django.shortcuts import render
from PIL import Image

def combine_images_horizontally(image1_path, image2_path, output_path):
    # Open the images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Resize both images to fit within the same height
    target_height = min(image1.height, image2.height)
    aspect_ratio1 = target_height / image1.height
    aspect_ratio2 = target_height / image2.height

    image1_width = int(image1.width * aspect_ratio1)
    image2_width = int(image2.width * aspect_ratio2)

    image1 = image1.resize((image1_width, target_height))
    image2 = image2.resize((image2_width, target_height))

    # Calculate the total width for the combined image
    total_width = image1_width + image2_width

    # Create a new image with the combined dimensions
    combined_image = Image.new('RGB', (total_width, target_height))

    # Paste the first image onto the combined image
    combined_image.paste(image1, (0, 0))

    # Paste the second image next to the first image
    combined_image.paste(image2, (image1_width, 0))

    # Save the combined image
    combined_image.save(output_path)

def image_combiner_view(request):
    if request.method == 'POST' and 'image1' in request.FILES and 'image2' in request.FILES:
        # Get the uploaded images
        image1 = request.FILES['image1']
        image2 = request.FILES['image2']

        # Define paths for the uploaded images and the combined image
        image1_path = os.path.join(str(settings.MEDIA_ROOT), 'temmpp.jpg')
        image2_path = os.path.join(str(settings.MEDIA_ROOT), 'temmpp2.jpg')
        output_path = os.path.join(str(settings.MEDIA_ROOT), 'up_h.jpg')  # Change the output path as needed

        # Save the uploaded images to the media directory
        with open(image1_path, 'wb') as f1, open(image2_path, 'wb') as f2:
            for chunk in image1.chunks():
                f1.write(chunk)
            for chunk in image2.chunks():
                f2.write(chunk)

        # Call the combine_images_horizontally function
        combine_images_horizontally(image1_path, image2_path, output_path)

        # Pass the combined image path to the template for display
        context = {'combined_image_url': '/media/up_h.jpg'}
        return render(request, 'image_upload1.html', context)
    return render(request, 'image_upload1.html')

from django.shortcuts import render
from django.conf import settings
import os

from django.shortcuts import render
from django.conf import settings
from PIL import Image

def combine_images_vertically(image1_path, image2_path, output_path):
    # Open the images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Resize both images to fit within the same width
    target_width = min(image1.width, image2.width)
    aspect_ratio1 = target_width / image1.width
    aspect_ratio2 = target_width / image2.width

    image1_height = int(image1.height * aspect_ratio1)
    image2_height = int(image2.height * aspect_ratio2)

    image1 = image1.resize((target_width, image1_height))
    image2 = image2.resize((target_width, image2_height))

    # Calculate the total height for the combined image
    total_height = image1_height + image2_height

    # Create a new image with the combined dimensions
    combined_image = Image.new('RGB', (target_width, total_height))

    # Paste the first image onto the combined image
    combined_image.paste(image1, (0, 0))

    # Paste the second image below the first image
    combined_image.paste(image2, (0, image1_height))

    # Save the combined image
    combined_image.save(output_path)

def combine_images_vertically_view(request):
    if request.method == 'POST':
        # Handle form submission
        image1 = request.FILES['image1']
        image2 = request.FILES['image2']

        # Define paths for the uploaded images and combined image
        image1_path = os.path.join(settings.MEDIA_ROOT, 'image1.jpg')
        image2_path = os.path.join(settings.MEDIA_ROOT, 'image2.jpg')
        combined_image_path = os.path.join(settings.MEDIA_ROOT, 'combined_image_vertical.jpg')

        # Save the uploaded images to disk
        with open(image1_path, 'wb') as f:
            for chunk in image1.chunks():
                f.write(chunk)
        
        with open(image2_path, 'wb') as f:
            for chunk in image2.chunks():
                f.write(chunk)

        # Combine images vertically
        combine_images_vertically(image1_path, image2_path, combined_image_path)

        # Construct the URL for the combined image
        combined_image_url = os.path.join(settings.MEDIA_URL, 'combined_image_vertical.jpg')

        # Pass the combined image URL to the template
        return render(request, 'combine_vertical.html', {'combined_image_url': combined_image_url})

    # Render the form template for GET requests
    return render(request, 'combine_vertical.html')
