from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.core.files.storage import FileSystemStorage
import os
from django.conf import settings
from .clip_model import generate_captions  # Import our function
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

def app(request):
    return render(request, 'test.html')
def home(request):
    if request.method == 'POST':
        if request.POST.get('regenerate') == 'true':
            # Regenerate using existing image path
            image_url = request.POST.get('image_path')
            uploaded_file_url = image_url  # ✅ assign it here too for reuse in render
            file_path = os.path.join(settings.MEDIA_ROOT, image_url.replace(settings.MEDIA_URL, ''))
        else:
            # Handle new image upload
            uploaded_file = request.FILES['image']
            fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'uploads'))
            filename = fs.save(uploaded_file.name, uploaded_file)
            uploaded_file_url = settings.MEDIA_URL + 'uploads/' + filename
            file_path = os.path.join(settings.MEDIA_ROOT, 'uploads', filename)
        
        # Generate captions using CLIP
        captions = generate_captions(file_path)
        
        return render(request, 'home.html', {
            'uploaded_file_url': uploaded_file_url,
            'captions': captions
        })
    
    return render(request, 'home.html')


@csrf_exempt
def caption_api(request):
    print("METHOD:", request.method)
    print("FILES:", request.FILES)

    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = next(iter(request.FILES.values()), None)
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'uploads'))
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = os.path.join(settings.MEDIA_ROOT, 'uploads', filename)

        # Optional: add debug print
        print(f"Saved: {file_path}")

        # Generate captions
        category, captions = generate_captions(file_path)

        return JsonResponse({
            'category': category,
            'captions': captions
        })

    return JsonResponse({'error': 'no file'}, status=400)
