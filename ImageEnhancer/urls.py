from django.urls import path
from . import views
from .views import enhance_image
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name='index'),
    path('enhancer/', enhance_image, name='enhance_image'),
    path('object-detection/', views.object_detection, name='object_detection'),
    path('generate_image/', views.generate_image, name='generate_image'),
    path('resize/', views.resize_image, name='resize_image'),
    path('classify_image/', views.classify_image, name='classify_image'),
    path('depth_analysis/', views.depth_analysis, name='depth_analysis'),
    path('image_questioning/', views.process_image_questioning, name='process_image_questioning'),
    path('redesigner/', views.redesigner, name='redesigner'),
    path('cartoonizer/', views.cartoonizer_view, name='cartoonizer'),
    path('video-captioning/', views.video_captioning_view, name='video_captioning'),
    path('enhancerandcompressor/', views.enhance_image_compress, name='enhance_image_compress'),
    path('compress-image/', views.compress_image, name='compress_image'),
    path('citations/', views.citations_page, name='citations'),
    path('video-compressor/', views.video_compressor, name='video_compressor'),
    path('video-fps-changer/', views.video_fps_changer, name='video_fps_changer'),
    path('combine/', views.image_combiner_view, name='image_combiner'),
    path('combine_vertical/', views.combine_images_vertically_view, name='combine_images_vertically'),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
