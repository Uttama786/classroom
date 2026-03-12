from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('flipped_app.urls')),
]

# Serve media files locally (dev) or when CLOUDINARY_URL is not set.
# In production with Cloudinary, files are served directly from Cloudinary CDN.
import os
if not os.environ.get('CLOUDINARY_URL'):
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
