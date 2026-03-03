"""
Management command: create_admin
Creates a Django superuser from environment variables if one doesn't exist.

Usage (Procfile release step):
    python manage.py create_admin

Required env vars:
    DJANGO_SUPERUSER_USERNAME  (default: admin)
    DJANGO_SUPERUSER_EMAIL     (default: admin@example.com)
    DJANGO_SUPERUSER_PASSWORD  (default: admin1234)
"""

import os
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Create a superuser from environment variables if none exists.'

    def handle(self, *args, **options):
        User = get_user_model()
        if User.objects.filter(is_superuser=True).exists():
            self.stdout.write('[create_admin] Superuser already exists — skipping.')
            return

        username = os.environ.get('DJANGO_SUPERUSER_USERNAME', 'admin')
        email    = os.environ.get('DJANGO_SUPERUSER_EMAIL',    'admin@example.com')
        password = os.environ.get('DJANGO_SUPERUSER_PASSWORD', 'admin1234')

        User.objects.create_superuser(username=username, email=email, password=password)
        self.stdout.write(
            self.style.SUCCESS(f'[create_admin] Superuser "{username}" created successfully.')
        )
