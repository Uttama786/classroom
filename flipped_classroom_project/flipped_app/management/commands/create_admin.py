"""
Management command: create_admin
Creates a Django superuser from environment variables if one doesn't exist.

Usage (Procfile release step):
    python manage.py create_admin

Required env vars:
    DJANGO_SUPERUSER_USERNAME  (default: admin)
    DJANGO_SUPERUSER_EMAIL     (default: admin@example.com)
    DJANGO_SUPERUSER_PASSWORD  (required)
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

        username = os.environ.get('DJANGO_SUPERUSER_USERNAME', 'admin').strip()
        email = os.environ.get('DJANGO_SUPERUSER_EMAIL', 'admin@example.com').strip()
        password = os.environ.get('DJANGO_SUPERUSER_PASSWORD', '').strip()

        if not password:
            self.stdout.write(
                self.style.WARNING(
                    '[create_admin] DJANGO_SUPERUSER_PASSWORD not set — skipping superuser creation.'
                )
            )
            return

        if len(password) < 12:
            self.stdout.write(
                self.style.WARNING(
                    '[create_admin] Password too short (<12 chars) — skipping superuser creation.'
                )
            )
            return

        User.objects.create_superuser(username=username, email=email, password=password)
        self.stdout.write(
            self.style.SUCCESS(f'[create_admin] Superuser "{username}" created successfully.')
        )
