"""
Load videos, notes, quizzes from fixtures/fliplearn_content.json into the database.

Creates required uploader accounts first (admin, teacher, prof_sharma), then runs loaddata.
Used automatically on Render startup when the DB has little or no content.

  python manage.py load_content_fixture
  python manage.py load_content_fixture --force   # delete content models and reload
"""

import os
from pathlib import Path

from django.contrib.auth.models import User
from django.core.management import call_command
from django.core.management.base import BaseCommand

from flipped_app.models import (
    Assignment,
    Quiz,
    QuizQuestion,
    StudyMaterial,
    Subject,
    VideoLecture,
)

FIXTURE_PATH = Path(__file__).resolve().parents[3] / 'fixtures' / 'fliplearn_content.json'
MIN_VIDEOS_LOADED = 50  # skip reload if production already has content

REQUIRED_USERS = [
    ('admin', 'admin@fliplearn.edu', 'admin', True, True, 'Admin', 'User'),
    ('teacher', 'teacher@fliplearn.edu', 'teacher', True, False, 'Rajesh', 'Sharma'),
    ('prof_patil', 'patil@fliplearn.edu', 'teacher', True, False, 'Sunita', 'Patil'),
    ('prof_sharma', 'prof@fliplearn.edu', 'teacher', True, False, 'Rajesh', 'Sharma'),
    ('student', 'student@fliplearn.edu', 'student', False, False, 'Arjun', 'Desai'),
    ('student_arjun', 'arjun@fliplearn.edu', 'student', False, False, 'Arjun', 'Desai'),
    ('student_priya', 'priya@fliplearn.edu', 'student', False, False, 'Priya', 'Nair'),
    ('student_rohit', 'rohit@fliplearn.edu', 'student', False, False, 'Rohit', 'Mehta'),
]


def _ensure_uploaders(stdout, style):
    """Ensure essential users exist with working passwords and attached profiles."""
    from flipped_app.models import TeacherProfile, StudentProfile, Subject
    created = 0
    subjects = list(Subject.objects.all())

    for username, email, password, is_staff, is_superuser, first_name, last_name in REQUIRED_USERS:
        user, was_created = User.objects.get_or_create(
            username=username,
            defaults={
                'email': email,
                'is_staff': is_staff,
                'is_superuser': is_superuser,
                'is_active': True,
                'first_name': first_name,
                'last_name': last_name,
            },
        )
        if was_created:
            user.set_password(password)
            user.save()
            created += 1
            stdout.write(style.SUCCESS(f'  Created user account: {username} (password: {password})'))
        elif not user.has_usable_password():
            user.set_password(password)
            user.save()

        # Ensure Teacher / Student Profile
        if is_staff or is_superuser:
            tp, _ = TeacherProfile.objects.get_or_create(
                user=user,
                defaults={'employee_id': f'EMP_{username}', 'designation': 'Faculty'}
            )
            if subjects:
                tp.subjects.set(subjects)
        else:
            sp, _ = StudentProfile.objects.get_or_create(
                user=user,
                defaults={'roll_number': f'CSE_{username}', 'department': 'Computer Science & Engineering', 'semester': 4}
            )
            if subjects:
                sp.enrolled_subjects.set(subjects)

    if created == 0:
        stdout.write('  User accounts already present.')
    return created


def _clear_content():
    QuizQuestion.objects.all().delete()
    Quiz.objects.all().delete()
    Assignment.objects.all().delete()
    StudyMaterial.objects.all().delete()
    VideoLecture.objects.all().delete()
    Subject.objects.all().delete()


class Command(BaseCommand):
    help = 'Load fliplearn_content.json (131 videos, 60 materials, quizzes) for production.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Delete existing subjects/videos/materials/quizzes then reload fixture.',
        )

    def handle(self, *args, **options):
        if not FIXTURE_PATH.is_file():
            self.stderr.write(self.style.ERROR(f'Fixture not found: {FIXTURE_PATH}'))
            return

        video_count = VideoLecture.objects.count()
        material_count = StudyMaterial.objects.count()

        if video_count >= MIN_VIDEOS_LOADED and not options['force']:
            self.stdout.write(self.style.SUCCESS(
                f'Content already loaded (videos={video_count}, materials={material_count}) — skipping.'
            ))
            return

        if video_count > 0:
            if video_count < MIN_VIDEOS_LOADED:
                self.stdout.write(self.style.WARNING(
                    f'Partial content detected (videos={video_count}, materials={material_count}) '
                    f'— replacing with full fixture (131 videos, 60 materials).'
                ))
            else:
                self.stdout.write('Force reload: clearing existing content …')
            _clear_content()

        self.stdout.write('Ensuring uploader accounts for fixture …')
        _ensure_uploaders(self.stdout, self.style)

        self.stdout.write(f'Loading {FIXTURE_PATH.name} …')
        try:
            call_command('loaddata', str(FIXTURE_PATH), verbosity=1)
        except Exception as exc:
            self.stderr.write(self.style.ERROR(f'loaddata failed: {exc}'))
            raise

        self.stdout.write(self.style.SUCCESS(
            f'Loaded. videos={VideoLecture.objects.count()} '
            f'materials={StudyMaterial.objects.count()} '
            f'subjects={Subject.objects.count()}'
        ))
