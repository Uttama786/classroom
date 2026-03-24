import random
import re

from django.contrib.auth.hashers import make_password
from django.contrib.auth.models import User
from django.core.management.base import BaseCommand
from django.db import transaction

from flipped_app.models import (
    StudentProfile,
    Subject,
    TeacherProfile,
    VideoLecture,
    VideoWatchHistory,
)


class Command(BaseCommand):
    help = "Seed dummy student and teacher users with profiles."

    def add_arguments(self, parser):
        parser.add_argument(
            "--students",
            type=int,
            default=1000,
            help="Number of dummy students to create (default: 1000).",
        )
        parser.add_argument(
            "--teachers",
            type=int,
            default=100,
            help="Number of dummy teachers to create (default: 100).",
        )
        parser.add_argument(
            "--password",
            type=str,
            default="Pass@123",
            help="Password assigned to all created dummy users.",
        )
        parser.add_argument(
            "--min-watched-videos",
            type=int,
            default=8,
            help="Minimum watched videos to seed per new student (default: 8).",
        )
        parser.add_argument(
            "--max-watched-videos",
            type=int,
            default=16,
            help="Maximum watched videos to seed per new student (default: 16).",
        )
        parser.add_argument(
            "--watch-completion-rate",
            type=float,
            default=0.85,
            help="Completion ratio for watched records from 0 to 1 (default: 0.85).",
        )

    def handle(self, *args, **options):
        students_to_create = max(0, options["students"])
        teachers_to_create = max(0, options["teachers"])
        raw_password = options["password"]
        min_watched = max(0, options["min_watched_videos"])
        max_watched = max(min_watched, options["max_watched_videos"])
        completion_rate = min(1.0, max(0.0, options["watch_completion_rate"]))

        if students_to_create == 0 and teachers_to_create == 0:
            self.stdout.write(self.style.WARNING("Nothing to create (both counts are 0)."))
            return

        subjects = list(Subject.objects.all())
        hashed_password = make_password(raw_password)

        student_start = self._next_index("student_dummy_", width=4)
        teacher_start = self._next_index("teacher_dummy_", width=3)

        self.stdout.write(
            f"Creating {students_to_create} students and {teachers_to_create} teachers..."
        )

        with transaction.atomic():
            students, created_student_ids = self._create_students(
                students_to_create,
                student_start,
                hashed_password,
                subjects,
            )
            teachers = self._create_teachers(
                teachers_to_create,
                teacher_start,
                hashed_password,
                subjects,
            )
            watched_rows = self._seed_watch_history(
                created_student_ids,
                min_watched,
                max_watched,
                completion_rate,
            )

        self.stdout.write(self.style.SUCCESS("Dummy user seeding complete."))
        self.stdout.write(
            f"Students created: {students}; Teachers created: {teachers}; "
            f"Watched rows seeded: {watched_rows}"
        )

    def _create_students(self, count, start_index, hashed_password, subjects):
        if count <= 0:
            return 0, []

        users = []
        created_usernames = []
        for i in range(start_index, start_index + count):
            username = f"student_dummy_{i:04d}"
            created_usernames.append(username)
            users.append(
                User(
                    username=username,
                    first_name="Student",
                    last_name=f"{i:04d}",
                    email=f"{username}@fliplearn.local",
                    password=hashed_password,
                    is_staff=False,
                    is_superuser=False,
                    is_active=True,
                )
            )

        User.objects.bulk_create(users, batch_size=500)

        created_users = list(
            User.objects.filter(username__in=created_usernames)
            .order_by("id")
            .values_list("id", "username")
        )

        profiles = []
        for user_id, username in created_users:
            suffix = username.split("_")[-1]
            profiles.append(
                StudentProfile(
                    user_id=user_id,
                    roll_number=f"ROLL{suffix}",
                    semester=random.randint(1, 8),
                    previous_gpa=round(random.uniform(5.0, 9.8), 2),
                    phone=f"9{random.randint(100000000, 999999999)}",
                )
            )

        StudentProfile.objects.bulk_create(profiles, batch_size=500)

        if subjects:
            student_profiles = list(
                StudentProfile.objects.filter(user__username__in=created_usernames)
            )
            for profile in student_profiles:
                pick = random.sample(subjects, k=min(len(subjects), random.randint(2, 4)))
                profile.enrolled_subjects.add(*pick)

        created_user_ids = [user_id for user_id, _ in created_users]
        return len(users), created_user_ids

    def _seed_watch_history(
        self,
        student_ids,
        min_watched_videos,
        max_watched_videos,
        completion_rate,
    ):
        if not student_ids or max_watched_videos <= 0:
            return 0

        videos = list(
            VideoLecture.objects.filter(is_active=True).values_list("id", "duration_minutes")
        )
        if not videos:
            self.stdout.write(
                self.style.WARNING(
                    "No active videos found; skipping watched-video dummy data."
                )
            )
            return 0

        existing_by_student = {student_id: set() for student_id in student_ids}
        for student_id, video_id in VideoWatchHistory.objects.filter(
            student_id__in=student_ids
        ).values_list("student_id", "video_id"):
            existing_by_student.setdefault(student_id, set()).add(video_id)

        rows = []
        for student_id in student_ids:
            seen_video_ids = existing_by_student.get(student_id, set())
            available_videos = [video for video in videos if video[0] not in seen_video_ids]
            if not available_videos:
                continue

            target = random.randint(min_watched_videos, max_watched_videos)
            if target <= 0:
                continue

            for video_id, duration_minutes in random.sample(
                available_videos,
                k=min(target, len(available_videos)),
            ):
                duration = float(duration_minutes or random.uniform(8, 40))
                completed = random.random() < completion_rate
                if completed:
                    watched = round(duration * random.uniform(0.85, 1.0), 2)
                else:
                    watched = round(duration * random.uniform(0.2, 0.8), 2)

                rows.append(
                    VideoWatchHistory(
                        student_id=student_id,
                        video_id=video_id,
                        watch_duration_minutes=watched,
                        completed=completed,
                    )
                )

        if not rows:
            return 0

        VideoWatchHistory.objects.bulk_create(rows, batch_size=1000, ignore_conflicts=True)
        return len(rows)

    def _create_teachers(self, count, start_index, hashed_password, subjects):
        if count <= 0:
            return 0

        users = []
        created_usernames = []
        for i in range(start_index, start_index + count):
            username = f"teacher_dummy_{i:03d}"
            created_usernames.append(username)
            users.append(
                User(
                    username=username,
                    first_name="Teacher",
                    last_name=f"{i:03d}",
                    email=f"{username}@fliplearn.local",
                    password=hashed_password,
                    is_staff=True,
                    is_superuser=False,
                    is_active=True,
                )
            )

        User.objects.bulk_create(users, batch_size=200)

        created_users = list(
            User.objects.filter(username__in=created_usernames)
            .order_by("id")
            .values_list("id", "username")
        )

        profiles = []
        for user_id, username in created_users:
            suffix = username.split("_")[-1]
            profiles.append(
                TeacherProfile(
                    user_id=user_id,
                    employee_id=f"DUMMYEMP{suffix}",
                    designation=random.choice(
                        [
                            "Assistant Professor",
                            "Associate Professor",
                            "Professor",
                        ]
                    ),
                )
            )

        TeacherProfile.objects.bulk_create(profiles, batch_size=200)

        if subjects:
            teacher_profiles = list(
                TeacherProfile.objects.filter(user__username__in=created_usernames)
            )
            for profile in teacher_profiles:
                pick = random.sample(subjects, k=min(len(subjects), random.randint(1, 3)))
                profile.subjects.add(*pick)

        return len(users)

    def _next_index(self, prefix, width):
        usernames = User.objects.filter(username__startswith=prefix).values_list(
            "username", flat=True
        )
        pattern = re.compile(rf"^{re.escape(prefix)}(\d{{{width}}})$")
        max_found = 0
        for name in usernames:
            match = pattern.match(name)
            if not match:
                continue
            max_found = max(max_found, int(match.group(1)))
        return max_found + 1
