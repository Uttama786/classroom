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

FIRST_NAMES = [
    # Male names
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayan", "Krishna", "Ishaan",
    "Shaurya", "Atharva", "Advait", "Pranav", "Aryan", "Dhruv", "Kabir", "Ritvik", "Darsh", "Rohan",
    "Rahul", "Amit", "Varun", "Nikhil", "Siddharth", "Harsh", "Yash", "Kunal", "Gaurav", "Mayank",
    "Ayush", "Akash", "Chirag", "Karan", "Rishabh", "Mohit", "Alok", "Dev", "Manish", "Abhishek",
    "Suresh", "Vikram", "Aniket", "Sanket", "Shubham", "Tejas", "Tanmay", "Omkar", "Prathamesh", "Pradeep",
    "Dinesh", "Manoj", "Pankaj", "Sachin", "Deepak", "Vivek", "Vishal", "Ashish", "Anand", "Rajesh",
    "Ravi", "Sanjay", "Sunil", "Ajay", "Vinay", "Hemant", "Chetan", "Tushar", "Girish", "Naveen",
    # Female names
    "Ananya", "Diya", "Aadhya", "Pari", "Saanvi", "Kiara", "Myra", "Riya", "Ira", "Avani",
    "Prisha", "Riddhi", "Sneha", "Tanvi", "Anika", "Navya", "Kavya", "Ishita", "Meera", "Pooja",
    "Neha", "Swati", "Shreya", "Divya", "Simran", "Mansi", "Payal", "Sonam", "Muskan", "Kriti",
    "Richa", "Pallavi", "Chetna", "Jyoti", "Vandana", "Preeti", "Komal", "Garima", "Sakshi", "Nidhi",
    "Bhavya", "Tanisha", "Shweta", "Deepika", "Shruti", "Rashmi", "Ankita", "Akanksha", "Sunita", "Monali",
    "Pragya", "Srishti", "Ritika", "Nisha", "Meenakshi", "Shilpa", "Trisha", "Lavanya", "Charu", "Harshita",
    "Manisha", "Urvashi", "Kavita", "Suman", "Geeta", "Anjali", "Bhumika", "Chhavi", "Devanshi", "Ekta",
]

LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Patel", "Singh", "Kumar", "Rao", "Reddy", "Nair", "Iyer",
    "Joshi", "Mehta", "Shah", "Agarwal", "Mishra", "Bhat", "Deshmukh", "Kulkarni", "Patil", "Banerjee",
    "Chatterjee", "Mukherjee", "Ghosh", "Das", "Sen", "Dutta", "Roy", "Bose", "Choudhury", "Chakraborty",
    "Pillai", "Menon", "Nambiar", "Kurian", "Thomas", "Mathew", "Varghese", "Joseph", "Fernandes", "D'Souza",
    "Lobo", "Pinto", "Chauhan", "Rajput", "Rathore", "Solanki", "Tomar", "Yadav", "Maurya", "Saini",
    "Jangid", "Prajapat", "Bishnoi", "Mittal", "Bansal", "Goyal", "Singhal", "Garg", "Jindal", "Goel",
    "Mahajan", "Kapoor", "Malhotra", "Khanna", "Arora", "Sethi", "Grover", "Batra", "Anand", "Bajaj",
    "Chopra", "Dhawan", "Kohli", "Suri", "Tandon", "Ahuja", "Lamba", "Bhasin", "Duggal", "Talwar",
]

DEPARTMENTS = ["CS", "IT", "AI", "DS", "EC", "EE", "ME", "CE"]


class Command(BaseCommand):
    help = "Seed student and teacher users with realistic profiles."

    def add_arguments(self, parser):
        parser.add_argument(
            "--students",
            type=int,
            default=1000,
            help="Number of students to create (default: 1000).",
        )
        parser.add_argument(
            "--target-students",
            type=int,
            default=None,
            help="Ensure database has at least this total number of students (creates difference).",
        )
        parser.add_argument(
            "--teachers",
            type=int,
            default=100,
            help="Number of teachers to create (default: 100).",
        )
        parser.add_argument(
            "--password",
            type=str,
            default="Pass@123",
            help="Password assigned to all created users.",
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
        if options.get("target_students") is not None:
            current_count = StudentProfile.objects.count()
            target = max(0, options["target_students"])
            if current_count >= target:
                self.stdout.write(self.style.SUCCESS(
                    f"Already at {current_count} students (target={target}) — skipping."
                ))
                return
            students_to_create = target - current_count
            self.stdout.write(f"Current students: {current_count}. Creating {students_to_create} to reach {target}...")
        else:
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

        self.stdout.write(
            f"Creating {students_to_create} students and {teachers_to_create} teachers with real names..."
        )

        with transaction.atomic():
            students, created_student_ids = self._create_students(
                students_to_create,
                hashed_password,
                subjects,
            )
            teachers = self._create_teachers(
                teachers_to_create,
                hashed_password,
                subjects,
            )
            watched_rows = self._seed_watch_history(
                created_student_ids,
                min_watched,
                max_watched,
                completion_rate,
            )

        self.stdout.write(self.style.SUCCESS("User seeding complete."))
        self.stdout.write(
            f"Students created: {students}; Teachers created: {teachers}; "
            f"Watched rows seeded: {watched_rows}"
        )

    def _create_students(self, count, hashed_password, subjects):
        if count <= 0:
            return 0, []

        existing_usernames = set(User.objects.values_list("username", flat=True))
        existing_rolls = set(StudentProfile.objects.values_list("roll_number", flat=True))
        
        users = []
        created_usernames = []
        roll_map = {}

        for i in range(1, count + 1):
            fname = random.choice(FIRST_NAMES)
            lname = random.choice(LAST_NAMES)
            clean_first = re.sub(r'[^a-zA-Z0-9]', '', fname).lower()
            clean_last = re.sub(r'[^a-zA-Z0-9]', '', lname).lower()

            base_username = f"{clean_first}.{clean_last}"
            username = base_username
            num = 1
            while username in existing_usernames:
                num += 1
                username = f"{base_username}{num}"
            existing_usernames.add(username)
            created_usernames.append(username)

            dept = DEPARTMENTS[i % len(DEPARTMENTS)]
            year = 2024
            roll = f"{year}{dept}{i:04d}"
            while roll in existing_rolls:
                i += 100000
                roll = f"{year}{dept}{i:04d}"
            existing_rolls.add(roll)
            roll_map[username] = roll

            users.append(
                User(
                    username=username,
                    first_name=fname,
                    last_name=lname,
                    email=f"{username}@fliplearn.edu",
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
            profiles.append(
                StudentProfile(
                    user_id=user_id,
                    roll_number=roll_map.get(username, f"ROLL_{user_id}"),
                    semester=random.randint(1, 8),
                    previous_gpa=round(random.uniform(5.0, 9.8), 2),
                    phone=f"9{random.randint(100000000, 999999999)}",
                )
            )

        StudentProfile.objects.bulk_create(profiles, batch_size=500)

        if subjects:
            ThroughModel = StudentProfile.enrolled_subjects.through
            student_profiles = list(
                StudentProfile.objects.filter(user__username__in=created_usernames).values_list('id', flat=True)
            )
            enrollments = []
            for profile_id in student_profiles:
                pick = random.sample(subjects, k=min(len(subjects), random.randint(2, 4)))
                for s in pick:
                    enrollments.append(
                        ThroughModel(studentprofile_id=profile_id, subject_id=s.id)
                    )
            ThroughModel.objects.bulk_create(enrollments, batch_size=2000)

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
                    "No active videos found; skipping watched-video data."
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

        VideoWatchHistory.objects.bulk_create(rows, batch_size=2000, ignore_conflicts=True)
        return len(rows)

    def _create_teachers(self, count, hashed_password, subjects):
        if count <= 0:
            return 0

        existing_usernames = set(User.objects.values_list("username", flat=True))
        existing_emp_ids = set(TeacherProfile.objects.values_list("employee_id", flat=True))

        users = []
        created_usernames = []
        emp_map = {}

        for i in range(1, count + 1):
            fname = random.choice(FIRST_NAMES)
            lname = random.choice(LAST_NAMES)
            clean_first = re.sub(r'[^a-zA-Z0-9]', '', fname).lower()
            clean_last = re.sub(r'[^a-zA-Z0-9]', '', lname).lower()

            base_username = f"prof.{clean_first}.{clean_last}"
            username = base_username
            num = 1
            while username in existing_usernames:
                num += 1
                username = f"{base_username}{num}"
            existing_usernames.add(username)
            created_usernames.append(username)

            emp_id = f"EMP{i:04d}"
            while emp_id in existing_emp_ids:
                i += 1000
                emp_id = f"EMP{i:04d}"
            existing_emp_ids.add(emp_id)
            emp_map[username] = emp_id

            users.append(
                User(
                    username=username,
                    first_name=fname,
                    last_name=lname,
                    email=f"{username}@fliplearn.edu",
                    password=hashed_password,
                    is_staff=True,
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
            profiles.append(
                TeacherProfile(
                    user_id=user_id,
                    employee_id=emp_map.get(username, f"EMP_{user_id}"),
                    designation=random.choice(
                        [
                            "Assistant Professor",
                            "Associate Professor",
                            "Professor",
                        ]
                    ),
                )
            )

        TeacherProfile.objects.bulk_create(profiles, batch_size=500)

        if subjects:
            ThroughModel = TeacherProfile.subjects.through
            teacher_profiles = list(
                TeacherProfile.objects.filter(user__username__in=created_usernames).values_list('id', flat=True)
            )
            assignments = []
            for profile_id in teacher_profiles:
                pick = random.sample(subjects, k=min(len(subjects), random.randint(1, 3)))
                for s in pick:
                    assignments.append(
                        ThroughModel(teacherprofile_id=profile_id, subject_id=s.id)
                    )
            ThroughModel.objects.bulk_create(assignments, batch_size=2000)

        return len(users)
