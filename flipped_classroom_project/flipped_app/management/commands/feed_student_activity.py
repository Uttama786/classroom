"""
Management command to feed rich, realistic student activity data:
- Video Watch History
- Quiz Attempts & Scores
- Assignment Submissions & Graded Marks
- Attendance Records
- StudentPerformance Aggregates (used for ML model training & analytics)
- Chat Messages & Platform Feedback
"""

import random
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from django.db import transaction
from django.utils import timezone
from flipped_app.models import (
    Subject,
    VideoLecture,
    Quiz,
    QuizAttempt,
    Assignment,
    AssignmentSubmission,
    VideoWatchHistory,
    Attendance,
    StudentPerformance,
    StudentProfile,
    ChatMessage,
    Feedback,
)

FEEDBACK_COMMENTS = [
    "Excellent work! Code is well-structured and properly commented.",
    "Good submission. Thorough answers and correct logic.",
    "Well done. Proper handling of edge cases and clean formatting.",
    "Good attempt. Needs slight improvement in algorithm efficiency.",
    "Decent effort. Please review test cases for better coverage.",
    "Satisfactory work. Make sure to follow naming conventions.",
    "Needs more detail in theoretical explanation, but code works well.",
    "Great effort and clear documentation.",
]

CHAT_QUERIES = [
    ("Explain the difference between BFS and DFS with time complexities.", "BFS explores layer-by-layer using a queue with O(V+E) complexity, whereas DFS explores deep into branches using a stack or recursion with O(V+E) complexity."),
    ("How does binary search work on a sorted array?", "Binary search compares the target value to the middle element. If not equal, the half in which the target cannot lie is eliminated, achieving O(log n) time complexity."),
    ("What are ACID properties in Database Management Systems?", "ACID stands for Atomicity, Consistency, Isolation, and Durability, ensuring reliable database transactions."),
    ("Explain gradient descent optimization in machine learning.", "Gradient descent iteratively adjusts model parameters in the direction of the negative gradient of the loss function to find a local or global minimum."),
    ("What is the difference between TCP and UDP?", "TCP is a connection-oriented, reliable protocol with error-checking and flow control, whereas UDP is connectionless, faster, but without delivery guarantees."),
    ("How do Python decorators work?", "Decorators in Python are higher-order functions that wrap another function to extend or modify its behavior without permanently altering the original function code."),
]

class Command(BaseCommand):
    help = "Feed realistic video seen data, quiz attempts, assignments, attendance, and student performance metrics."

    def add_arguments(self, parser):
        parser.add_argument(
            "--max-students",
            type=int,
            default=2000,
            help="Maximum number of students to populate with rich activity (default: 2000 for fast seeding).",
        )

    def handle(self, *args, **options):
        max_students = options["max_students"]
        rng = random.Random(42)

        self.stdout.write("Fetching enrolled students and subjects...")
        profiles = list(
            StudentProfile.objects.select_related("user")
            .prefetch_related("enrolled_subjects")
            .filter(enrolled_subjects__isnull=False)
            .distinct()[:max_students]
        )

        total_students = len(profiles)
        if total_students == 0:
            self.stdout.write(self.style.WARNING("No enrolled students found. Run seed_dummy_users first."))
            return

        self.stdout.write(f"Populating activity data for {total_students} students...")

        # Pre-fetch resources by subject
        subjects = list(Subject.objects.all())
        subject_videos = {s.id: list(VideoLecture.objects.filter(subject=s, is_active=True)) for s in subjects}
        subject_quizzes = {s.id: list(Quiz.objects.filter(subject=s, is_active=True)) for s in subjects}
        subject_assignments = {s.id: list(Assignment.objects.filter(subject=s)) for s in subjects}

        # Date range for attendance (last 20 weekdays)
        today = timezone.now().date()
        lecture_dates = []
        d = today - timedelta(days=40)
        while len(lecture_dates) < 20 and d <= today:
            if d.weekday() < 5:  # Monday to Friday
                lecture_dates.append(d)
            d += timedelta(days=1)

        watch_history_batch = []
        quiz_attempts_batch = []
        submissions_batch = []
        attendance_batch = []
        performance_batch = []
        chat_batch = []
        feedback_batch = []

        now = timezone.now()

        for idx, profile in enumerate(profiles):
            user = profile.user
            gpa = profile.previous_gpa or round(rng.uniform(5.5, 9.5), 2)
            # Student capability factor (0.4 to 1.0) based on GPA
            capability = min(1.0, max(0.4, (gpa / 10.0) + rng.uniform(-0.1, 0.1)))

            enrolled = list(profile.enrolled_subjects.all())
            for subject in enrolled:
                videos = subject_videos.get(subject.id, [])
                quizzes = subject_quizzes.get(subject.id, [])
                assignments = subject_assignments.get(subject.id, [])

                # 1. Video Watch History
                v_watched_count = 0
                v_total_minutes = 0.0
                if videos:
                    pick_count = min(len(videos), rng.randint(4, max(4, int(len(videos) * capability))))
                    chosen_videos = rng.sample(videos, k=pick_count)
                    for v in chosen_videos:
                        duration = float(v.duration_minutes or rng.uniform(10, 35))
                        completed = rng.random() < (0.5 + 0.45 * capability)
                        if completed:
                            watched = round(duration * rng.uniform(0.85, 1.0), 1)
                            v_watched_count += 1
                        else:
                            watched = round(duration * rng.uniform(0.2, 0.7), 1)
                        v_total_minutes += watched
                        watch_history_batch.append(
                            VideoWatchHistory(
                                student=user,
                                video=v,
                                watch_duration_minutes=watched,
                                completed=completed,
                            )
                        )

                # 2. Quiz Attempts
                quiz_scores = []
                if quizzes:
                    pick_q_count = rng.randint(1, len(quizzes))
                    chosen_quizzes = rng.sample(quizzes, k=pick_q_count)
                    for q in chosen_quizzes:
                        total_m = float(q.total_marks or 10)
                        score_ratio = min(1.0, max(0.3, capability + rng.uniform(-0.15, 0.15)))
                        score = round(total_m * score_ratio, 1)
                        time_spent = round(rng.uniform(6.0, float(q.time_limit_minutes or 20)), 1)
                        quiz_scores.append((score / total_m) * 100.0)
                        quiz_attempts_batch.append(
                            QuizAttempt(
                                student=user,
                                quiz=q,
                                score=score,
                                time_taken_minutes=time_spent,
                            )
                        )

                # 3. Assignment Submissions
                assignment_scores = []
                if assignments:
                    pick_a_count = rng.randint(1, len(assignments))
                    chosen_assignments = rng.sample(assignments, k=pick_a_count)
                    for a in chosen_assignments:
                        total_m = float(a.total_marks or 20)
                        score_ratio = min(1.0, max(0.35, capability + rng.uniform(-0.12, 0.12)))
                        obtained = round(total_m * score_ratio, 1)
                        assignment_scores.append((obtained / total_m) * 100.0)
                        feedback_txt = rng.choice(FEEDBACK_COMMENTS)
                        submissions_batch.append(
                            AssignmentSubmission(
                                student=user,
                                assignment=a,
                                marks_obtained=obtained,
                                feedback=feedback_txt,
                                is_graded=True,
                            )
                        )

                # 4. Attendance
                present_count = 0
                for ldate in lecture_dates:
                    is_present = rng.random() < (0.65 + 0.32 * capability)
                    if is_present:
                        present_count += 1
                    attendance_batch.append(
                        Attendance(
                            student=user,
                            subject=subject,
                            date=ldate,
                            present=is_present,
                        )
                    )
                attendance_pct = round((present_count / len(lecture_dates)) * 100.0, 1) if lecture_dates else 85.0

                # 5. Aggregate StudentPerformance
                quiz_avg = round(sum(quiz_scores) / len(quiz_scores), 1) if quiz_scores else round(capability * 80.0, 1)
                assign_avg = round(sum(assignment_scores) / len(assignment_scores), 1) if assignment_scores else round(capability * 82.0, 1)
                participation = round(min(10.0, max(1.0, capability * 9.5 + rng.uniform(-0.5, 0.5))), 1)
                materials_dl = rng.randint(2, 9)

                # Final exam calculation: weighted average of components + noise
                final_score = round(
                    (0.25 * quiz_avg) +
                    (0.30 * assign_avg) +
                    (0.20 * attendance_pct) +
                    (0.15 * (gpa * 10.0)) +
                    (0.10 * (participation * 10.0)) +
                    rng.uniform(-3.0, 3.0),
                    1
                )
                final_score = min(98.0, max(25.0, final_score))

                if final_score >= 75:
                    label = "High"
                    at_risk = False
                elif final_score >= 50:
                    label = "Medium"
                    at_risk = False
                elif final_score >= 35:
                    label = "Low"
                    at_risk = False
                else:
                    label = "At-Risk"
                    at_risk = True

                performance_batch.append(
                    StudentPerformance(
                        student=user,
                        subject=subject,
                        videos_watched=v_watched_count,
                        total_video_time_minutes=round(v_total_minutes, 1),
                        materials_downloaded=materials_dl,
                        quiz_avg_score=quiz_avg,
                        assignment_avg_marks=assign_avg,
                        attendance_percentage=attendance_pct,
                        participation_score=participation,
                        previous_gpa=gpa,
                        final_exam_score=final_score,
                        performance_label=label,
                        is_at_risk=at_risk,
                    )
                )

            # Sample chat interactions
            if rng.random() < 0.25:
                q_text, a_text = rng.choice(CHAT_QUERIES)
                chat_subj = rng.choice(enrolled) if enrolled else None
                chat_batch.append(ChatMessage(student=user, subject=chat_subj, role="user", content=q_text))
                chat_batch.append(ChatMessage(student=user, subject=chat_subj, role="assistant", content=a_text, sources=chat_subj.name if chat_subj else "FlipLearn Knowledge"))

            # Sample feedback
            if rng.random() < 0.08:
                rating = rng.choice([4, 5, 5, 4, 3])
                cat = rng.choice(["videos", "materials", "quizzes", "platform", "general"])
                msg = rng.choice([
                    "The video explanations and RAG AI Tutor are very helpful for quick revision!",
                    "Great interactive platform for flipped learning. Quizzes are well structured.",
                    "Assignments give practical hands-on experience on complex concepts.",
                    "The AI Tutor responses are very fast and accurate for subject queries.",
                ])
                feedback_batch.append(Feedback(author=user, category=cat, rating=rating, message=msg))

        self.stdout.write(f"Writing records to database in batches...")
        with transaction.atomic():
            if watch_history_batch:
                self.stdout.write(f" - Saving {len(watch_history_batch)} VideoWatchHistory entries...")
                VideoWatchHistory.objects.bulk_create(watch_history_batch, batch_size=2000, ignore_conflicts=True)

            if quiz_attempts_batch:
                self.stdout.write(f" - Saving {len(quiz_attempts_batch)} QuizAttempt entries...")
                QuizAttempt.objects.bulk_create(quiz_attempts_batch, batch_size=2000, ignore_conflicts=True)

            if submissions_batch:
                self.stdout.write(f" - Saving {len(submissions_batch)} AssignmentSubmission entries...")
                AssignmentSubmission.objects.bulk_create(submissions_batch, batch_size=2000, ignore_conflicts=True)

            if attendance_batch:
                self.stdout.write(f" - Saving {len(attendance_batch)} Attendance records...")
                Attendance.objects.bulk_create(attendance_batch, batch_size=3000, ignore_conflicts=True)

            if performance_batch:
                self.stdout.write(f" - Saving {len(performance_batch)} StudentPerformance records...")
                StudentPerformance.objects.bulk_create(performance_batch, batch_size=2000, ignore_conflicts=True)

            if chat_batch:
                self.stdout.write(f" - Saving {len(chat_batch)} ChatMessage entries...")
                ChatMessage.objects.bulk_create(chat_batch, batch_size=1000, ignore_conflicts=True)

            if feedback_batch:
                self.stdout.write(f" - Saving {len(feedback_batch)} Feedback entries...")
                Feedback.objects.bulk_create(feedback_batch, batch_size=500, ignore_conflicts=True)

        self.stdout.write(self.style.SUCCESS("Student activity, quiz, assignment, attendance, and performance feeding COMPLETE!"))
