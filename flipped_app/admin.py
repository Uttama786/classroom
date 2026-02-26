from django.contrib import admin
from .models import (
    Subject, StudentProfile, TeacherProfile, VideoLecture, StudyMaterial,
    Quiz, QuizQuestion, Assignment, AssignmentSubmission, QuizAttempt,
    VideoWatchHistory, Attendance, StudentPerformance, Notification
)


@admin.register(Subject)
class SubjectAdmin(admin.ModelAdmin):
    list_display = ['name', 'code', 'created_at']
    search_fields = ['name', 'code']


@admin.register(StudentProfile)
class StudentProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'roll_number', 'semester', 'department']
    search_fields = ['user__username', 'roll_number']
    list_filter = ['semester', 'department']


@admin.register(TeacherProfile)
class TeacherProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'employee_id', 'designation']


@admin.register(VideoLecture)
class VideoLectureAdmin(admin.ModelAdmin):
    list_display = ['title', 'subject', 'duration_minutes', 'uploaded_by', 'uploaded_at']
    list_filter = ['subject', 'is_active']
    search_fields = ['title']


@admin.register(StudyMaterial)
class StudyMaterialAdmin(admin.ModelAdmin):
    list_display = ['title', 'subject', 'uploaded_by', 'uploaded_at']
    list_filter = ['subject']


@admin.register(Quiz)
class QuizAdmin(admin.ModelAdmin):
    list_display = ['title', 'subject', 'total_marks', 'time_limit_minutes', 'is_active']
    list_filter = ['subject', 'is_active']


@admin.register(QuizQuestion)
class QuizQuestionAdmin(admin.ModelAdmin):
    list_display = ['quiz', 'question_text', 'correct_answer', 'marks']


@admin.register(Assignment)
class AssignmentAdmin(admin.ModelAdmin):
    list_display = ['title', 'subject', 'total_marks', 'due_date']
    list_filter = ['subject']


@admin.register(AssignmentSubmission)
class AssignmentSubmissionAdmin(admin.ModelAdmin):
    list_display = ['student', 'assignment', 'submitted_at', 'marks_obtained', 'is_graded']
    list_filter = ['is_graded']


@admin.register(QuizAttempt)
class QuizAttemptAdmin(admin.ModelAdmin):
    list_display = ['student', 'quiz', 'score', 'attempted_at']


@admin.register(VideoWatchHistory)
class VideoWatchHistoryAdmin(admin.ModelAdmin):
    list_display = ['student', 'video', 'completed', 'watch_duration_minutes']


@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ['student', 'subject', 'date', 'present']
    list_filter = ['subject', 'present']


@admin.register(StudentPerformance)
class StudentPerformanceAdmin(admin.ModelAdmin):
    list_display = [
        'student', 'subject', 'final_exam_score', 'performance_label',
        'predicted_label', 'is_at_risk'
    ]
    list_filter = ['performance_label', 'is_at_risk', 'subject']
    search_fields = ['student__username']


@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ['recipient', 'message', 'is_read', 'created_at']
