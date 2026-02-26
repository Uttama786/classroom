from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import (
    StudentProfile, TeacherProfile, VideoLecture,
    StudyMaterial, Quiz, QuizQuestion, Assignment,
    AssignmentSubmission, QuizAttempt, Subject
)


class StudentRegistrationForm(UserCreationForm):
    _fc = {'class': 'form-control'}
    first_name   = forms.CharField(max_length=50, required=True,  widget=forms.TextInput(attrs=_fc))
    last_name    = forms.CharField(max_length=50, required=True,  widget=forms.TextInput(attrs=_fc))
    email        = forms.EmailField(required=True,                 widget=forms.EmailInput(attrs=_fc))
    roll_number  = forms.CharField(max_length=20,                  widget=forms.TextInput(attrs=_fc))
    semester     = forms.IntegerField(min_value=1, max_value=8,    widget=forms.NumberInput(attrs=_fc))
    phone        = forms.CharField(max_length=15, required=False,  widget=forms.TextInput(attrs=_fc))
    previous_gpa = forms.FloatField(min_value=0.0, max_value=10.0, widget=forms.NumberInput(attrs={**_fc, 'step': '0.01'}))

    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'password1', 'password2']
        widgets = {
            'username': forms.TextInput(attrs={'class': 'form-control'}),
        }

    def save(self, commit=True):
        user = super().save(commit=False)
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
            StudentProfile.objects.create(
                user=user,
                roll_number=self.cleaned_data['roll_number'],
                semester=self.cleaned_data['semester'],
                phone=self.cleaned_data.get('phone', ''),
                previous_gpa=self.cleaned_data.get('previous_gpa', 0.0),
            )
        return user


class VideoLectureForm(forms.ModelForm):
    class Meta:
        model = VideoLecture
        fields = ['subject', 'title', 'description', 'video_file', 'youtube_url', 'duration_minutes']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
        }


class StudyMaterialForm(forms.ModelForm):
    class Meta:
        model = StudyMaterial
        fields = ['subject', 'title', 'description', 'file']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
        }


class QuizForm(forms.ModelForm):
    class Meta:
        model = Quiz
        fields = ['subject', 'title', 'description', 'total_marks', 'time_limit_minutes', 'due_date']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
            'due_date': forms.DateTimeInput(attrs={'type': 'datetime-local'}),
        }


class QuizQuestionForm(forms.ModelForm):
    class Meta:
        model = QuizQuestion
        fields = ['question_text', 'option_a', 'option_b', 'option_c', 'option_d', 'correct_answer', 'marks']
        widgets = {
            'question_text': forms.Textarea(attrs={'rows': 2}),
        }


class AssignmentForm(forms.ModelForm):
    class Meta:
        model = Assignment
        fields = ['subject', 'title', 'description', 'total_marks', 'due_date', 'attachment']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 4}),
            'due_date': forms.DateTimeInput(attrs={'type': 'datetime-local'}),
        }


class AssignmentSubmissionForm(forms.ModelForm):
    class Meta:
        model = AssignmentSubmission
        fields = ['submitted_file']


class GradeSubmissionForm(forms.ModelForm):
    class Meta:
        model = AssignmentSubmission
        fields = ['marks_obtained', 'feedback']
        widgets = {
            'feedback': forms.Textarea(attrs={'rows': 3}),
        }
