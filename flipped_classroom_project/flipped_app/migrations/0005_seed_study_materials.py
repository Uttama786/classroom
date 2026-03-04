"""
Migration 0005 — seeds demo study materials (notes/PDFs) for all 6 subjects.
Files are stored as placeholder paths; real PDFs can be uploaded via the teacher panel.
"""

from django.db import migrations

MATERIALS = [
    # (subject_code, title, description, fake_filename)
    ('DS', 'Data Structures – Complete Notes',
     'Comprehensive notes covering arrays, linked lists, stacks, queues, trees, graphs and sorting algorithms.',
     'materials/DS_Complete_Notes.pdf'),
    ('DS', 'Algorithm Complexity Cheat Sheet',
     'Quick reference for Big-O complexity of common data structure operations.',
     'materials/DS_Complexity_Cheatsheet.pdf'),

    ('PY', 'Python Programming Handbook',
     'From Python basics to OOP, file handling, decorators and generators.',
     'materials/PY_Programming_Handbook.pdf'),
    ('PY', 'Python Standard Library Reference',
     'Useful built-in modules: os, sys, json, csv, collections, itertools.',
     'materials/PY_Stdlib_Reference.pdf'),

    ('WD', 'HTML5 & CSS3 Reference Guide',
     'Semantic HTML5 tags, CSS3 properties, Flexbox and Grid layout cheat sheet.',
     'materials/WD_HTML5_CSS3_Guide.pdf'),
    ('WD', 'JavaScript ES6+ Quick Reference',
     'Arrow functions, destructuring, promises, async/await, and modules.',
     'materials/WD_JavaScript_ES6.pdf'),

    ('CN', 'Computer Networks Lecture Notes',
     'OSI and TCP/IP models, IP addressing, subnetting, routing protocols, and network security.',
     'materials/CN_Lecture_Notes.pdf'),
    ('CN', 'Network Protocols Summary',
     'TCP, UDP, HTTP, FTP, DNS, DHCP — how each protocol works with port numbers.',
     'materials/CN_Protocols_Summary.pdf'),

    ('DSC', 'Data Science with Python – Notes',
     'NumPy, Pandas, Matplotlib, Seaborn, and scikit-learn walkthrough with examples.',
     'materials/DSC_Python_Notes.pdf'),
    ('DSC', 'Statistics for Data Science',
     'Descriptive statistics, probability distributions, hypothesis testing, and regression.',
     'materials/DSC_Statistics_Notes.pdf'),

    ('AIML', 'Machine Learning Algorithms – Notes',
     'Supervised, unsupervised, and reinforcement learning. Linear regression to neural networks.',
     'materials/AIML_ML_Algorithms.pdf'),
    ('AIML', 'Deep Learning Fundamentals',
     'ANNs, CNNs, RNNs, backpropagation, loss functions, and optimisers explained.',
     'materials/AIML_Deep_Learning.pdf'),
]


def seed_materials(apps, schema_editor):
    User = apps.get_model('auth', 'User')
    Subject = apps.get_model('flipped_app', 'Subject')
    StudyMaterial = apps.get_model('flipped_app', 'StudyMaterial')

    teacher = User.objects.filter(username='teacher').first()

    for code, title, desc, filepath in MATERIALS:
        subj = Subject.objects.filter(code=code).first()
        if subj and not StudyMaterial.objects.filter(title=title, subject=subj).exists():
            StudyMaterial.objects.create(
                subject=subj,
                title=title,
                description=desc,
                file=filepath,
                uploaded_by=teacher,
            )


def noop(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ('flipped_app', '0004_seed_demo_data'),
    ]

    operations = [
        migrations.RunPython(seed_materials, noop),
    ]
