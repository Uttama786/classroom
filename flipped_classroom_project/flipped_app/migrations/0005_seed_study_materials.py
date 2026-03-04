"""
Migration 0005 — seeds study materials using the real rag_knowledge/ .txt files
that are already committed to the repo (AIML.txt, CN.txt, DS.txt, DSC.txt, PY.txt, WD.txt).
Copies them into MEDIA_ROOT/materials/ so they are downloadable.
"""

import os
import shutil
import pathlib
from django.db import migrations
from django.conf import settings

# Maps subject code → (rag_knowledge filename, display title, description)
MATERIALS = [
    ('DS',   'DS.txt',   'Data Structures – Lecture Notes',
     'Complete notes on arrays, linked lists, stacks, queues, trees, graphs and sorting.'),
    ('PY',   'PY.txt',   'Python Programming – Lecture Notes',
     'Core Python: syntax, OOP, file handling, modules, decorators and generators.'),
    ('WD',   'WD.txt',   'Web Development – Lecture Notes',
     'HTML5, CSS3, JavaScript, DOM manipulation, and Django MVT architecture.'),
    ('CN',   'CN.txt',   'Computer Networks – Lecture Notes',
     'OSI model, TCP/IP, IP addressing, routing protocols and network security.'),
    ('DSC',  'DSC.txt',  'Data Science – Lecture Notes',
     'NumPy, Pandas, Matplotlib, statistics, and scikit-learn with worked examples.'),
    ('AIML', 'AIML.txt', 'AI & Machine Learning – Lecture Notes',
     'Supervised/unsupervised learning, neural networks, backpropagation and evaluation.'),
]


def seed_materials(apps, schema_editor):
    User = apps.get_model('auth', 'User')
    Subject = apps.get_model('flipped_app', 'Subject')
    StudyMaterial = apps.get_model('flipped_app', 'StudyMaterial')

    teacher = User.objects.filter(username='teacher').first()

    # Locate rag_knowledge/ relative to this migration file
    # flipped_classroom_project/flipped_app/migrations/ → PROJECT_ROOT/rag_knowledge/
    migrations_dir = pathlib.Path(__file__).resolve().parent          # migrations/
    project_root   = migrations_dir.parent.parent.parent              # repo root
    knowledge_dir  = project_root / 'rag_knowledge'

    # Destination inside MEDIA_ROOT
    media_materials = pathlib.Path(settings.MEDIA_ROOT) / 'materials'
    media_materials.mkdir(parents=True, exist_ok=True)

    for code, filename, title, desc in MATERIALS:
        subj = Subject.objects.filter(code=code).first()
        if not subj:
            continue
        if StudyMaterial.objects.filter(title=title, subject=subj).exists():
            continue

        src  = knowledge_dir / filename
        dest = media_materials / filename

        # Copy the real txt file into media/materials/ if it exists
        if src.exists() and not dest.exists():
            shutil.copy2(str(src), str(dest))

        # The FileField value is relative to MEDIA_ROOT
        file_rel = f'materials/{filename}' if (dest.exists() or src.exists()) else ''

        StudyMaterial.objects.create(
            subject=subj,
            title=title,
            description=desc,
            file=file_rel,
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
