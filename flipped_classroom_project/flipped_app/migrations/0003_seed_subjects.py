from django.db import migrations


SUBJECTS = [
    ('DS',   'Data Structures',                        'Fundamental data structures and algorithms.'),
    ('PY',   'Python Programming',                     'Python language fundamentals and applications.'),
    ('WD',   'Web Development',                        'HTML, CSS, JavaScript, and web frameworks.'),
    ('CN',   'Computer Networks',                      'Network protocols, architecture, and security.'),
    ('DSC',  'Data Science',                           'Statistical analysis, data wrangling, and visualisation.'),
    ('AIML', 'Artificial Intelligence & Machine Learning', 'ML algorithms, deep learning, and AI applications.'),
]


def seed_subjects(apps, schema_editor):
    Subject = apps.get_model('flipped_app', 'Subject')
    for code, name, description in SUBJECTS:
        Subject.objects.get_or_create(code=code, defaults={'name': name, 'description': description})


def unseed_subjects(apps, schema_editor):
    # Leave subjects intact on reverse — safer for production rollback
    pass


class Migration(migrations.Migration):

    dependencies = [
        ('flipped_app', '0002_chatmessage'),
    ]

    operations = [
        migrations.RunPython(seed_subjects, unseed_subjects),
    ]
