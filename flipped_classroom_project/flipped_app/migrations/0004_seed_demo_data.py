"""
Demo data migration — seeds a teacher account plus video lectures,
quizzes (with questions), and assignments for each subject.
Runs automatically via `python manage.py migrate` on first deploy.
"""

from django.db import migrations
from django.utils import timezone
import datetime

# ── Video lectures (YouTube URLs) ────────────────────────────────────────────
VIDEOS = [
    # Data Structures
    ('DS', 'Introduction to Arrays and Linked Lists',
     'Fundamentals of arrays and linked list data structures.',
     'https://www.youtube.com/watch?v=RBSGKlAvoiM', 75),
    ('DS', 'Stacks and Queues Explained',
     'Stack and queue operations with real-world examples.',
     'https://www.youtube.com/watch?v=wjI1WNcIntg', 62),
    ('DS', 'Binary Trees and BST',
     'Binary search trees: insertion, deletion, and traversal.',
     'https://www.youtube.com/watch?v=oSWTXtMglKE', 88),

    # Python Programming
    ('PY', 'Python Basics – Variables and Data Types',
     'Core Python syntax: variables, strings, lists, and dicts.',
     'https://www.youtube.com/watch?v=kqtD5dpn9C8', 60),
    ('PY', 'Functions and Modules in Python',
     'Defining functions, scope, and organising code into modules.',
     'https://www.youtube.com/watch?v=9Os0o3wzS_I', 55),
    ('PY', 'Object-Oriented Python',
     'Classes, inheritance, and polymorphism in Python.',
     'https://www.youtube.com/watch?v=JeznW_7DlB0', 70),

    # Web Development
    ('WD', 'HTML & CSS Crash Course',
     'Build your first webpage with HTML5 and CSS3.',
     'https://www.youtube.com/watch?v=916GWv2Qs08', 65),
    ('WD', 'JavaScript DOM Manipulation',
     'Selecting elements, events, and dynamic page updates.',
     'https://www.youtube.com/watch?v=5fb2aPlgoys', 58),
    ('WD', 'Introduction to Django',
     'Build a web app with Django MVT architecture.',
     'https://www.youtube.com/watch?v=rHux0gMZ3Eg', 90),

    # Computer Networks
    ('CN', 'OSI Model – 7 Layers Explained',
     'Overview of the OSI reference model layers.',
     'https://www.youtube.com/watch?v=vv4y_uOneC0', 50),
    ('CN', 'TCP/IP Protocol Suite',
     'How TCP/IP underpins the modern internet.',
     'https://www.youtube.com/watch?v=PpsEaqJV_A0', 68),
    ('CN', 'DNS and HTTP Explained',
     'Domain name resolution and HyperText Transfer Protocol.',
     'https://www.youtube.com/watch?v=al5B7nZSvzs', 55),

    # Data Science
    ('DSC', 'Introduction to Data Science',
     'What is data science? Tools, workflow, and career paths.',
     'https://www.youtube.com/watch?v=X3paOmcrTjQ', 52),
    ('DSC', 'Pandas for Data Analysis',
     'DataFrames, groupby, merging, and cleaning with Pandas.',
     'https://www.youtube.com/watch?v=vmEHCJofslg', 80),
    ('DSC', 'Data Visualisation with Matplotlib',
     'Line charts, bar graphs, scatter plots, and heatmaps.',
     'https://www.youtube.com/watch?v=3Xc3CA655Y4', 63),

    # AI & ML
    ('AIML', 'Machine Learning Overview',
     'Supervised, unsupervised, and reinforcement learning concepts.',
     'https://www.youtube.com/watch?v=ukzFI9rgwfU', 56),
    ('AIML', 'Linear Regression from Scratch',
     'Cost function, gradient descent, and model evaluation.',
     'https://www.youtube.com/watch?v=4b4MUYve_U8', 75),
    ('AIML', 'Neural Networks and Deep Learning Basics',
     'Perceptrons, activation functions, backpropagation.',
     'https://www.youtube.com/watch?v=aircAruvnKk', 82),
]

# ── Quizzes + questions ──────────────────────────────────────────────────────
QUIZZES = {
    'DS': {
        'title': 'Data Structures Quiz 1',
        'description': 'Test your knowledge of fundamental data structures.',
        'questions': [
            ('Which data structure uses LIFO order?',
             'Queue', 'Stack', 'Array', 'Linked List', 'B'),
            ('Time complexity of binary search on a sorted array?',
             'O(n)', 'O(n²)', 'O(log n)', 'O(1)', 'C'),
            ('Which traversal visits root first?',
             'Inorder', 'Postorder', 'Preorder', 'Level-order', 'C'),
            ('A full binary tree with n leaves has how many internal nodes?',
             'n', 'n-1', 'n+1', '2n', 'B'),
            ('Which structure is best for implementing a priority queue?',
             'Stack', 'Queue', 'Heap', 'Array', 'C'),
        ],
    },
    'PY': {
        'title': 'Python Fundamentals Quiz',
        'description': 'Core Python concepts and syntax.',
        'questions': [
            ('What is the output of type([])?',
             "<class 'list'>", "<class 'tuple'>", "<class 'dict'>", "<class 'set'>", 'A'),
            ('Which keyword defines a function in Python?',
             'func', 'define', 'def', 'function', 'C'),
            ('What does len("hello") return?',
             '4', '5', '6', 'Error', 'B'),
            ('Which of these is immutable in Python?',
             'list', 'dict', 'set', 'tuple', 'D'),
            ('What symbol is used for single-line comments?',
             '//', '#', '/*', '--', 'B'),
        ],
    },
    'WD': {
        'title': 'Web Development Basics Quiz',
        'description': 'HTML, CSS, and JavaScript fundamentals.',
        'questions': [
            ('Which tag creates a hyperlink in HTML?',
             '<link>', '<a>', '<href>', '<url>', 'B'),
            ('CSS property to change text colour?',
             'font-color', 'text-color', 'color', 'foreground', 'C'),
            ('JavaScript method to select an element by ID?',
             'querySelector', 'getElementById', 'getElement', 'selectId', 'B'),
            ('HTTP status code for "Not Found"?',
             '200', '301', '404', '500', 'C'),
            ('Which HTTP method is used to submit form data?',
             'GET', 'POST', 'PUT', 'DELETE', 'B'),
        ],
    },
    'CN': {
        'title': 'Computer Networks Quiz',
        'description': 'Network protocols, OSI model, and TCP/IP.',
        'questions': [
            ('Which layer of OSI handles routing?',
             'Physical', 'Data Link', 'Network', 'Transport', 'C'),
            ('Full form of DNS?',
             'Dynamic Network Service', 'Domain Name System',
             'Distributed Name Server', 'Data Node System', 'B'),
            ('Which protocol is connectionless?',
             'TCP', 'FTP', 'UDP', 'HTTP', 'C'),
            ('Default port for HTTP?',
             '21', '22', '80', '443', 'C'),
            ('What does IP stand for?',
             'Internet Protocol', 'Internal Protocol',
             'Integrated Protocol', 'Internet Port', 'A'),
        ],
    },
    'DSC': {
        'title': 'Data Science Fundamentals Quiz',
        'description': 'Statistics, pandas, and data analysis basics.',
        'questions': [
            ('Which library is primary used for data manipulation in Python?',
             'NumPy', 'Pandas', 'Matplotlib', 'Scikit-learn', 'B'),
            ('Mean of [2, 4, 6, 8, 10]?',
             '4', '5', '6', '7', 'C'),
            ('Which chart best shows distribution of a single variable?',
             'Bar chart', 'Pie chart', 'Histogram', 'Line chart', 'C'),
            ('Full form of CSV?',
             'Comma Separated Values', 'Column Separated Values',
             'Computed Statistical Values', 'Categorical Sorted Values', 'A'),
            ('Which Pandas method removes missing values?',
             'drop()', 'remove()', 'dropna()', 'fillna()', 'C'),
        ],
    },
    'AIML': {
        'title': 'AI & Machine Learning Quiz',
        'description': 'ML algorithms, evaluation, and deep learning basics.',
        'questions': [
            ('Which algorithm is used for classification and regression?',
             'K-Means', 'Decision Tree', 'PCA', 'Apriori', 'B'),
            ('Overfitting occurs when a model?',
             'Performs poorly on training data',
             'Performs well on training but poorly on test data',
             'Performs well on both training and test data',
             'Has too few parameters', 'B'),
            ('Which metric measures classification accuracy?',
             'MSE', 'RMSE', 'F1-Score', 'R²', 'C'),
            ('What is the activation function used in output layer for binary classification?',
             'ReLU', 'Tanh', 'Sigmoid', 'Softmax', 'C'),
            ('Gradient descent minimizes?',
             'Accuracy', 'Loss function', 'Learning rate', 'Epochs', 'B'),
        ],
    },
}

# ── Assignments ───────────────────────────────────────────────────────────────
ASSIGNMENTS = [
    ('DS',   'Assignment 1: Implement a Stack using Array',
     'Implement push, pop, peek, and isEmpty operations for a stack using a Python list. '
     'Submit a .py file with a working Stack class and test cases.',
     20),
    ('DS',   'Assignment 2: Linked List Operations',
     'Implement a singly linked list with insert, delete, and reverse methods.',
     20),
    ('PY',   'Assignment 1: File Handling & Exception Handling',
     'Write a Python program that reads a CSV file, processes data, and handles '
     'FileNotFoundError and ValueError exceptions gracefully.',
     20),
    ('PY',   'Assignment 2: OOP – Bank Account System',
     'Design a BankAccount class with deposit, withdraw, and statement methods '
     'using OOP principles (encapsulation, inheritance).',
     20),
    ('WD',   'Assignment 1: Personal Portfolio Page',
     'Build a responsive HTML/CSS personal portfolio page with sections for '
     'About, Skills, Projects, and Contact.',
     20),
    ('WD',   'Assignment 2: JavaScript To-Do App',
     'Create a to-do list web app using HTML, CSS, and vanilla JavaScript with '
     'add, complete, and delete functionality.',
     20),
    ('CN',   'Assignment 1: Network Topology Design',
     'Design a star, bus, and ring topology diagram for a 10-node network. '
     'Explain advantages and disadvantages of each.',
     20),
    ('CN',   'Assignment 2: Socket Programming',
     'Write a simple TCP client-server program in Python that exchanges messages.',
     20),
    ('DSC',  'Assignment 1: Exploratory Data Analysis',
     'Perform EDA on the Titanic dataset using Pandas and Matplotlib. '
     'Identify missing values, distributions, and correlations.',
     20),
    ('DSC',  'Assignment 2: Data Cleaning Pipeline',
     'Build a reusable data cleaning function that handles missing values, '
     'outliers, and type conversion for tabular datasets.',
     20),
    ('AIML', 'Assignment 1: Linear Regression from Scratch',
     'Implement linear regression using gradient descent in Python without '
     'using scikit-learn. Evaluate on the Boston Housing dataset.',
     20),
    ('AIML', 'Assignment 2: Classification with Scikit-learn',
     'Train, evaluate, and compare Logistic Regression, Decision Tree, and '
     'Random Forest classifiers on the Iris dataset.',
     20),
]


def seed_demo_data(apps, schema_editor):
    User = apps.get_model('auth', 'User')
    Subject = apps.get_model('flipped_app', 'Subject')
    TeacherProfile = apps.get_model('flipped_app', 'TeacherProfile')
    VideoLecture = apps.get_model('flipped_app', 'VideoLecture')
    Quiz = apps.get_model('flipped_app', 'Quiz')
    QuizQuestion = apps.get_model('flipped_app', 'QuizQuestion')
    Assignment = apps.get_model('flipped_app', 'Assignment')

    # ── Teacher user ─────────────────────────────────────────
    teacher_user, created = User.objects.get_or_create(
        username='teacher',
        defaults={
            'first_name': 'Demo',
            'last_name': 'Teacher',
            'email': 'teacher@fliplearn.edu',
            'is_staff': True,
        }
    )
    if created:
        teacher_user.set_password('teacher1234')
        teacher_user.save()

    if not TeacherProfile.objects.filter(user=teacher_user).exists():
        tp = TeacherProfile.objects.create(
            user=teacher_user,
            employee_id='EMP001',
            department='Computer Science & Engineering',
            designation='Assistant Professor',
        )
        for subj in Subject.objects.all():
            tp.subjects.add(subj)

    # ── Video lectures ────────────────────────────────────────
    for code, title, desc, url, duration in VIDEOS:
        subj = Subject.objects.filter(code=code).first()
        if subj and not VideoLecture.objects.filter(title=title, subject=subj).exists():
            VideoLecture.objects.create(
                subject=subj,
                title=title,
                description=desc,
                youtube_url=url,
                duration_minutes=duration,
                uploaded_by=teacher_user,
                is_active=True,
            )

    # ── Quizzes + questions ───────────────────────────────────
    due = timezone.now() + datetime.timedelta(days=30)
    for code, data in QUIZZES.items():
        subj = Subject.objects.filter(code=code).first()
        if not subj:
            continue
        quiz, created = Quiz.objects.get_or_create(
            title=data['title'],
            subject=subj,
            defaults={
                'description': data['description'],
                'total_marks': len(data['questions']),
                'time_limit_minutes': 20,
                'created_by': teacher_user,
                'is_active': True,
                'due_date': due,
            }
        )
        if created:
            for q_text, opt_a, opt_b, opt_c, opt_d, answer in data['questions']:
                QuizQuestion.objects.create(
                    quiz=quiz,
                    question_text=q_text,
                    option_a=opt_a,
                    option_b=opt_b,
                    option_c=opt_c,
                    option_d=opt_d,
                    correct_answer=answer,
                    marks=1,
                )

    # ── Assignments ───────────────────────────────────────────
    due_assign = timezone.now() + datetime.timedelta(days=21)
    for code, title, desc, marks in ASSIGNMENTS:
        subj = Subject.objects.filter(code=code).first()
        if subj and not Assignment.objects.filter(title=title, subject=subj).exists():
            Assignment.objects.create(
                subject=subj,
                title=title,
                description=desc,
                total_marks=marks,
                due_date=due_assign,
                created_by=teacher_user,
            )


def noop(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ('flipped_app', '0003_seed_subjects'),
    ]

    operations = [
        migrations.RunPython(seed_demo_data, noop),
    ]
