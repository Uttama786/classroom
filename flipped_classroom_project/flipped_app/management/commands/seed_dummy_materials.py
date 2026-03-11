"""
Management command: seed_dummy_materials
Generates dummy PDF lecture notes for every subject and seeds them into
the StudyMaterial table.  Existing entries with the same title are skipped
so the command is safe to run multiple times.

Usage:
    python manage.py seed_dummy_materials
    python manage.py seed_dummy_materials --force   # re-create files & DB rows
"""

import pathlib
import textwrap

from django.conf import settings
from django.contrib.auth.models import User
from django.core.management.base import BaseCommand


# ─── content blueprint per subject ──────────────────────────────────────────
MATERIALS = [
    {
        "subject_code": "DS",
        "title": "Data Structures – Lecture Notes",
        "description": "Complete notes on arrays, linked lists, stacks, queues, trees, graphs and sorting algorithms.",
        "filename": "DS_Lecture_Notes.pdf",
        "notes_filename": "DS_Quick_Notes.txt",
        "chapters": [
            ("1. Arrays & Strings",
             "An array stores elements of the same type in contiguous memory locations.\n"
             "Operations: access O(1), search O(n), insert/delete O(n).\n"
             "Multi-dimensional arrays, row-major vs column-major order."),
            ("2. Linked Lists",
             "Singly, doubly and circular linked lists.\n"
             "Node structure: data + pointer(s).  Insertion/deletion O(1) with pointer.\n"
             "Applications: LRU cache, polynomial arithmetic."),
            ("3. Stacks & Queues",
             "Stack – LIFO; operations push, pop, peek.  Queue – FIFO; enqueue, dequeue.\n"
             "Circular queue, deque (double-ended queue).\n"
             "Applications: expression evaluation, BFS, call stack."),
            ("4. Trees",
             "Binary Tree, BST, AVL Tree, Red-Black Tree, B-Tree.\n"
             "Tree traversals: inorder, preorder, postorder, level-order.\n"
             "Height, diameter, lowest common ancestor."),
            ("5. Graphs",
             "Directed & undirected graphs, adjacency matrix vs list.\n"
             "DFS, BFS, topological sort, Dijkstra, Bellman-Ford, Floyd-Warshall.\n"
             "Minimum spanning tree: Prim's and Kruskal's algorithms."),
            ("6. Sorting & Searching",
             "Bubble, selection, insertion, merge, quick, heap sort.\n"
             "Time complexities and stability comparison.\n"
             "Binary search, interpolation search, hashing."),
        ],
    },
    {
        "subject_code": "PY",
        "title": "Python Programming – Lecture Notes",
        "description": "Core Python: syntax, OOP, file handling, modules, decorators and generators.",
        "filename": "Python_Lecture_Notes.pdf",
        "notes_filename": "Python_Quick_Notes.txt",
        "chapters": [
            ("1. Basics",
             "Variables, data types (int, float, str, bool, NoneType).\n"
             "Operators, expressions, type conversion.\n"
             "Input / output with input() and print()."),
            ("2. Control Flow",
             "if-elif-else, while loop, for loop.\n"
             "break, continue, pass.  List comprehensions.\n"
             "Exception handling: try-except-finally, custom exceptions."),
            ("3. Functions & Modules",
             "def, return, *args, **kwargs, default arguments.\n"
             "Lambda functions, map(), filter(), reduce().\n"
             "Modules, packages, __name__ == '__main__'."),
            ("4. Object-Oriented Programming",
             "Classes, objects, __init__, self.\n"
             "Inheritance, polymorphism, encapsulation, abstraction.\n"
             "Magic/dunder methods: __str__, __len__, __repr__."),
            ("5. File Handling & Libraries",
             "open(), read(), write(), with statement.\n"
             "os, sys, datetime, math, random standard modules.\n"
             "pip, virtual environments, requirements.txt."),
            ("6. Advanced Topics",
             "Decorators, generators, iterators.\n"
             "Context managers, metaclasses.\n"
             "Async / await basics with asyncio."),
        ],
    },
    {
        "subject_code": "WD",
        "title": "Web Development – Lecture Notes",
        "description": "HTML5, CSS3, JavaScript, DOM manipulation, and Django MVT architecture.",
        "filename": "WebDev_Lecture_Notes.pdf",
        "notes_filename": "WebDev_Quick_Notes.txt",
        "chapters": [
            ("1. HTML5 Essentials",
             "Document structure: <!DOCTYPE>, <html>, <head>, <body>.\n"
             "Semantic elements: header, nav, main, article, footer.\n"
             "Forms, input types, validation attributes."),
            ("2. CSS3 & Responsive Design",
             "Selectors, specificity, box model, display, position.\n"
             "Flexbox and CSS Grid layouts.\n"
             "Media queries, Bootstrap grid system."),
            ("3. JavaScript & DOM",
             "Variables (var/let/const), functions, closures, promises.\n"
             "DOM selection: querySelector, getElementById.\n"
             "Events: addEventListener, event delegation, fetch API."),
            ("4. Django Framework",
             "MVT pattern: Model – View – Template.\n"
             "URL routing, views, template language (DTL).\n"
             "ORM basics: models, migrations, QuerySets."),
            ("5. REST APIs & AJAX",
             "HTTP methods: GET, POST, PUT, DELETE.\n"
             "Django REST Framework basics, JSON responses.\n"
             "Fetch / Axios for async front-end requests."),
            ("6. Deployment",
             "Static files, WhiteNoise, gunicorn.\n"
             "Environment variables, .env files.\n"
             "Deploying to Render / Railway."),
        ],
    },
    {
        "subject_code": "CN",
        "title": "Computer Networks – Lecture Notes",
        "description": "OSI model, TCP/IP, IP addressing, routing protocols and network security.",
        "filename": "ComputerNetworks_Lecture_Notes.pdf",
        "notes_filename": "CN_Quick_Notes.txt",
        "chapters": [
            ("1. Network Models",
             "OSI 7-layer model and functions of each layer.\n"
             "TCP/IP 4-layer model comparison.\n"
             "Encapsulation and decapsulation."),
            ("2. Data Link Layer",
             "Framing, error detection (CRC, checksum), error correction.\n"
             "MAC addresses, ARP, Ethernet.\n"
             "Switches, VLANs, Spanning Tree Protocol."),
            ("3. Network Layer & IP",
             "IPv4 vs IPv6 addressing, subnetting, CIDR.\n"
             "Routing: static, RIP, OSPF, BGP.\n"
             "NAT, ICMP, fragmentation."),
            ("4. Transport Layer",
             "TCP: connection setup (3-way handshake), flow control, congestion control.\n"
             "UDP: characteristics and use cases.\n"
             "Port numbers, sockets, multiplexing."),
            ("5. Application Layer",
             "DNS, HTTP/HTTPS, FTP, SMTP, POP3, IMAP.\n"
             "DHCP, SNMP, SSH.\n"
             "TLS/SSL handshake overview."),
            ("6. Network Security",
             "Threats: DoS, MitM, phishing, SQL injection.\n"
             "Firewalls, IDS/IPS, VPN, IPSec.\n"
             "Cryptography: symmetric, asymmetric, PKI."),
        ],
    },
    {
        "subject_code": "DSC",
        "title": "Data Science – Lecture Notes",
        "description": "NumPy, Pandas, Matplotlib, statistics, and scikit-learn with worked examples.",
        "filename": "DataScience_Lecture_Notes.pdf",
        "notes_filename": "DS_Science_Quick_Notes.txt",
        "chapters": [
            ("1. Statistics Foundations",
             "Mean, median, mode, variance, standard deviation.\n"
             "Normal distribution, z-scores, Central Limit Theorem.\n"
             "Hypothesis testing, p-values, confidence intervals."),
            ("2. NumPy",
             "ndarray creation, shape, dtype, broadcasting.\n"
             "Vectorised operations, slicing, fancy indexing.\n"
             "Linear algebra: dot, linalg, matrix ops."),
            ("3. Pandas",
             "Series and DataFrame creation, indexing (.loc, .iloc).\n"
             "Data cleaning: dropna, fillna, duplicates.\n"
             "GroupBy, merge, pivot tables."),
            ("4. Data Visualisation",
             "Matplotlib: figure, axes, plot, scatter, histogram.\n"
             "Seaborn: heatmap, pairplot, boxplot.\n"
             "Best practices for readable charts."),
            ("5. Machine Learning with scikit-learn",
             "Pipeline, train_test_split, cross_val_score.\n"
             "Linear regression, logistic regression, decision tree.\n"
             "Metrics: accuracy, precision, recall, F1, AUC-ROC."),
            ("6. Feature Engineering",
             "Encoding categorical variables (LabelEncoder, OneHotEncoder).\n"
             "Feature scaling (StandardScaler, MinMaxScaler).\n"
             "Dimensionality reduction: PCA, t-SNE."),
        ],
    },
    {
        "subject_code": "AIML",
        "title": "AI & Machine Learning – Lecture Notes",
        "description": "Supervised/unsupervised learning, neural networks, backpropagation and evaluation.",
        "filename": "AIML_Lecture_Notes.pdf",
        "notes_filename": "AIML_Quick_Notes.txt",
        "chapters": [
            ("1. Introduction to AI",
             "History of AI, Turing Test, AI vs ML vs DL.\n"
             "Types of AI: narrow, general, super.\n"
             "Search algorithms: BFS, DFS, A*."),
            ("2. Supervised Learning",
             "Regression: linear, polynomial, ridge, lasso.\n"
             "Classification: k-NN, Naive Bayes, SVM, decision trees.\n"
             "Ensemble methods: bagging, boosting, Random Forest, XGBoost."),
            ("3. Unsupervised Learning",
             "Clustering: K-Means, DBSCAN, hierarchical.\n"
             "Association rules: Apriori, FP-Growth.\n"
             "Dimensionality reduction: PCA, autoencoders."),
            ("4. Neural Networks",
             "Perceptron, multi-layer perceptron, activation functions.\n"
             "Backpropagation, gradient descent variants (SGD, Adam).\n"
             "Overfitting: dropout, batch norm, regularisation."),
            ("5. Deep Learning",
             "CNNs: convolution, pooling, VGG, ResNet.\n"
             "RNNs: LSTM, GRU, sequence modelling.\n"
             "Transformers, attention mechanism, BERT."),
            ("6. Model Evaluation & Deployment",
             "Bias-variance tradeoff, learning curves.\n"
             "Hyperparameter tuning: grid search, random search, Bayesian.\n"
             "Model serving: REST API, ONNX, TensorFlow Serving."),
        ],
    },
    {
        "subject_code": "PROJECT",
        "title": "Project Work – Guidelines & Notes",
        "description": "Software project lifecycle, documentation standards, and final-year project guidelines.",
        "filename": "Project_Guidelines.pdf",
        "notes_filename": "Project_Quick_Notes.txt",
        "chapters": [
            ("1. Project Lifecycle",
             "Waterfall, Agile (Scrum/Kanban), Spiral models.\n"
             "Phases: requirements, design, implementation, testing, deployment.\n"
             "Work breakdown structure (WBS)."),
            ("2. Requirements Engineering",
             "Functional vs non-functional requirements.\n"
             "Use-case diagrams, user stories.\n"
             "SRS document structure."),
            ("3. System Design",
             "Architecture patterns: MVC, microservices, event-driven.\n"
             "UML diagrams: class, sequence, activity, ERD.\n"
             "Database normalisation (1NF–3NF, BCNF)."),
            ("4. Implementation Best Practices",
             "Version control with Git: branching, merging, pull requests.\n"
             "Code review, linting, unit testing.\n"
             "CI/CD pipelines."),
            ("5. Testing",
             "Unit, integration, system, acceptance testing.\n"
             "Test cases, coverage, regression testing.\n"
             "Tools: pytest, Selenium, Postman."),
            ("6. Documentation & Presentation",
             "Technical report: abstract, intro, methodology, results.\n"
             "IEEE / ACM citation format.\n"
             "Presentation tips: slides, demo, Q&A preparation."),
        ],
    },
]


# ─── PDF generation using PyMuPDF (fitz) ────────────────────────────────────

def _make_pdf(dest_path: pathlib.Path, title: str, chapters: list) -> None:
    """Create a styled multi-page PDF using PyMuPDF."""
    import fitz  # PyMuPDF

    doc = fitz.open()

    # ── colour palette ──
    C_HEADER   = (0.09, 0.47, 0.25)   # green
    C_CHAPTER  = (0.10, 0.30, 0.60)   # blue
    C_TEXT     = (0.15, 0.15, 0.15)
    C_LIGHT    = (0.94, 0.97, 0.94)
    C_WHITE    = (1.0,  1.0,  1.0)
    PAGE_W, PAGE_H = 595, 842          # A4

    def new_page():
        pg = doc.new_page(width=PAGE_W, height=PAGE_H)
        # white background
        pg.draw_rect(fitz.Rect(0, 0, PAGE_W, PAGE_H), color=C_WHITE, fill=C_WHITE)
        return pg

    def wrap_lines(text: str, width_chars: int = 85) -> list:
        lines = []
        for para in text.split("\n"):
            lines.extend(textwrap.wrap(para, width_chars) or [""])
        return lines

    # ── Cover page ──────────────────────────────────────────────────────────
    pg = new_page()
    # header bar
    pg.draw_rect(fitz.Rect(0, 0, PAGE_W, 140), color=C_HEADER, fill=C_HEADER)
    pg.insert_text((40, 60),  "FlipLearn",  fontsize=22, color=C_WHITE,  fontname="helv")
    pg.insert_text((40, 90),  "Flipped Classroom Platform", fontsize=13, color=C_WHITE, fontname="helv")
    # light band
    pg.draw_rect(fitz.Rect(0, 140, PAGE_W, 230), color=C_LIGHT, fill=C_LIGHT)
    pg.insert_text((40, 190), title, fontsize=18, color=C_CHAPTER, fontname="helv")
    # body
    pg.insert_text((40, 260), "Course Lecture Notes", fontsize=13, color=C_TEXT, fontname="helv")
    pg.insert_text((40, 290), "Academic Year 2025-26",  fontsize=11, color=C_TEXT, fontname="helv")
    pg.insert_text((40, 320), "Department of Computer Science & Engineering", fontsize=11, color=C_TEXT, fontname="helv")
    # Table of contents
    pg.insert_text((40, 380), "Contents", fontsize=14, color=C_CHAPTER, fontname="helv")
    pg.draw_line(fitz.Point(40, 395), fitz.Point(PAGE_W - 40, 395), color=C_CHAPTER, width=1)
    y = 410
    for idx, (ch_title, _) in enumerate(chapters, 1):
        pg.insert_text((50, y), f"{idx}.  {ch_title}", fontsize=11, color=C_TEXT, fontname="helv")
        y += 22
    # footer
    pg.draw_rect(fitz.Rect(0, PAGE_H - 30, PAGE_W, PAGE_H), color=C_HEADER, fill=C_HEADER)
    pg.insert_text((40, PAGE_H - 12), "FlipLearn – Study Material  |  confidential", fontsize=9, color=C_WHITE, fontname="helv")

    # ── Chapter pages ────────────────────────────────────────────────────────
    for ch_title, ch_body in chapters:
        pg = new_page()
        # chapter header bar
        pg.draw_rect(fitz.Rect(0, 0, PAGE_W, 70), color=C_CHAPTER, fill=C_CHAPTER)
        pg.insert_text((40, 42), ch_title, fontsize=16, color=C_WHITE, fontname="helv")
        # body text
        y = 100
        for line in wrap_lines(ch_body):
            if y > PAGE_H - 50:
                # overflow to new page
                pg = new_page()
                pg.draw_rect(fitz.Rect(0, 0, PAGE_W, 40), color=C_CHAPTER, fill=C_CHAPTER)
                pg.insert_text((40, 26), f"(continued) {ch_title}", fontsize=12, color=C_WHITE, fontname="helv")
                y = 60
            pg.insert_text((50, y), line, fontsize=11, color=C_TEXT, fontname="helv")
            y += 18
        # footer
        pg.draw_rect(fitz.Rect(0, PAGE_H - 30, PAGE_W, PAGE_H), color=C_HEADER, fill=C_HEADER)
        pg.insert_text((40, PAGE_H - 12), f"FlipLearn – {title}", fontsize=9, color=C_WHITE, fontname="helv")

    doc.save(str(dest_path))
    doc.close()


# ─── plain-text quick notes ──────────────────────────────────────────────────

def _make_notes(dest_path: pathlib.Path, title: str, chapters: list) -> None:
    """Write a plain-text quick-reference notes file."""
    lines = [
        "=" * 70,
        f"  {title}",
        "  FlipLearn | Quick Reference Notes | 2025-26",
        "=" * 70,
        "",
    ]
    for ch_title, ch_body in chapters:
        lines += [
            "",
            f"{'─' * 60}",
            f"  {ch_title}",
            f"{'─' * 60}",
        ]
        for para in ch_body.split("\n"):
            for wrapped in textwrap.wrap(para, 68) or [""]:
                lines.append(f"  {wrapped}")
    lines += ["", "=" * 70, "  End of Notes", "=" * 70, ""]
    dest_path.write_text("\n".join(lines), encoding="utf-8")


# ─── Django management command ───────────────────────────────────────────────

class Command(BaseCommand):
    help = "Generate dummy PDF lecture notes and seed them into StudyMaterial."

    def add_arguments(self, parser):
        parser.add_argument(
            "--force",
            action="store_true",
            help="Overwrite existing files and DB rows.",
        )

    def handle(self, *args, **options):
        force = options["force"]

        media_dir = pathlib.Path(settings.MEDIA_ROOT) / "materials"
        media_dir.mkdir(parents=True, exist_ok=True)

        from flipped_app.models import Subject, StudyMaterial

        teacher = User.objects.filter(is_staff=True).first()

        created_count = 0
        skipped_count = 0

        for mat in MATERIALS:
            subj = Subject.objects.filter(code=mat["subject_code"]).first()
            if subj is None:
                self.stdout.write(
                    self.style.WARNING(
                        f"  Subject '{mat['subject_code']}' not found – skipping."
                    )
                )
                continue

            # ── PDF ──────────────────────────────────────────────────────────
            pdf_path = media_dir / mat["filename"]
            pdf_rel  = f"materials/{mat['filename']}"
            pdf_title = mat["title"]

            if StudyMaterial.objects.filter(title=pdf_title, subject=subj).exists() and not force:
                self.stdout.write(f"  [skip] {pdf_title}")
                skipped_count += 1
            else:
                if not pdf_path.exists() or force:
                    self.stdout.write(f"  [pdf]  Generating {mat['filename']} …")
                    _make_pdf(pdf_path, pdf_title, mat["chapters"])
                StudyMaterial.objects.update_or_create(
                    title=pdf_title,
                    subject=subj,
                    defaults={
                        "description": mat["description"],
                        "file": pdf_rel,
                        "uploaded_by": teacher,
                    },
                )
                created_count += 1
                self.stdout.write(self.style.SUCCESS(f"  [ok]   {pdf_title}"))

            # ── Notes (.txt) ─────────────────────────────────────────────────
            notes_path  = media_dir / mat["notes_filename"]
            notes_rel   = f"materials/{mat['notes_filename']}"
            notes_title = pdf_title.replace("– Lecture Notes", "– Quick Notes").replace(
                "– Guidelines & Notes", "– Quick Reference"
            )

            if StudyMaterial.objects.filter(title=notes_title, subject=subj).exists() and not force:
                self.stdout.write(f"  [skip] {notes_title}")
                skipped_count += 1
            else:
                if not notes_path.exists() or force:
                    self.stdout.write(f"  [txt]  Generating {mat['notes_filename']} …")
                    _make_notes(notes_path, notes_title, mat["chapters"])
                StudyMaterial.objects.update_or_create(
                    title=notes_title,
                    subject=subj,
                    defaults={
                        "description": mat["description"] + " (Quick-reference text version.)",
                        "file": notes_rel,
                        "uploaded_by": teacher,
                    },
                )
                created_count += 1
                self.stdout.write(self.style.SUCCESS(f"  [ok]   {notes_title}"))

        self.stdout.write("")
        self.stdout.write(
            self.style.SUCCESS(
                f"Done.  Created/updated: {created_count}  |  Skipped: {skipped_count}"
            )
        )
