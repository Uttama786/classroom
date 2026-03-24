"""
Management command: fix_missing_materials
Scans every StudyMaterial row in the database, and for any whose file is
missing on disk it generates a proper dummy PDF (using PyMuPDF) so the
Download button always works.

Usage:
    python manage.py fix_missing_materials             # generate only missing files
    python manage.py fix_missing_materials --all       # regenerate every PDF
"""

import pathlib
import textwrap

from django.conf import settings
from django.core.files.base import File
from django.core.management.base import BaseCommand


def _normalized_material_name(name: str) -> str:
    """Ensure material file names are stored as materials/<filename>."""
    clean = (name or "").replace("\\", "/").strip("/")
    filename = pathlib.PurePosixPath(clean).name
    return f"materials/{filename}" if filename else ""


def _storage_exists(material) -> bool:
    """Safely check storage existence across local and Cloudinary backends."""
    try:
        return bool(material.file and material.file.name and material.file.storage.exists(material.file.name))
    except Exception:
        return False


# ── subject-specific chapter content ─────────────────────────────────────────
SUBJECT_CONTENT = {
    "DS": [
        ("Arrays & Strings",
         "An array stores elements of the same type contiguously in memory.\n"
         "Access O(1), search O(n), insert/delete O(n).\n"
         "Multi-dimensional arrays, row-major vs column-major order."),
        ("Linked Lists",
         "Singly, doubly and circular linked lists.\n"
         "Node structure: data + pointer(s). Insert/delete O(1) with a held pointer.\n"
         "Applications: LRU cache, undo systems, polynomial arithmetic."),
        ("Stacks & Queues",
         "Stack – LIFO: push, pop, peek. Queue – FIFO: enqueue, dequeue.\n"
         "Circular queue, deque (double-ended queue).\n"
         "Applications: expression evaluation, BFS, call stack simulation."),
        ("Trees",
         "Binary Tree, BST, AVL Tree, Red-Black Tree, B-Tree.\n"
         "Traversals: inorder, preorder, postorder, level-order.\n"
         "Height, diameter, lowest common ancestor (LCA)."),
        ("Graphs",
         "Directed & undirected graphs, adjacency matrix vs adjacency list.\n"
         "DFS, BFS, topological sort, Dijkstra, Bellman-Ford, Floyd-Warshall.\n"
         "Minimum spanning tree: Prim's and Kruskal's algorithms."),
        ("Sorting & Searching",
         "Bubble, selection, insertion, merge, quick, heap sort.\n"
         "Time/space complexities and stability comparison table.\n"
         "Binary search, interpolation search, hashing and hash tables."),
    ],
    "PY": [
        ("Basics",
         "Variables, data types (int, float, str, bool, NoneType), operators.\n"
         "Type conversion: int(), float(), str(), bool().\n"
         "Input/output: input() and print() with f-strings."),
        ("Control Flow",
         "if-elif-else, while loop, for loop over iterables.\n"
         "break, continue, pass. List/dict/set comprehensions.\n"
         "Exception handling: try-except-else-finally, custom exceptions."),
        ("Functions & Modules",
         "def, return, *args, **kwargs, default and keyword arguments.\n"
         "Lambda functions, map(), filter(), functools.reduce().\n"
         "Modules, packages, __name__ == '__main__', import system."),
        ("Object-Oriented Programming",
         "Classes, objects, __init__, self, class vs instance attributes.\n"
         "Inheritance, multiple inheritance, MRO, polymorphism, encapsulation.\n"
         "Magic/dunder methods: __str__, __repr__, __len__, __eq__, __iter__."),
        ("File Handling & Standard Library",
         "open(), read(), readline(), write(), with statement (context manager).\n"
         "os, sys, pathlib, datetime, math, random, json, csv modules.\n"
         "pip, virtual environments (venv), requirements.txt, pyproject.toml."),
        ("Advanced Python",
         "Decorators (@wraps), generators (yield), iterators (__iter__/__next__).\n"
         "Context managers (__enter__/__exit__), metaclasses, descriptors.\n"
         "Async/await basics: asyncio, coroutines, event loop, gather()."),
    ],
    "WD": [
        ("HTML5 Essentials",
         "Document structure: <!DOCTYPE>, <html>, <head>, <body>.\n"
         "Semantic elements: header, nav, main, article, section, footer.\n"
         "Forms, input types (text, email, date, file), validation attributes."),
        ("CSS3 & Responsive Design",
         "Selectors, specificity, cascade, inheritance, box model.\n"
         "Display: block/inline/flex/grid. Position: static/relative/absolute/fixed.\n"
         "Flexbox and CSS Grid. Media queries, Bootstrap 5 grid and utilities."),
        ("JavaScript & DOM",
         "var/let/const, functions, arrow functions, closures, promises, async/await.\n"
         "DOM: querySelector, getElementById, createElement, event listeners.\n"
         "fetch() API, JSON handling, event delegation, localStorage."),
        ("Django Framework",
         "MVT pattern: Model – View – Template.\n"
         "URL routing (urls.py), class-based vs function-based views.\n"
         "ORM: models, migrations, QuerySets, Q objects, aggregation."),
        ("REST APIs & AJAX",
         "HTTP methods: GET, POST, PUT, PATCH, DELETE. Status codes.\n"
         "Django REST Framework: serializers, viewsets, routers, permissions.\n"
         "Consuming APIs from the front end with fetch / Axios."),
        ("Deployment",
         "Static files with collectstatic and WhiteNoise.\n"
         "gunicorn as WSGI server. Environment variables and .env files.\n"
         "Deploying to Render / Railway: Procfile, runtime.txt, build commands."),
    ],
    "CN": [
        ("Network Models",
         "OSI 7-layer model: Physical, Data Link, Network, Transport, Session, Presentation, Application.\n"
         "TCP/IP 4-layer model and mapping to OSI layers.\n"
         "Encapsulation (sender) and decapsulation (receiver)."),
        ("Data Link Layer",
         "Framing, error detection (CRC, checksum, parity), error correction (Hamming).\n"
         "MAC addresses, ARP (Address Resolution Protocol), Ethernet (IEEE 802.3).\n"
         "Switches, VLANs, Spanning Tree Protocol (STP), PoE."),
        ("Network Layer & IP",
         "IPv4: 32-bit addresses, subnetting, CIDR notation, private ranges.\n"
         "IPv6: 128-bit addresses, EUI-64, stateless autoconfiguration.\n"
         "Routing: static routes, RIP, OSPF, BGP. NAT, PAT, ICMP."),
        ("Transport Layer",
         "TCP: 3-way handshake (SYN/SYN-ACK/ACK), 4-way teardown, flow control (sliding window).\n"
         "Congestion control: slow start, congestion avoidance, fast retransmit.\n"
         "UDP: connectionless, best-effort, low overhead. Ports and sockets."),
        ("Application Layer",
         "DNS: hierarchy, recursive vs iterative queries, A/AAAA/MX/CNAME records.\n"
         "HTTP/1.1 vs HTTP/2 vs HTTP/3. FTP, SMTP, POP3, IMAP, DHCP, SSH.\n"
         "TLS/SSL: handshake, certificates, cipher suites."),
        ("Network Security",
         "Threats: DoS/DDoS, MitM, ARP spoofing, phishing, SQL injection.\n"
         "Defences: firewalls (stateful/stateless), IDS/IPS, VPN, IPSec.\n"
         "Cryptography: symmetric (AES), asymmetric (RSA), PKI, digital signatures."),
    ],
    "DSC": [
        ("Statistics Foundations",
         "Descriptive: mean, median, mode, variance, standard deviation, IQR.\n"
         "Inferential: Central Limit Theorem, z-scores, t-tests, ANOVA.\n"
         "Hypothesis testing: null/alternative hypothesis, p-value, confidence intervals."),
        ("NumPy",
         "ndarray creation: zeros, ones, arange, linspace, random.\n"
         "Shape, dtype, reshape, broadcasting rules.\n"
         "Vectorised operations, slicing, fancy indexing, linalg, einsum."),
        ("Pandas",
         "Series and DataFrame creation, indexing (.loc, .iloc, .at).\n"
         "Data cleaning: dropna, fillna, replace, duplicates, astype.\n"
         "GroupBy, merge/join, pivot_table, melt, apply, transform."),
        ("Data Visualisation",
         "Matplotlib: Figure, Axes, plot(), scatter(), hist(), bar(), subplots().\n"
         "Seaborn: heatmap, pairplot, boxplot, violinplot, regplot.\n"
         "Best practices: labels, colour palettes, accessibility, export resolution."),
        ("Machine Learning with scikit-learn",
         "Pipeline, ColumnTransformer, train_test_split, cross_val_score.\n"
         "Models: LinearRegression, LogisticRegression, DecisionTree, RandomForest.\n"
         "Metrics: accuracy, precision, recall, F1, AUC-ROC, confusion matrix."),
        ("Feature Engineering",
         "Encoding: LabelEncoder, OrdinalEncoder, OneHotEncoder, TargetEncoder.\n"
         "Scaling: StandardScaler, MinMaxScaler, RobustScaler.\n"
         "Dimensionality reduction: PCA, t-SNE, UMAP. Feature selection: RFE, SelectKBest."),
    ],
    "AIML": [
        ("Introduction to AI",
         "History of AI, Turing Test, AI winters, modern renaissance.\n"
         "AI vs ML vs DL. Narrow vs General vs Superintelligence.\n"
         "Uninformed search: BFS, DFS. Informed search: greedy, A*."),
        ("Supervised Learning",
         "Regression: linear, polynomial, ridge (L2), lasso (L1), elastic net.\n"
         "Classification: k-NN, Naive Bayes, SVM (kernel trick), decision trees.\n"
         "Ensemble: bagging (Random Forest), boosting (AdaBoost, XGBoost, LightGBM)."),
        ("Unsupervised Learning",
         "Clustering: K-Means (elbow method), DBSCAN (epsilon/minPts), hierarchical (dendrograms).\n"
         "Association rules: Apriori, FP-Growth, support, confidence, lift.\n"
         "Dimensionality reduction: PCA, t-SNE, autoencoders."),
        ("Neural Networks",
         "Perceptron, multi-layer perceptron (MLP), activation functions (ReLU, sigmoid, tanh, softmax).\n"
         "Backpropagation algorithm, gradient descent variants (SGD, Momentum, Adam, RMSProp).\n"
         "Overfitting remedies: dropout, batch normalisation, L1/L2 regularisation, early stopping."),
        ("Deep Learning",
         "CNNs: convolution, pooling, stride, padding. Architectures: VGG, ResNet, InceptionNet.\n"
         "RNNs: vanishing gradient problem. LSTM and GRU gating mechanisms.\n"
         "Transformers: self-attention, multi-head attention, positional encoding, BERT, GPT."),
        ("Model Evaluation & Deployment",
         "Bias-variance tradeoff, learning curves, validation curves.\n"
         "Hyperparameter tuning: grid search, random search, Bayesian optimisation (Optuna).\n"
         "Model serving: REST API (FastAPI/Flask), ONNX export, TensorFlow Serving, MLflow."),
    ],
    "AI": [
        ("Machine Learning Overview",
         "Supervised, unsupervised, semi-supervised, reinforcement learning.\n"
         "Bias-variance tradeoff, overfitting vs underfitting, No Free Lunch theorem.\n"
         "Regularisation: L1, L2, elastic net, dropout, data augmentation."),
        ("Core Algorithms",
         "Linear regression (OLS), logistic regression (sigmoid, cross-entropy loss).\n"
         "SVM: hard/soft margin, kernel trick (RBF, polynomial). k-NN: distance metrics.\n"
         "Decision trees (Gini, entropy), Random Forest, GBM, XGBoost, LightGBM, CatBoost."),
        ("Deep Learning",
         "Feedforward networks: layers, activations, weight initialisation (He, Xavier).\n"
         "CNNs: convolution, pooling. RNNs: BPTT, vanishing gradient. LSTM/GRU.\n"
         "Backpropagation, Adam optimiser, learning rate schedulers."),
        ("NLP & Transformers",
         "Tokenisation: word, subword (BPE, WordPiece), character-level.\n"
         "Embeddings: Word2Vec, GloVe, fastText, contextual (ELMo, BERT).\n"
         "Transformer architecture: attention, BERT fine-tuning, GPT autoregression."),
        ("Model Evaluation",
         "Classification: accuracy, precision, recall, F1, macro/micro/weighted average.\n"
         "Regression: MAE, MSE, RMSE, R², adjusted R².\n"
         "AUC-ROC, PR curves, confusion matrix, cross-validation strategies."),
        ("Deployment & MLOps",
         "Model serialisation: pickle, joblib, ONNX, SavedModel.\n"
         "Serving: REST API (FastAPI), gRPC, TensorFlow Serving, TorchServe.\n"
         "MLOps: versioning (MLflow, DVC), CI/CD pipelines, monitoring data drift."),
    ],
}

# fallback for unknown subject codes
_FALLBACK_CHAPTERS = SUBJECT_CONTENT["DS"]


# ── PDF generation ────────────────────────────────────────────────────────────

def _build_txt(dest_path: pathlib.Path, title: str, chapters: list) -> None:
    """Write a plain-text quick-reference notes file."""
    lines = [
        "=" * 70,
        f"  {title}",
        "  FlipLearn | Quick Reference Notes | 2025-26",
        "=" * 70,
        "",
    ]
    for cht, chb in chapters:
        lines += [
            "",
            "─" * 60,
            f"  {cht}",
            "─" * 60,
        ]
        for para in chb.split("\n"):
            for wrapped in textwrap.wrap(para, 68) or [""]:
                lines.append(f"  {wrapped}")
    lines += ["", "=" * 70, "  End of Notes", "=" * 70, ""]
    dest_path.write_text("\n".join(lines), encoding="utf-8")


def _build_pdf(dest_path: pathlib.Path, title: str, subject_name: str,
               chapters: list) -> None:
    """Create a styled multi-page A4 PDF using PyMuPDF."""
    import fitz  # PyMuPDF

    C_HEADER  = (0.09, 0.47, 0.25)
    C_CHAPTER = (0.10, 0.30, 0.60)
    C_TEXT    = (0.15, 0.15, 0.15)
    C_LIGHT   = (0.94, 0.97, 0.94)
    C_WHITE   = (1.0,  1.0,  1.0)
    W, H = 595, 842  # A4

    doc = fitz.open()

    def new_page():
        pg = doc.new_page(width=W, height=H)
        pg.draw_rect(fitz.Rect(0, 0, W, H), color=C_WHITE, fill=C_WHITE)
        return pg

    def wrap(text: str, width: int = 85) -> list:
        lines = []
        for para in text.split("\n"):
            lines.extend(textwrap.wrap(para, width) or [""])
        return lines

    # ── cover page ──
    pg = new_page()
    pg.draw_rect(fitz.Rect(0, 0, W, 140), color=C_HEADER, fill=C_HEADER)
    pg.insert_text((40, 58),  "FlipLearn",  fontsize=22, color=C_WHITE,  fontname="helv")
    pg.insert_text((40, 88),  "Flipped Classroom Platform", fontsize=12, color=C_WHITE, fontname="helv")
    pg.draw_rect(fitz.Rect(0, 140, W, 235), color=C_LIGHT, fill=C_LIGHT)
    pg.insert_text((40, 182), subject_name, fontsize=13, color=C_CHAPTER, fontname="helv")
    pg.insert_text((40, 215), title,        fontsize=17, color=C_CHAPTER, fontname="helv")
    pg.insert_text((40, 260), "Lecture Notes  |  Academic Year 2025-26",      fontsize=11, color=C_TEXT, fontname="helv")
    pg.insert_text((40, 285), "Department of Computer Science & Engineering", fontsize=11, color=C_TEXT, fontname="helv")
    # TOC
    pg.insert_text((40, 345), "Table of Contents", fontsize=13, color=C_CHAPTER, fontname="helv")
    pg.draw_line(fitz.Point(40, 358), fitz.Point(W - 40, 358), color=C_CHAPTER, width=1)
    y = 372
    for i, (cht, _) in enumerate(chapters, 1):
        pg.insert_text((52, y), f"{i}.   {cht}", fontsize=11, color=C_TEXT, fontname="helv")
        y += 22
    # footer
    pg.draw_rect(fitz.Rect(0, H - 30, W, H), color=C_HEADER, fill=C_HEADER)
    pg.insert_text((40, H - 12), "FlipLearn – Study Material  |  confidential",
                   fontsize=9, color=C_WHITE, fontname="helv")

    # ── chapter pages ──
    for cht, chb in chapters:
        pg = new_page()
        pg.draw_rect(fitz.Rect(0, 0, W, 72), color=C_CHAPTER, fill=C_CHAPTER)
        pg.insert_text((40, 44), cht, fontsize=16, color=C_WHITE, fontname="helv")
        y = 105
        for line in wrap(chb):
            if y > H - 52:
                pg = new_page()
                pg.draw_rect(fitz.Rect(0, 0, W, 42), color=C_CHAPTER, fill=C_CHAPTER)
                pg.insert_text((40, 28), f"(continued)  {cht}", fontsize=12,
                               color=C_WHITE, fontname="helv")
                y = 62
            pg.insert_text((52, y), line, fontsize=11, color=C_TEXT, fontname="helv")
            y += 19
        pg.draw_rect(fitz.Rect(0, H - 30, W, H), color=C_HEADER, fill=C_HEADER)
        pg.insert_text((40, H - 12), f"FlipLearn – {title}",
                       fontsize=9, color=C_WHITE, fontname="helv")

    doc.save(str(dest_path))
    doc.close()


# ── management command ────────────────────────────────────────────────────────

class Command(BaseCommand):
    help = ("Generate dummy PDFs for every StudyMaterial whose file is missing "
            "on disk. Use --all to regenerate every entry.")

    def add_arguments(self, parser):
        parser.add_argument(
            "--all",
            action="store_true",
            help="Regenerate PDFs for ALL StudyMaterial entries, not just missing ones.",
        )

    def handle(self, *args, **options):
        from flipped_app.models import StudyMaterial

        media = pathlib.Path(settings.MEDIA_ROOT)

        all_materials = StudyMaterial.objects.select_related("subject").all()

        if options["all"]:
            targets = list(all_materials)
            self.stdout.write(f"Regenerating all {len(targets)} PDFs…")
        else:
            targets = [
                m
                for m in all_materials
                if (not m.file)
                or (not m.file.name)
                or (not _storage_exists(m))
            ]
            self.stdout.write(
                f"Found {len(targets)} material(s) missing in storage – generating now…"
            )

        if not targets:
            self.stdout.write(self.style.SUCCESS("Nothing to do – all files present."))
            return

        ok = 0
        skipped = 0
        for mat in targets:
            if not mat.file or not mat.file.name:
                self.stdout.write(f"  [skip] {mat.title}  (no file name set)")
                skipped += 1
                continue

            normalized_name = _normalized_material_name(mat.file.name)
            if not normalized_name:
                self.stdout.write(f"  [skip] {mat.title}  (invalid file name)")
                skipped += 1
                continue

            dest = media / normalized_name
            dest.parent.mkdir(parents=True, exist_ok=True)

            chapters = SUBJECT_CONTENT.get(mat.subject.code, _FALLBACK_CHAPTERS)
            fname_lower = normalized_name.lower()
            try:
                if fname_lower.endswith(".pdf"):
                    _build_pdf(dest, mat.title, mat.subject.name, chapters)
                elif fname_lower.endswith(".txt"):
                    _build_txt(dest, mat.title, chapters)
                else:
                    self.stdout.write(f"  [skip] {normalized_name}  (unsupported type)")
                    skipped += 1
                    continue
                with dest.open("rb") as fh:
                    mat.file.save(normalized_name, File(fh), save=True)
                self.stdout.write(self.style.SUCCESS(f"  [ok]   {normalized_name}"))
                ok += 1
            except Exception as exc:
                self.stdout.write(
                    self.style.ERROR(f"  [err]  {normalized_name} – {exc}")
                )

        self.stdout.write("")
        self.stdout.write(
            self.style.SUCCESS(
                f"Done.  Generated: {ok}  |  Skipped: {skipped}  |  "
                f"Total targets: {len(targets)}"
            )
        )
