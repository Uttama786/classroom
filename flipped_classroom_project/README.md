# 📘 Performance Analysis of the Flipped Classroom in CSE Education Using Machine Learning

## 📌 Project Overview
This project is a web-based learning platform designed to implement and analyze the **Flipped Classroom** model in **Computer Science and Engineering (CSE)** education. Students learn from videos and notes before class, then practice with quizzes, discussions, and coding tasks during class. The system collects interaction data and uses Machine Learning models to **predict performance and identify at-risk students**.

---

## 🎯 Objectives
- Implement a flipped classroom learning system.
- Collect and analyze engagement and performance data.
- Apply ML algorithms to:
  - Predict student performance.
  - Classify students (High / Medium / Low performers).
  - Identify at-risk students.
- Compare flipped classroom vs traditional teaching outcomes.

---

## 🏗 System Architecture
**Workflow**
1. Student registers and logs in.
2. Watches lecture videos.
3. Downloads study materials.
4. Attempts quizzes and submits assignments.
5. System stores engagement + performance data.
6. ML model analyzes data and predicts results.
7. Admin dashboard shows analytics.

---

## 🛠 Technology Stack
**Web Application**
- Frontend: HTML, CSS, Bootstrap
- Backend: Django (Python)
- Database: SQLite / MySQL

**Machine Learning**
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn

---

## 👥 User Modules
**Admin / Teacher Module**
- Upload video lectures / PDF notes
- Create quizzes and assignments
- Monitor progress
- View analytics and reports

**Student Module**
- Register / Login
- Access materials
- Watch videos, attempt quizzes
- Submit assignments
- View performance dashboard

---

## 📊 Dataset Features
- Number of videos watched
- Total time spent on videos
- Quiz scores
- Assignment marks
- Attendance
- Participation score
- Previous academic performance
- Final exam score

---

## 🤖 Machine Learning Models Used
**Regression**
- Linear Regression
- Random Forest Regressor

**Classification**
- Logistic Regression
- Decision Tree
- Random Forest Classifier

---

## 📈 Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Mean Squared Error (Regression)
- Feature Importance

---

## 🚀 Installation & Setup
```bash
# 1) Clone the repo

git clone https://github.com/your-username/flipped-classroom-ml.git
cd flipped-classroom-ml

# 2) Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run migrations
python manage.py makemigrations
python manage.py migrate

# 5) Run server
python manage.py runserver
```

---

## 📌 Future Enhancements
- Deep Learning-based performance prediction
- Personalized learning recommendation system
- Real-time analytics dashboard
- Cloud deployment (AWS / Azure)
- Mobile app support

---

## 👨‍💻 Author
**Uttam Vitthal Bhise**  
M.Tech – Computer Science & Engineering
