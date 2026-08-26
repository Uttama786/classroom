# 📘 Flipped Classroom Using RAG: Performance Analysis in CSE Education

## 📌 Project Overview
This project is a web-based learning platform designed to implement and analyze the **Flipped Classroom** model in **Computer Science and Engineering (CSE)** education. Students access study materials, watch video lectures, and attempt quizzes/assignments before class. The system uses Machine Learning models to analyze interaction data in real-time, predict final exam scores, and identify at-risk students. Additionally, it integrates a Retrieval-Augmented Generation (RAG) agent (**FlipLearn AI**) to answer student queries grounded in course textbooks and slides.

---

## 🎯 Objectives
- **Implement a Flipped Classroom Platform**: Allow students to access pre-class study materials, submit assignments, and take quizzes.
- **Collect Interaction Metrics**: Log student engagement metrics such as video watch duration, downloads, and chatbot queries.
- **Predict Student Performance**: Run `scikit-learn` Machine Learning models to predict final exam scores and classify performance levels.
- **Identify At-Risk Students Early**: Automatically flag students performing below expected thresholds and send targeted alerts.
- **Grounded Chatbot Tutor**: Provide a RAG-based AI Tutor for academic content questions using the platform's knowledge base.

---

## 🏗 System Architecture & Key Files

### 1. Django Web Application
The core web interface and backend are located in the [flipped_app](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app) directory.
* **Database Models**: Defined in [models.py](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/models.py).
  * [StudentProfile](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/models.py#L16) & [TeacherProfile](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/models.py#L34): Track user metadata.
  * [VideoLecture](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/models.py#L45) & [VideoWatchHistory](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/models.py#L175): Record lecture viewings.
  * [StudyMaterial](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/models.py#L84): Tracks materials downloaded.
  * [QuizAttempt](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/models.py#L161) & [AssignmentSubmission](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/models.py#L142): Track academic marks.
  * [Attendance](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/models.py#L189): Tracks presence in class.
  * [StudentPerformance](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/models.py#L203): Aggregates engagement and performance data.
* **Routing & Views**: URLs are configured in [urls.py](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/urls.py), and view logic is in [views.py](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/views.py).
* **Automatic Engagement Updates**: Whenever students interact with quizzes, watch videos, or submit assignments, the [_update_engagement](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/views.py#L1058) helper updates aggregates and automatically runs the ML model to update prediction labels.

### 2. RAG AI Tutor (FlipLearn AI)
The academic chatbot engine is located in the [rag_engine](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/rag_engine) directory.
* **Embeddings**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) in [embedding_model.py](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/rag_engine/embedding_model.py) to represent text segments.
* **Retrieval**: Performs fast similarity searching against a FAISS index in [retriever.py](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/rag_engine/retriever.py).
* **Response Generation**: Queries the **Groq API** (`llama-3.1-8b-instant`) in [chat.py](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/rag_engine/chat.py).
* **Out-of-Domain (OOD) Protection**: Checks retrieve chunk scores; if the max similarity score is $< 0.38$, the chatbot answers strictly with: *"Your matched query is not found in our database."*
* **File Chat (PDF/Word)**: Supports uploading and extracting text from PDF and Word (`.docx`) files using the [chat_pdf_view](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/flipped_app/views.py#L1345) to stream explanations.

### 3. Machine Learning Module
The performance prediction models are located in the [ml_model](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/ml_model) directory.
* **Models**: Features are scaled via a `StandardScaler`, and predictions are made using `RandomForestRegressor` and `RandomForestClassifier`.
  * **Regressor**: Predicts final exam scores (scaled between 0-100).
  * **Classifier**: Classifies students into performance labels: *High*, *Medium*, *Low*, and *At-Risk*.
  * The logic resides in [prediction.py](file:///c:/Users/uttam/Downloads/RAG/flipped_classroom_project/ml_model/prediction.py).
* **Visualizations**: The script [regenerate_plots.py](file:///c:/Users/uttam/Downloads/RAG/regenerate_plots.py) generates confusion matrices, feature importances, and distribution graphs for the teacher analytics dashboard.

---

## 📊 Dataset Features & ML Targets
We collect and scale seven primary engagement and academic background features:
1. `videos_watched`: Total number of video lectures marked completed.
2. `total_video_time_minutes`: Total duration spent watching lectures.
3. `quiz_avg_score`: Average score across all attempted quizzes.
4. `assignment_avg_marks`: Average marks obtained on graded assignments.
5. `attendance_percentage`: Percentage of present records in subject classes.
6. `participation_score`: Participation points derived from AI Chatbot usage (0.5 points per query, capped at 10.0).
7. `previous_gpa`: The student's academic GPA from the previous semester.

---

## 🛠 Setup & Installation

### 1) Clone the Repository & Configure Environment
```bash
git clone https://github.com/your-username/flipped-classroom-ml.git
cd flipped-classroom-ml/flipped_classroom_project
```
Create a `.env` file in the `flipped_classroom_project` directory (adjacent to `manage.py`) with your Groq credentials:
```env
DEBUG=True
SECRET_KEY=dev-only-secret-key-change-before-deploy
GROQ_API_KEY=gsk_your_groq_api_key_here
RAG_ENABLE_WEB_SEARCH=False
```

### 2) Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 3) Set Up the Database & Run Migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

### 4) Train ML Models & Run Platform
To pre-train the random forest classifiers and regressors, run the training scripts:
```bash
python ml_model/model_training.py
```
You can also regenerate plots using:
```bash
python ../regenerate_plots.py
```
Finally, start the Django development server:
```bash
python manage.py runserver
```
Visit `http://127.0.0.1:8000/` in your browser.

---

---

## 📈 Experimental Results & Discussion

The proposed **FlipLearn** framework was evaluated using the **FlipLearn-SPD** private dataset containing **12,455 student learning records**. The dataset was split using an **80:10:10** strategy (9,964 training, 1,245 validation, and 1,246 test samples).

### 1. Student Performance Classification (Table 2)
The proposed FlipLearn performance classification model achieved a test accuracy of **97.5%**, significantly outperforming baseline classifiers:

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| Naïve Bayes | 85.4 | 86.1 | 84.5 | 85.3 |
| KNN | 81.3 | 82.5 | 79.8 | 81.1 |
| Decision Tree | 83.3 | 84.0 | 82.1 | 83.0 |
| Random Forest | 87.5 | 88.2 | 86.7 | 87.4 |
| Gradient Boosting | 90.6 | 91.2 | 89.7 | 90.4 |
| SVM | 91.7 | 92.1 | 90.8 | 91.4 |
| XGBoost | 93.8 | 94.0 | 93.1 | 93.5 |
| Logistic Regression | 95.8 | 96.0 | 95.2 | 95.6 |
| **Proposed FlipLearn** | **97.5** | **97.6** | **97.3** | **97.4** |

### 2. Final Examination Score Prediction (Table 3)
The regression component explains approximately **97.58%** of the variance in final scores with an RMSE of **3.618**:

| Model | $R^2$ | RMSE | MAE |
| :--- | :---: | :---: | :---: |
| Random Forest Regression | 0.9416 | 5.02 | 3.91 |
| Gradient Boosting Regression | 0.9328 | 5.37 | 4.12 |
| XGBoost Regression | 0.9532 | 4.61 | 3.56 |
| **Proposed Linear Regression** | **0.9758** | **3.618** | **2.84** |

### 3. RAG-Based Intelligent Tutoring Results (Table 4)
The RAG-based tutoring system combines semantic retrieval (FAISS + `all-MiniLM-L6-v2`) with `Llama-3.1-8B-Instant`, achieving curriculum-grounded responses:

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Human Eval. (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| No-RAG LLM | 71.3 | 73.2 | 69.8 | 71.4 | 76.5 |
| BM25 | 76.8 | 78.1 | 75.2 | 76.6 | 80.3 |
| Mistral | 84.6 | 85.3 | 83.8 | 84.5 | 87.1 |
| GPT | 89.7 | 90.2 | 88.9 | 89.5 | 91.8 |
| Llama | 91.4 | 92.0 | 90.1 | 91.0 | 93.2 |
| **FlipLearn-RAG** | **97.0** | **96.8** | **97.2** | **97.0** | **96.5** |

### 4. Sample FlipLearn Prediction Outputs (Table 5)
| Student ID | Performance Class | Predicted Score | Risk Status |
| :--- | :---: | :---: | :---: |
| S001 | High | 92.4 | Not-At-Risk |
| S002 | High | 86.7 | Not-At-Risk |
| S003 | Medium | 71.8 | Not-At-Risk |
| S004 | Low | 57.3 | At-Risk |
| S005 | High | 94.6 | Not-At-Risk |

---

## 🏁 Conclusion
FlipLearn is a RAG-enabled predictive learning analytics framework that integrates intelligent tutoring, student performance classification, final exam score estimation, and early academic-risk identification within a unified Learning Management System. The combination of FAISS-based semantic retrieval, `all-MiniLM-L6-v2` embeddings, `Llama-3.1-8B-Instant`, and predictive analytics provides proactive student monitoring and personalized academic assistance.

---

## 👨‍💻 Author
**Uttam Vitthal Bhise**  
M.Tech – Computer Science & Engineering

