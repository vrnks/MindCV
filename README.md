# **Mind CV**

A machine learning project to predict a candidate's **personality type (MBTI Group)** and **Belbin Team Role** based on resume data.
Currently supports **manual input** and **PDF resumes** (DOCX and image OCR coming soon).

---

## **📌 Project Overview**

The project extracts **key resume sections** and applies **ML** to classify personality traits into:

* **MBTI Group**: Analysts, Diplomats, Explorers, Sentinels
* **Belbin Role**: Team roles such as Implementer, Coordinator, Plant, etc.

---

## **✅ Key Features**

✔ Accepts resume input via:

* **PDF file upload**
* **Manual entry** (Position, SoftSkills, Summary, Projects)

✔ **Text Preprocessing**

* Clean text (remove symbols, extra spaces)
* Extract key sections:

  * Position
  * Soft Skills
  * Summary
  * Projects

✔ **Models Evaluated**

* Logistic Regression ✅ *(selected as best)*
* Random Forest
* XGBoost

---

### 📂 Dataset Collection and Structure

The model was trained on a **custom dataset** combining information about job positions, soft skills, resume summaries, and project descriptions. Each record is labeled with **Belbin Team Role**, **MBTI type**, and the corresponding **MBTI Group**.

**CSV Structure:**

| Column       | Description                                                            |
| ------------ | ---------------------------------------------------------------------- |
| `Position`   | Job title or role mentioned in the CV                                  |
| `SoftSkills` | List of soft skills extracted from the CV                              |
| `Summary`    | Candidate summary or profile description                               |
| `Projects`   | Major projects or key achievements                                     |
| `BelbinRole` | One of the 9 Belbin team roles (e.g., Plant, Coordinator, Implementer) |
| `MBTI`       | The full 4-letter MBTI type (e.g., INTJ, ENFP)                         |
| `MBTI_Group` | One of the 4 MBTI groups (Analysts, Diplomats, Sentinels, Explorers)   |

Shape: (788, 7)
The project uses a custom-built dataset of 788 CV samples with 108 unique positions from the IT sector.

⚠ Currently, all samples come from the IT industry. Future versions will expand to other domains for better generalization.

**Example row:**

```
Data Scientist,"critical thinking, problem solving, logical reasoning, objectivity, analysis",Data scientist with a strong analytical background focused on evidence-based decision making.,1. Risk Modeling Platform — created probabilistic models to assess loan default risks.\n2. A/B Testing Automation Tool — reduced time to insights by 40%.,Monitor Evaluator,INTP,Analysts
```

---

## **📂 Project Structure**

```
MindCV/
├── app.py
├── data/
│   └── data_preprocessing.ipynb
├── models/
│   ├── column_transformer.pkl
│   ├── en_cv_info_extr/      
│   ├── labelencoder_belbinrole.pkl
│   ├── labelencoder_mbti_group.pkl
│   ├── logreg_belbinrole.pkl
│   ├── logreg_mbti_group.pkl
│   ├── pos_tfidf.pkl
│   ├── proj_tfidf.pkl
│   ├── soft_tfidf.pkl
│   └── summ_tfidf.pkl
├── src/
│   ├── data_loader.py
│   ├── inference.py
│   └── ner_extractor.py
└── tests/
│   └── model_test.py
│
├── requirements.txt
└── README.md                           
```

---

## **⚙️ How It Works**

1. **Load Data**: Accepts PDF or manual input.
2. **Extract Sections**: Identify `Position`, `Soft Skills`, `Summary`, `Projects`.
3. **Prediction**: Use trained **Logistic Regression** models:

   * `logreg_mbtigroup.pkl` → MBTI Group
   * `logreg_belbinrole.pkl` → Belbin Role

---

## **🚀 Usage**

### **Streamlit**: https://resume-personality-predictor.streamlit.app

---

## **📊 Model Performance**

### **MBTI Group**

```
Mean Accuracy: 0.8528
Mean F1 Macro: 0.8543
```

Class-level:

```
Analysts:   F1 ~ 0.82
Diplomats:  F1 ~ 0.86
Explorers:  F1 ~ 0.90
Sentinels:  F1 ~ 0.83
```

### **Belbin Role**

```
Accuracy: ~0.75
F1 Macro: ~0.76
```

---

## **💡 Future Improvements**

* Add support for **DOCX** and **image OCR**


