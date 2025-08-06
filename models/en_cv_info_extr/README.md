---
widget:
- text: |2-
     John Doe
     123 Main Street, Cityville, CA 12345
     johndoe@email.com
     (555) 123-4567
     linkedin.com/in/johndoe
     Professional Summary
     Experienced and results-driven Data Scientist with a strong background in statistical analysis, machine learning, and data visualization. Proven track record of delivering actionable insights and driving data-driven decision-making processes. Adept at leveraging advanced analytics to solve complex business problems.

    Education
     Master of Science in Data Science
     ABC University, Cityville, CA
     May 2021
     
     Bachelor of Science in Computer Science
     XYZ University, Townsville, CA
     Graduation Date: May 2018
     
     Professional Experience
     Data Scientist | Tech Innovators Inc., Cityville, CA | June 2021 - Present
     Lead data analysis projects, extracting valuable insights to inform business strategies.
     Develop and deploy machine learning models to optimize key processes, resulting in a 15% increase in efficiency.
     Collaborate with cross-functional teams to design and implement data-driven solutions.
     Utilize Python, R, and SQL for data extraction, transformation, and analysis.
     Create compelling data visualizations to communicate findings to non-technical stakeholders.

    Data Analyst | Data Solutions Co., Townsville, CA | January 2019 - May 2021
     Conducted exploratory data analysis to identify trends, patterns, and anomalies.
     Implemented data cleaning and preprocessing techniques to ensure data quality.
     Produced comprehensive reports and dashboards, aiding in executive decision-making.
     Collaborated with business units to define and refine analytical requirements.

    Skills
     Programming Languages: Python, R
     Data Analysis Tools: Pandas, NumPy
     Machine Learning: Scikit-Learn, TensorFlow, Keras
     Database Management: SQL
     Data Visualization: Matplotlib, Seaborn
     Statistical Analysis: Hypothesis testing, Regression analysis
     Communication: Strong written and verbal communication skills

    Certifications
     Certified Data Scientist (CDS)
     Machine Learning Specialist Certification
     
tags:
- spacy
- token-classification
- cv
- resume parsing
- resume extraction
- named entity recognition
- resume
language:
- en
model-index:
- name: en_cv_info_extr
  results:
  - task:
      name: NER
      type: token-classification
    metrics:
    - name: NER Precision
      type: precision
      value: 0.8333333333
    - name: NER Recall
      type: recall
      value: 0.8067729084
    - name: NER F Score
      type: f_score
      value: 0.8198380567
library_name: spacy
pipeline_tag: token-classification
---

# Information extraction from Resumes/CVs written in English

### Model Description
This model is designed for information extraction from resumes/CVs written in English. It employs a transformer-based architecture with spaCy for named entity recognition (NER) tasks. The model aims to parse various sections of resumes, including personal details, education history, professional experience, skills, and certifications, enabling users to extract structured information for further processing or analysis.

### Model Details
| Feature | Description |
| --- | --- |
| `Language` | English |
| `Task` | Named Entity Recognition (NER) |
| `Objective` | Information extraction from resumes/CVs |
| `Spacy Components` | Transformer, Named Entity Recognition (NER) |
| `Author` | [Youssef Chafiqui](https://huggingface.co/ychafiqui) |

### NER Entities
The model recognizes various entities corresponding to different sections of a resume. Below are the entities used by the model:
| Label | Description |
| --- | --- |
| 'FNAME' | First name |
| 'LNAME' | Last name |
| 'ADDRESS' | Address |
| 'CERTIFICATION' | Certification |
| 'EDUCATION' | Education section |
| 'EMAIL' | Email address |
| 'EXPERIENCE' | Experience section |
| 'HOBBY' | Hobby |
| 'HSKILL' | Hard skill |
| 'LANGUAGE' | Language |
| 'PHONE' | Phone number |
| 'PROFILE' | Profile |
| 'PROJECT' | Project section |
| 'SSKILL' | Soft skill |

### Evaluation Metrics

| Type | Score |
| --- | --- |
| `F1 score` | 81.98 |
| `Precision` | 83.33 |
| `Recall` | 80.68 |

## Usage
### Presequities
Install spaCy library
```bash
pip install spacy
```

Install Transformers library
```bash
pip install transformers
```

Download the model
```bash
pip install https://huggingface.co/ychafiqui/en_cv_info_extr/resolve/main/en_cv_info_extr-1.0.0-py3-none-any.whl
```

### Load the model
```python
import spacy
nlp = spacy.load("en_cv_info_extr")
```

### Inference using the model
```python
doc = nlp('put your resume here')

for ent in doc.ents:
  print(ent.text, "-", ent.label_)
```