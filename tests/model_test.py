import joblib
import pandas as pd


column_transformer = joblib.load("./models/column_transformer.pkl")

model_mbti = joblib.load("./models/logreg_mbti_group.pkl")
le_mbti = joblib.load("./models/labelencoder_mbti_group.pkl")

model_belbin = joblib.load("./models/logreg_belbinrole.pkl")
le_belbin = joblib.load("./models/labelencoder_belbinrole.pkl")


sample = pd.DataFrame({
    'Position': ["Data Scientist"],
    'SoftSkills': ["leadership, strategic thinking, Problem-Solving, Communication, logic, resilience, Analytical thinking, decision making"],
    'Summary': ["Analytical and results-driven person with a strong problem-solving mindset. Committed to delivering high-quality insights and solutions under tight deadlines while meeting complex project goals with precision."],
    'Projects': ["1. HappyLens-NN - Python, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost, TensorFlow, Keras, LSTM, JSON, GenAI. "
                 "An extension of HappyLens with forecasting and scenario-based modeling. Cleaned and analyzed a 1,956-row dataset, "
                 "engineered features, trained models, evaluated MSE, simulated what-if scenarios. "
                 "2. Personal assistant App - Django, PostgreSQL. Designed notebook module with advanced note/tag/search features."]
})


sample_transformed = column_transformer.transform(sample)

pred_mbti = model_mbti.predict(sample_transformed)
decoded_mbti = le_mbti.inverse_transform(pred_mbti)

pred_belbin = model_belbin.predict(sample_transformed)
decoded_belbin = le_belbin.inverse_transform(pred_belbin)

print(f"Pred MBTI_Group: {decoded_mbti[0]}")
print(f"Pred BelbinRole: {decoded_belbin[0]}")
