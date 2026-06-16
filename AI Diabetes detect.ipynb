{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "8a4d0b43-8235-4df3-83ea-01fd398b181c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \\\n",
      "0            6      148             72             35        0  33.6   \n",
      "1            1       85             66             29        0  26.6   \n",
      "2            8      183             64              0        0  23.3   \n",
      "3            1       89             66             23       94  28.1   \n",
      "4            0      137             40             35      168  43.1   \n",
      "\n",
      "   DiabetesPedigreeFunction  Age  Outcome  \n",
      "0                     0.627   50        1  \n",
      "1                     0.351   31        0  \n",
      "2                     0.672   32        1  \n",
      "3                     0.167   21        0  \n",
      "4                     2.288   33        1  \n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "import os\n",
    "import mlflow\n",
    "import mlflow.sklearn\n",
    "from sklearn.datasets import make_classification\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import classification_report\n",
    "\n",
    "from imblearn.over_sampling import SMOTE\n",
    "from sklearn.model_selection import cross_val_score\n",
    "import mlflow\n",
    "import mlflow.sklearn\n",
    "\n",
    "\n",
    "# Load the medical dataset directly into your script\n",
    "url = \"https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv\"\n",
    "df = pd.read_csv(url)\n",
    "\n",
    "# Let's print out the first 5 rows to see what columns you have\n",
    "print(df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "92d3720f-0d0a-4a0f-87c0-bda89c290fbc",
   "metadata": {},
   "outputs": [],
   "source": [
    "x = df.drop(\"Outcome\",axis = 1)\n",
    "y = df[\"Outcome\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "6e73fcdf-e3a1-4d0b-a105-702dcc34ccd6",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train,x_test, y_train, y_test = train_test_split(x,y,test_size = 42,random_state = 42, stratify = y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "730753bb-c378-4baa-b5b8-b48ab6366291",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "dc112b4c-83ba-4d48-a12c-74ff44a5127d",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "scaler = StandardScaler()\n",
    "x_train = scaler.fit_transform(x_train)\n",
    "x_test = scaler.transform(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "0d3b89c0-d6a4-4de8-86e5-cce60c868aee",
   "metadata": {},
   "outputs": [],
   "source": [
    "from imblearn.pipeline import Pipeline\n",
    "desktop_mlflow_path = r\"file:///C:/Users/DELL 3070/Desktop/mlruns\"\n",
    "mlflow.set_tracking_uri(desktop_mlflow_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "f570c363-9e00-40a0-9c76-4bea7c39dc8c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.7254902  0.78       0.72       0.78431373 0.78431373]\n",
      "our cross validation is 0.7588235294117647\n"
     ]
    }
   ],
   "source": [
    "dt_Pipeline = Pipeline([\n",
    "    ('smote',SMOTE(random_state = 42)),\n",
    "    ('model',  DecisionTreeClassifier(max_depth = 5, random_state = 42) )\n",
    "])\n",
    "\n",
    "dt_score = cross_val_score(dt_Pipeline, x_train,y_train, scoring = \"recall\", n_jobs = -1)\n",
    "print(dt_score)\n",
    "print(\"our cross validation is\",dt_score.mean())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "06c7b622-5ca7-44be-81eb-044600e40cb0",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026/06/14 17:46:22 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.\n",
      "2026/06/14 17:46:22 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "final test recall is 0.8\n",
      "final test precision is 0.6666666666666666\n"
     ]
    }
   ],
   "source": [
    "\n",
    "with mlflow.start_run(run_name = \"Decision_tree_base\"):\n",
    "    dt_base = dt_Pipeline.fit(x_train,y_train)\n",
    "    dt_base_pred =  dt_Pipeline.predict(x_test)\n",
    "    \n",
    "    dt_report = classification_report(y_test, dt_base_pred, output_dict = True)\n",
    "    \n",
    "    mlflow.log_param(\"max_depth\" , 5)\n",
    "    mlflow.log_metric(\"test_recall\" , dt_report[\"1\"][\"recall\"])\n",
    "     \n",
    "    \n",
    "    \n",
    "    mlflow.sklearn.log_model(dt_base,\"model\")\n",
    "    \n",
    "    \n",
    "    print(\"final test recall is\",dt_report[\"1\"][\"recall\"])\n",
    "    print(\"final test precision is\",dt_report[\"1\"][\"precision\"])\n",
    "    \n",
    "    \n",
    "    import joblib\n",
    "    \n",
    "    joblib.dump(dt_Pipeline , r\"C:\\Users\\DELL 3070\\Desktop\\diabetes_dt_model.pkl\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "fbfbb49b-f965-456f-bf68-9aee4e6d97ac",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.70588235 0.72       0.66       0.70588235 0.82352941]\n",
      "our cross validation is 0.7230588235294118\n"
     ]
    }
   ],
   "source": [
    "rf_Pipeline = Pipeline([\n",
    "    ('smote',SMOTE(random_state = 42)),\n",
    "    ('model',  RandomForestClassifier(n_estimators = 150 , max_depth = 10, random_state = 42) )\n",
    "])\n",
    "\n",
    "rf_score = cross_val_score(rf_Pipeline, x_train,y_train, scoring = \"recall\", n_jobs = -1)\n",
    "print(rf_score)\n",
    "print(\"our cross validation is\",rf_score.mean())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "364936ec-9355-4bb2-9ac2-7743d9b429cc",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026/06/14 18:04:16 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.\n",
      "2026/06/14 18:04:16 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "final test recall is 0.8666666666666667\n",
      "final test precision is 0.65\n"
     ]
    }
   ],
   "source": [
    "\n",
    "with mlflow.start_run(run_name = \"random_forest_base\"):\n",
    "    rf_base = rf_Pipeline.fit(x_train,y_train)\n",
    "    rf_base_pred =  rf_Pipeline.predict(x_test)\n",
    "    \n",
    "    rf_report = classification_report(y_test, rf_base_pred, output_dict = True)\n",
    "    \n",
    "    mlflow.log_param(\"max_depth\" ,12)\n",
    "     \n",
    "    mlflow.log_param(\"n_estimators\" , 150)\n",
    "    mlflow.log_metric(\"test_recall\" , rf_report[\"1\"][\"recall\"])\n",
    "     \n",
    "    \n",
    "    \n",
    "    mlflow.sklearn.log_model(rf_base,\"model\")\n",
    "    \n",
    "    \n",
    "    print(\"final test recall is\",rf_report[\"1\"][\"recall\"])\n",
    "    print(\"final test precision is\",rf_report[\"1\"][\"precision\"])\n",
    "    \n",
    "    \n",
    "    import joblib\n",
    "    \n",
    "    joblib.dump(rf_Pipeline , r\"C:\\Users\\DELL 3070\\Desktop\\diabetes_dt_model.pkl\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "1f6eacc0-64d8-44f3-b6e9-a7f2bfc59753",
   "metadata": {},
   "outputs": [],
   "source": [
    "from xgboost import XGBClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "2bc38e44-f3d4-4851-9342-168c31fb3005",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.78431373 0.66       0.66       0.74509804 0.66666667]\n",
      "our cross validation is 0.7032156862745098\n"
     ]
    }
   ],
   "source": [
    "xgb_Pipeline = Pipeline([\n",
    "    ('smote',SMOTE(random_state = 42)),\n",
    "    ('model', XGBClassifier(max_depth = 5 , learning_rate = 0.1,random_state = 42))\n",
    "])\n",
    "\n",
    "xgb_score = cross_val_score(xgb_Pipeline, x_train,y_train, scoring = \"recall\", n_jobs = -1)\n",
    "print(xgb_score)\n",
    "print(\"our cross validation is\",xgb_score.mean())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "f522e9cc-5edc-4871-9fa2-4f677e297e5a",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026/06/14 18:16:33 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.\n",
      "2026/06/14 18:16:33 WARNING mlflow.sklearn: Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution because these formats rely on Python's object serialization mechanism, which can execute arbitrary code during deserialization. The recommended safe alternative is the 'skops' format. For more information, see: https://scikit-learn.org/stable/model_persistence.html\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "final test recall is 0.8\n",
      "final test precision is 0.6\n"
     ]
    }
   ],
   "source": [
    "\n",
    "with mlflow.start_run(run_name = \"XGB\"):\n",
    "    xgb_base = xgb_Pipeline.fit(x_train,y_train)\n",
    "    xgb_base_pred =  xgb_Pipeline.predict(x_test)\n",
    "    \n",
    "    xgb_report = classification_report(y_test, xgb_base_pred, output_dict = True)\n",
    "    \n",
    "    mlflow.log_param(\"max_depth\" ,5)\n",
    "     \n",
    "    mlflow.log_param(\"learning_rate\", 0.1)\n",
    "    mlflow.log_metric(\"test_recall\" , xgb_report[\"1\"][\"recall\"])\n",
    "     \n",
    "    \n",
    "    \n",
    "    mlflow.sklearn.log_model(xgb_base,\"model\")\n",
    "    \n",
    "    \n",
    "    print(\"final test recall is\",xgb_report[\"1\"][\"recall\"])\n",
    "    print(\"final test precision is\",xgb_report[\"1\"][\"precision\"])\n",
    "    \n",
    "    \n",
    "    import joblib\n",
    "    \n",
    "    joblib.dump(xgb_Pipeline , r\"C:\\Users\\DELL 3070\\Desktop\\diabetes_dt_model.pkl\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "49fb09db-a0c4-467f-b3e9-1dfba06009ca",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['AI_diabetes_model.pkl']"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "joblib.dump(rf_base, \"AI_diabetes_model.pkl\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "84081498-1466-4aef-ba74-a4cf64d6f5b8",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = joblib.load(\"AI_diabetes_model.pkl\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "857ca863-2c02-4b41-9710-82dcc53d32e1",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
