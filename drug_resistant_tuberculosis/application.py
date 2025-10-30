"""Flask application to train/evaluate DR-TB models and save best models.

Routes
- /           GET -> form to upload dataset and choose action
- /run        POST -> run training (single) or model selection and show results

This is a minimal synchronous app intended for local experimentation.
"""
from flask import Flask, request, render_template, redirect, url_for, send_file, flash
import os
from werkzeug.utils import secure_filename
from pathlib import Path
import traceback

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'drtb-secret')

# upload and output folders (inside package folder)
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.route('/')
def index():
    """Show upload form and action choices."""
    return render_template('index.html')


def save_uploaded_file(f):
    filename = secure_filename(f.filename)
    dest = os.path.join(UPLOAD_DIR, filename)
    f.save(dest)
    return dest


@app.route('/run', methods=['POST'])
def run():
    try:
        # form values
        action = request.form.get('action')  # 'single' or 'selection'
        save_best = bool(request.form.get('save_best'))
        save_dir = request.form.get('save_dir') or 'outputs/models'

        # dataset: either uploaded or use repo's default dr_dataset.csv
        uploaded = request.files.get('dataset')
        if uploaded and uploaded.filename:
            csv_path = save_uploaded_file(uploaded)
        else:
            # use repo dataset
            csv_path = os.path.join(BASE_DIR, 'dr_dataset.csv')
            if not os.path.exists(csv_path):
                return render_template('result.html', error='No dataset provided and default dataset not found.')

        # Load and preprocess
        from drtb.data import load_data
        from drtb.preprocess import preprocess_pipeline, split_X_y, apply_smote, train_test_split_stratified

        df = load_data(csv_path)
        df_proc = preprocess_pipeline(df)
        X, y = split_X_y(df_proc)
        X_res, y_res = apply_smote(X, y)
        X_train, X_test, y_train, y_test = train_test_split_stratified(X_res, y_res)

        if action == 'single':
            # train GaussianNB as in original flow
            from drtb.model import train_gaussian_nb, save_model
            model = train_gaussian_nb(X_train, y_train)
            accuracy = float(model.score(X_test, y_test))
            y_pred = model.predict(X_test)

            # optionally save
            saved_path = None
            if save_best:
                save_folder = os.path.join(BASE_DIR, save_dir)
                os.makedirs(save_folder, exist_ok=True)
                fname = os.path.join(save_folder, 'gaussian_nb_model.joblib')
                saved_path = save_model(model, fname)

            return render_template('result.html', single=True, accuracy=accuracy, y_test=y_test.tolist(), y_pred=y_pred.tolist(), saved_path=saved_path)

        else:
            # model selection
            from drtb.model import train_and_select_best, save_model
            # build same candidates as run_train
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.linear_model import LogisticRegression
            from xgboost import XGBClassifier
            from sklearn.naive_bayes import GaussianNB, ComplementNB
            from sklearn.svm import SVC
            try:
                from catboost import CatBoostClassifier
                has_catboost = True
            except Exception:
                has_catboost = False

            models = {
                "RandomForest": RandomForestClassifier(n_jobs=-1),
                "GradientBoosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "DecisionTree": DecisionTreeClassifier(),
                "KNeighbors": KNeighborsClassifier(),
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "GaussianNB": GaussianNB(),
                "ComplementNB": ComplementNB(),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                "SVC": SVC(probability=True)
            }
            if has_catboost:
                models["CatBoost"] = CatBoostClassifier(verbose=False)

            # stacking ensemble
            try:
                estimators = [("rf", RandomForestClassifier(n_estimators=50, random_state=42)), ("xgb", XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50))]
                if has_catboost:
                    estimators.append(("cat", CatBoostClassifier(verbose=False, random_state=42)))
                stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), n_jobs=-1)
                models["StackingEnsemble"] = stacking
            except Exception:
                pass

            # add the two user-requested Voting ensembles
            try:
                from sklearn.ensemble import VotingClassifier
                v1 = VotingClassifier(estimators=[('rf', RandomForestClassifier(n_estimators=100, random_state=42)), ('gb', GradientBoostingClassifier(n_estimators=100)), ('dt', DecisionTreeClassifier())], voting='soft', n_jobs=-1)
                models['Ensemble_RF_GB_DT'] = v1
            except Exception:
                pass
            try:
                from sklearn.ensemble import VotingClassifier
                v2_estimators = []
                try:
                    v2_estimators.append(('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50)))
                except Exception:
                    pass
                if 'stacking' in locals():
                    v2_estimators.append(('stacking', stacking))
                if has_catboost:
                    v2_estimators.append(('cat', CatBoostClassifier(verbose=False, random_state=42)))
                if v2_estimators:
                    v2 = VotingClassifier(estimators=v2_estimators, voting='soft', n_jobs=-1)
                    models['Ensemble_XGB_Stack_Cat'] = v2
            except Exception:
                pass

            params = {
                "RandomForest": {"n_estimators": [50, 100], "max_depth": [None, 10]},
                "GradientBoosting": {"n_estimators": [50], "learning_rate": [0.1]},
                "XGBoost": {"n_estimators": [50], "learning_rate": [0.1]},
                "KNeighbors": {"n_neighbors": [3, 5]}
            }

            best_name, best_model, report = train_and_select_best(X_train, y_train, X_test, y_test, models, params)

            saved_path = None
            if save_best:
                save_folder = os.path.join(BASE_DIR, save_dir)
                os.makedirs(save_folder, exist_ok=True)
                fname = os.path.join(save_folder, f"{best_name}_model.joblib")
                saved_path = save_model(best_model, fname)

            # convert report to list for template
            items = sorted(report.items(), key=lambda x: -x[1])
            return render_template('result.html', single=False, best_name=best_name, report=items, saved_path=saved_path)

    except Exception as e:
        tb = traceback.format_exc()
        return render_template('result.html', error=str(e), traceback=tb)


if __name__ == '__main__':
    # run development server
    app.run(host='0.0.0.0', port=8502, debug=True)
