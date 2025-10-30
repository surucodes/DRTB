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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'drtb-secret')

# upload and output folders (inside package folder)
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# directory for pretrained models
MODEL_DIR = os.path.join(BASE_DIR, 'outputs', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# default production behavior: prefer pretrained models (set to '1' to enable)
USE_PRETRAINED_DEFAULT = os.environ.get('USE_PRETRAINED', '1') == '1'


@app.route('/')
def index():
    """Show upload form and action choices."""
    # pass available pretrained models to the template
    models = find_pretrained_models()
    return render_template('index.html', pretrained_models=models)


def save_uploaded_file(f):
    filename = secure_filename(f.filename)
    dest = os.path.join(UPLOAD_DIR, filename)
    f.save(dest)
    return dest


def find_pretrained_models():
    """Return a sorted list of model filenames available in MODEL_DIR."""
    try:
        entries = []
        # Only consider files inside the outputs/models directory
        if os.path.isdir(MODEL_DIR):
            for f in os.listdir(MODEL_DIR):
                full = os.path.join(MODEL_DIR, f)
                if os.path.isfile(full) and f.lower().endswith(('.joblib', '.pkl', '.bin', '.model')):
                    entries.append(f)

        # sort deterministically (reverse alphabetical can put newer timestamped names first)
        items = sorted(entries, reverse=True)
        return items
    except Exception:
        return []


def load_pretrained_model(path):
    """Load a pretrained model. Try joblib first, then attempt CatBoost native load as fallback."""
    import joblib
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    # try joblib
    try:
        return joblib.load(str(p))
    except Exception:
        # try CatBoost
        try:
            from catboost import CatBoostClassifier
            m = CatBoostClassifier()
            m.load_model(str(p))
            return m
        except Exception:
            # as last resort, raise
            raise


@app.route('/run', methods=['POST'])
def run():
    try:
        # form values
        action = request.form.get('action')  # 'single' or 'selection'
        save_best = bool(request.form.get('save_best'))
        save_dir = request.form.get('save_dir') or 'outputs/models'
        # determine mode: 'pretrained' or 'selection'
        mode = request.form.get('mode')
        pretrained_name = request.form.get('pretrained_name') or ''
        if mode is None:
            # default behavior: prefer pretrained if available
            mode = 'pretrained' if (USE_PRETRAINED_DEFAULT and len(find_pretrained_models()) > 0) else 'selection'
        use_pretrained = (mode == 'pretrained')
        overwrite_if_improved = bool(request.form.get('overwrite_if_improved'))

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
        # If user requested to use a pretrained model, attempt to load and evaluate it
        if use_pretrained:
            # determine pretrained model path: explicit selection or latest in MODEL_DIR
            selected = None
            if pretrained_name:
                cand = os.path.join(MODEL_DIR, pretrained_name)
                if os.path.exists(cand):
                    selected = cand
                else:
                    cand2 = os.path.join(BASE_DIR, pretrained_name)
                    if os.path.exists(cand2):
                        selected = cand2
            else:
                # pick latest file from MODEL_DIR
                models = find_pretrained_models()
                if models:
                    selected = os.path.join(MODEL_DIR, models[0])

            if not selected:
                return render_template('result.html', error='No pretrained model found. Uncheck "Use pretrained" to run model selection.')

            try:
                model = load_pretrained_model(selected)
            except Exception as e:
                return render_template('result.html', error=f'Failed to load pretrained model: {e}')

            # evaluate
            try:
                accuracy = float(model.score(X_test, y_test))
            except Exception:
                accuracy = None
            try:
                y_pred = model.predict(X_test)
            except Exception:
                y_pred = None

            # create plots
            cm_fname = roc_fname = pr_fname = feat_fname = None
            try:
                from sklearn.metrics import confusion_matrix
                from drtb.metrics import print_confusion_matrix, plot_roc, plot_precision_recall, convert_binary_category_to_string
                if y_pred is not None:
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = print_confusion_matrix(cm, convert_binary_category_to_string(sorted(list(set(y_test)))))
                    cm_fname = f"confusion_pretrained_{int(np.random.uniform(0,1)*1e9)}.png"
                    fig_cm.savefig(os.path.join(OUTPUT_DIR, cm_fname), bbox_inches='tight')
                    plt.close(fig_cm)

                probs = None
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, 'decision_function'):
                    try:
                        probs = model.decision_function(X_test)
                    except Exception:
                        probs = None

                if probs is not None:
                    fig_roc, _ = plot_roc(y_test, probs)
                    roc_fname = f"roc_pretrained_{int(np.random.uniform(0,1)*1e9)}.png"
                    fig_roc.savefig(os.path.join(OUTPUT_DIR, roc_fname), bbox_inches='tight')
                    plt.close(fig_roc)

                    fig_pr, _ = plot_precision_recall(y_test, probs)
                    pr_fname = f"pr_pretrained_{int(np.random.uniform(0,1)*1e9)}.png"
                    fig_pr.savefig(os.path.join(OUTPUT_DIR, pr_fname), bbox_inches='tight')
                    plt.close(fig_pr)

                # feature importances
                try:
                    importances = None
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                    elif hasattr(model, 'coef_'):
                        importances = np.abs(np.ravel(model.coef_))
                    if importances is not None and len(importances) == X_train.shape[1]:
                        feat_names = getattr(X_train, 'columns', [f'feat_{i}' for i in range(X_train.shape[1])])
                        fig_f = plt.figure(figsize=(8, 4))
                        idx = np.argsort(importances)[::-1][:20]
                        plt.barh(range(len(idx)), importances[idx][::-1], color='#0b5ed7')
                        plt.yticks(range(len(idx)), [feat_names[i] for i in idx][::-1])
                        plt.title('Top feature importances')
                        feat_fname = f"feat_imp_pretrained_{int(np.random.uniform(0,1)*1e9)}.png"
                        fig_f.savefig(os.path.join(OUTPUT_DIR, feat_fname), bbox_inches='tight')
                        plt.close(fig_f)
                except Exception:
                    feat_fname = None
            except Exception:
                pass

            images = {'confusion': cm_fname, 'roc': roc_fname, 'pr': pr_fname, 'feat_imp': feat_fname}
            return render_template('result.html', single=False, best_name=os.path.basename(selected), report=[(os.path.basename(selected), accuracy if accuracy is not None else 0.0)], saved_path=selected, images=images)

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
            # compute accuracy for best model on test set
            try:
                best_acc = float(best_model.score(X_test, y_test))
            except Exception:
                best_acc = None

            # find existing pretrained model path to compare/overwrite
            existing_model_path = None
            if pretrained_name:
                candidate = os.path.join(MODEL_DIR, pretrained_name)
                if os.path.exists(candidate):
                    existing_model_path = candidate
                else:
                    candidate2 = os.path.join(BASE_DIR, pretrained_name)
                    if os.path.exists(candidate2):
                        existing_model_path = candidate2
            else:
                # try to find a file that matches the best_name in MODEL_DIR
                try:
                    for f in os.listdir(MODEL_DIR):
                        if best_name.lower() in f.lower():
                            existing_model_path = os.path.join(MODEL_DIR, f)
                            break
                except Exception:
                    existing_model_path = None

            # compute existing accuracy if possible
            existing_acc = None
            if existing_model_path:
                try:
                    existing_model = load_pretrained_model(existing_model_path)
                    existing_acc = float(existing_model.score(X_test, y_test))
                except Exception:
                    existing_acc = None

            # If overwrite_if_improved is requested and the new model is better, overwrite existing
            if overwrite_if_improved and existing_model_path is not None and best_acc is not None:
                try:
                    if existing_acc is None or (best_acc > existing_acc):
                        saved_path = save_model(best_model, existing_model_path)
                except Exception:
                    saved_path = None

            # if user explicitly requested save_best (non-overwrite), save to chosen folder
            if save_best and saved_path is None:
                save_folder = os.path.join(BASE_DIR, save_dir)
                os.makedirs(save_folder, exist_ok=True)
                fname = os.path.join(save_folder, f"{best_name}_model.joblib")
                saved_path = save_model(best_model, fname)

            # convert report to list for template
            items = sorted(report.items(), key=lambda x: -x[1])

            # create a horizontal bar chart of accuracies
            try:
                fig = plt.figure(figsize=(8, 4))
                names = [n for n, a in items]
                accs = [a for n, a in items]
                y_pos = np.arange(len(names))
                plt.barh(y_pos, accs, align='center', color='#0b5ed7')
                plt.yticks(y_pos, names)
                plt.xlabel('Accuracy')
                plt.xlim(0, 1)
                plt.gca().invert_yaxis()
                acc_fname = f"model_accuracies_{int(np.random.uniform(0,1)*1e9)}.png"
                acc_path = os.path.join(OUTPUT_DIR, acc_fname)
                fig.savefig(acc_path, bbox_inches='tight')
                plt.close(fig)
            except Exception:
                acc_fname = None

            # best-model visualizations (confusion, roc, pr, feature importances)
            cm_fname = roc_fname = pr_fname = feat_fname = None
            try:
                from sklearn.metrics import confusion_matrix
                from drtb.metrics import print_confusion_matrix, plot_roc, plot_precision_recall, convert_binary_category_to_string

                y_pred = best_model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = print_confusion_matrix(cm, convert_binary_category_to_string(sorted(list(set(y_test)))))
                cm_fname = f"confusion_{best_name}_{int(np.random.uniform(0,1)*1e9)}.png"
                fig_cm.savefig(os.path.join(OUTPUT_DIR, cm_fname), bbox_inches='tight')
                plt.close(fig_cm)

                probs = None
                if hasattr(best_model, 'predict_proba'):
                    probs = best_model.predict_proba(X_test)[:, 1]
                elif hasattr(best_model, 'decision_function'):
                    try:
                        probs = best_model.decision_function(X_test)
                    except Exception:
                        probs = None

                if probs is not None:
                    fig_roc, _ = plot_roc(y_test, probs)
                    roc_fname = f"roc_{best_name}_{int(np.random.uniform(0,1)*1e9)}.png"
                    fig_roc.savefig(os.path.join(OUTPUT_DIR, roc_fname), bbox_inches='tight')
                    plt.close(fig_roc)

                    fig_pr, _ = plot_precision_recall(y_test, probs)
                    pr_fname = f"pr_{best_name}_{int(np.random.uniform(0,1)*1e9)}.png"
                    fig_pr.savefig(os.path.join(OUTPUT_DIR, pr_fname), bbox_inches='tight')
                    plt.close(fig_pr)

                # feature importances if available
                try:
                    importances = None
                    if hasattr(best_model, 'feature_importances_'):
                        importances = best_model.feature_importances_
                    elif hasattr(best_model, 'coef_'):
                        arr = np.ravel(best_model.coef_)
                        importances = np.abs(arr)
                    if importances is not None and len(importances) == X_train.shape[1]:
                        feat_names = getattr(X_train, 'columns', [f'feat_{i}' for i in range(X_train.shape[1])])
                        fig_f = plt.figure(figsize=(8, 4))
                        idx = np.argsort(importances)[::-1][:20]
                        plt.barh(range(len(idx)), importances[idx][::-1], color='#0b5ed7')
                        plt.yticks(range(len(idx)), [feat_names[i] for i in idx][::-1])
                        plt.title('Top feature importances')
                        feat_fname = f"feat_imp_{best_name}_{int(np.random.uniform(0,1)*1e9)}.png"
                        fig_f.savefig(os.path.join(OUTPUT_DIR, feat_fname), bbox_inches='tight')
                        plt.close(fig_f)
                except Exception:
                    feat_fname = None
            except Exception:
                pass

            images = {'accuracies': acc_fname, 'confusion': cm_fname, 'roc': roc_fname, 'pr': pr_fname, 'feat_imp': feat_fname}
            return render_template('result.html', single=False, best_name=best_name, report=items, saved_path=saved_path, images=images)

    except Exception as e:
        tb = traceback.format_exc()
        return render_template('result.html', error=str(e), traceback=tb)


@app.route('/outputs/<path:filename>')
def outputs_file(filename):
    # serve files from the package OUTPUT_DIR
    full = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(full):
        return send_file(full)
    return ('Not found', 404)


if __name__ == '__main__':
    # run development server
    app.run(host='0.0.0.0', port=8502, debug=True)
