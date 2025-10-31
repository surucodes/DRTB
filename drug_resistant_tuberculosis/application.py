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

# Ensure the package directory is on sys.path so 'import drtb' works when running inside Docker
import sys
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# default production behavior: prefer pretrained models (set to '1' to enable)
USE_PRETRAINED_DEFAULT = os.environ.get('USE_PRETRAINED', '1') == '1'

# one-time initializer flag for manifest backfill
_FEATURE_MANIFESTS_INITIALIZED = False


def ensure_feature_manifests():
    """Best-effort backfill of feature manifests for existing pretrained models.

    If a model lacks <model>.features.json, we'll try to infer expected features:
    - Prefer model.feature_names_in_ if available
    - Else, use columns from preprocessing the default dataset (dr_dataset.csv) if the
      length matches model.n_features_in_, then write manifest.
    This improves prediction alignment for legacy models trained without feature manifests.
    """
    global _FEATURE_MANIFESTS_INITIALIZED
    if _FEATURE_MANIFESTS_INITIALIZED:
        return
    try:
        models = find_pretrained_models()
        if not models:
            _FEATURE_MANIFESTS_INITIALIZED = True
            return
        default_csv = os.path.join(BASE_DIR, 'dr_dataset.csv')
        default_feats = None
        if os.path.exists(default_csv):
            try:
                from drtb.data import load_data
                from drtb.preprocess import preprocess_pipeline, split_X_y
                import pandas as pd
                df = load_data(default_csv)
                dfp = preprocess_pipeline(df)
                X_all, _ = split_X_y(dfp)
                default_feats = list(getattr(X_all, 'columns', []))
            except Exception:
                default_feats = None
        for name in models:
            full = resolve_pretrained_fullpath(name)
            if not full:
                continue
            # skip if manifest exists
            from pathlib import Path as _P
            mf = str(_P(full)) + '.features.json'
            if os.path.exists(mf):
                continue
            try:
                mdl = load_pretrained_model(full)
                # 1) if model carries names, write them
                if hasattr(mdl, 'feature_names_in_'):
                    _save_feature_manifest(full, list(mdl.feature_names_in_))
                    continue
                # 2) else, if we have default feature list and dims match, use that
                if default_feats and hasattr(mdl, 'n_features_in_'):
                    if len(default_feats) == int(mdl.n_features_in_):
                        _save_feature_manifest(full, default_feats)
            except Exception:
                # ignore models we cannot load
                continue
    finally:
        _FEATURE_MANIFESTS_INITIALIZED = True


# Log runtime versions at startup (helps diagnose Railway env)
try:
    import numpy as _np, sklearn as _sk, joblib as _joblib
    app.logger.info("Runtime versions -> numpy=%s, sklearn=%s, joblib=%s", _np.__version__, _sk.__version__, _joblib.__version__)
except Exception:
    pass


@app.route('/health', methods=['GET'])
def health():
    """Simple health/diagnostic endpoint returning versions and discovered model filenames."""
    try:
        import numpy as _np, sklearn as _sk, joblib as _joblib
        models = find_pretrained_models()
        return {
            "numpy": getattr(_np, "__version__", None),
            "sklearn": getattr(_sk, "__version__", None),
            "joblib": getattr(_joblib, "__version__", None),
            "models_found": models,
        }
    except Exception as e:
        return {"error": str(e)}, 500


@app.route('/')
def index():
    """Show upload form and action choices."""
    # pass available pretrained models to the template
    ensure_feature_manifests()
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

        # Candidate directories to search for pretrained models. Keep MODEL_DIR first
        # so files in outputs/models are preferred when available.
        candidate_dirs = [
            MODEL_DIR,
            os.path.join(BASE_DIR, 'outputs', 'models'),
            os.path.join(BASE_DIR, 'models'),
            os.path.join(BASE_DIR),
            os.path.join(os.path.dirname(BASE_DIR), 'outputs', 'models'),
        ]

        # also respect an explicit env var if the deploy environment sets it
        env_dir = os.environ.get('PRETRAINED_MODELS_DIR')
        if env_dir:
            candidate_dirs.insert(0, env_dir)

        seen = set()
        for d in candidate_dirs:
            try:
                if not d or not os.path.isdir(d):
                    continue
                for f in os.listdir(d):
                    full = os.path.join(d, f)
                    if os.path.isfile(full) and f.lower().endswith(('.joblib', '.pkl', '.bin', '.model')):
                        if f not in seen:
                            seen.add(f)
                            entries.append(f)
            except Exception:
                # ignore unreadable dirs
                continue

        # sort deterministically (reverse alphabetical can put newer timestamped names first)
        items = sorted(entries, reverse=True)
        app.logger.debug('find_pretrained_models looked in: %s and found: %s', candidate_dirs, items)
        return items
    except Exception:
        return []


def resolve_pretrained_fullpath(filename: str):
    """Given a filename (basename), return the full path searching candidate dirs or None."""
    if not filename:
        return None
    candidate_dirs = [
        MODEL_DIR,
        os.path.join(BASE_DIR, 'outputs', 'models'),
        os.path.join(BASE_DIR, 'models'),
        os.path.join(BASE_DIR),
        os.path.join(os.path.dirname(BASE_DIR), 'outputs', 'models'),
    ]
    env_dir = os.environ.get('PRETRAINED_MODELS_DIR')
    if env_dir:
        candidate_dirs.insert(0, env_dir)

    for d in candidate_dirs:
        try:
            if not d:
                continue
            p = os.path.join(d, filename)
            if os.path.exists(p) and os.path.isfile(p):
                return p
        except Exception:
            continue
    return None


def _save_feature_manifest(model_path: str, feature_names):
    """Save a small JSON manifest listing the feature names expected by the model.

    The manifest will be written to <model_path>.features.json next to the model file.
    """
    try:
        import json
        p = Path(model_path)
        manifest_path = str(p) + '.features.json'
        with open(manifest_path, 'w', encoding='utf-8') as fh:
            json.dump(list(feature_names), fh)
        app.logger.info('Wrote feature manifest: %s', manifest_path)
        return manifest_path
    except Exception:
        app.logger.exception('Failed to write feature manifest for %s', model_path)
        return None


def _load_feature_manifest(model_path: str):
    """Load feature manifest if present. Returns list of feature names or None."""
    try:
        import json
        p = Path(model_path)
        manifest_path = str(p) + '.features.json'
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        return None
    except Exception:
        app.logger.exception('Failed to read feature manifest for %s', model_path)
        return None


def load_pretrained_model(path):
    """Load a pretrained model. Try joblib first, then attempt CatBoost native load as fallback."""
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    ext = p.suffix.lower()
    # choose loader based on file extension when possible
    try:
        if ext in ('.joblib', '.pkl'):
            import joblib
            return joblib.load(str(p))

        if ext in ('.cbm', '.catboost'):
            from catboost import CatBoostClassifier
            m = CatBoostClassifier()
            m.load_model(str(p))
            return m

        # for other extensions (.model, .bin) try to infer loader by attempting joblib first
        # but if joblib fails, do NOT automatically try CatBoost unless the extension explicitly
        # indicates CatBoost; re-raise the joblib error so failures are visible in logs instead
        try:
            import joblib
            return joblib.load(str(p))
        except Exception as e_job:
            app.logger.exception('joblib.load failed for %s: %s', str(p), e_job)
            # if the file looks like a CatBoost binary (heuristic), try CatBoost as a last resort
            try:
                head = p.open('rb').read(64)
                if b'CatBoost' in head or b'catboost' in head:
                    from catboost import CatBoostClassifier
                    m = CatBoostClassifier()
                    m.load_model(str(p))
                    return m
            except Exception:
                pass
            # otherwise raise the original joblib error
            raise
    except Exception:
        app.logger.exception('Failed to load pretrained model: %s', str(p))
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
                selected = resolve_pretrained_fullpath(pretrained_name)
            else:
                # pick latest discovered model from find_pretrained_models()
                models = find_pretrained_models()
                if models:
                    selected = resolve_pretrained_fullpath(models[0])

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
                existing_model_path = resolve_pretrained_fullpath(pretrained_name)
            else:
                # try to find a file that matches the best_name in MODEL_DIR
                try:
                    # search candidate dirs for a filename containing best_name
                    for cand in find_pretrained_models():
                        if best_name.lower() in cand.lower():
                            existing_model_path = resolve_pretrained_fullpath(cand)
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
                        # save feature manifest alongside model for deterministic alignment at prediction
                        try:
                            feat_names = getattr(X_train, 'columns', None)
                            if feat_names is not None:
                                _save_feature_manifest(saved_path or existing_model_path, feat_names)
                        except Exception:
                            app.logger.exception('Failed to save feature manifest after overwrite')
                except Exception:
                    saved_path = None

            # if user explicitly requested save_best (non-overwrite), save to chosen folder
            if save_best and saved_path is None:
                save_folder = os.path.join(BASE_DIR, save_dir)
                os.makedirs(save_folder, exist_ok=True)
                fname = os.path.join(save_folder, f"{best_name}_model.joblib")
                saved_path = save_model(best_model, fname)
                try:
                    feat_names = getattr(X_train, 'columns', None)
                    if feat_names is not None:
                        _save_feature_manifest(saved_path or fname, feat_names)
                except Exception:
                    app.logger.exception('Failed to save feature manifest for saved best model')

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


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Allow a user to pick a pretrained model and submit feature values for a single prediction.

    GET -> show the form; POST -> run preprocess on the single-row inputs, load model, predict and show result.
    """
    try:
        from drtb.preprocess import preprocess_pipeline
        import pandas as pd

        ensure_feature_manifests()
        models = find_pretrained_models()
        if request.method == 'GET':
            return render_template('predict.html', pretrained_models=models)

        # POST: collect feature values from form
        form = request.form
        # expected raw fields matching original CSV header
        fields = ['Gender', 'Age', 'Contact DR', 'Smoking', 'Alcohol', 'Cavitary pulmonary', 'Diabetes', 'Nutritional', 'TBoutside']
        row = {}
        for f in fields:
            # form keys use safe names (no spaces) so try both
            key = f if f in form else f.replace(' ', '_')
            val = form.get(key)
            # normalize checkboxes (on -> Yes)
            if val is None:
                # if not provided, default to 'No' for yes/no fields and empty for others
                if f in ['Contact DR', 'Smoking', 'Alcohol', 'Cavitary pulmonary', 'Diabetes', 'TBoutside']:
                    val = 'No'
                else:
                    val = ''
            row[f] = val

        df_in = pd.DataFrame([row])
        # apply preprocessing used during training
        try:
            df_proc = preprocess_pipeline(df_in)
        except Exception as e:
            return render_template('predict_result.html', error=f'Preprocessing failed: {e}')

        # load model
        pretrained_name = request.form.get('pretrained_name') or ''
        model_path = None
        if pretrained_name:
            model_path = resolve_pretrained_fullpath(pretrained_name)
        else:
            models = find_pretrained_models()
            if models:
                model_path = resolve_pretrained_fullpath(models[0])

        if not model_path:
            return render_template('predict_result.html', error='No pretrained model selected or found.')

        try:
            model = load_pretrained_model(model_path)
            app.logger.info('Using pretrained model for prediction: %s', model_path)
        except Exception as e:
            return render_template('predict_result.html', error=f'Failed to load model: {e}')

        # align features to model expectations when possible
        X_input = df_proc.copy()
        # drop Class if present
        if 'Class' in X_input.columns:
            X_input = X_input.drop(columns=['Class'])

        try:
            # Prefer a saved feature manifest (written during training) for exact alignment
            manifest = _load_feature_manifest(model_path)
            if manifest:
                expected = list(manifest)
                # add missing columns with zeros
                for c in expected:
                    if c not in X_input.columns:
                        X_input[c] = 0
                # ensure ordering
                X_input = X_input[expected]
            else:
                # fall back to model-provided hints
                if hasattr(model, 'feature_names_in_'):
                    expected = list(model.feature_names_in_)
                    for c in expected:
                        if c not in X_input.columns:
                            X_input[c] = 0
                    X_input = X_input[expected]
                elif hasattr(model, 'n_features_in_'):
                    n_expected = int(model.n_features_in_)
                    if X_input.shape[1] < n_expected:
                        # add dummy zero columns
                        cur = X_input.shape[1]
                        for i in range(n_expected - cur):
                            X_input[f'_pad_{i}'] = 0
                    elif X_input.shape[1] > n_expected:
                        # attempt to reduce to n_expected columns (best-effort)
                        X_input = X_input.iloc[:, :n_expected]
        except Exception as e:
            app.logger.exception('Failed to align input features: %s', e)

        # convert to numpy array for prediction
        try:
            X_arr = X_input.values
            pred = model.predict(X_arr)
            pred_label = int(pred[0]) if hasattr(pred, '__len__') else int(pred)
        except Exception as e:
            app.logger.exception('Prediction failed with model %s', model_path)
            return render_template('predict_result.html', error=f'Failed to run prediction: {e}')

        proba = None
        try:
            if hasattr(model, 'predict_proba'):
                pr = model.predict_proba(X_arr)
                # try to take probability of positive class
                if pr.shape[1] == 2:
                    proba = float(pr[0, 1])
                else:
                    proba = float(pr[0, 0])
        except Exception:
            proba = None

        from drtb.metrics import convert_binary_category_to_string
        pred_str = convert_binary_category_to_string([pred_label])[0]

        return render_template('predict_result.html', model_name=os.path.basename(model_path), prediction=pred_label, prediction_str=pred_str, probability=proba, input_row=row)
    except Exception as e:
        tb = traceback.format_exc()
        return render_template('predict_result.html', error=str(e), traceback=tb)


if __name__ == '__main__':
    # run development server
    app.run(host='0.0.0.0', port=8502, debug=True)
