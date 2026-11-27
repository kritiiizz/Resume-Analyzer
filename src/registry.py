"""One interface over every trained model, used by the demo and by error analysis.

Each model exposes:  .name   .kind   .predict(text, k) -> [(label, prob), ...]
                     .top_words(text, k) -> [(word, weight), ...]  (may be empty)

Where models are found (all optional):
  runs/<run>/            BiLSTM / TextCNN / CNN+BiLSTM runs from src/dl/train.py
  models/*.joblib        scikit-learn pipelines (TF-IDF + LR / SVM / NB) saved with joblib.dump(pipeline, path)
  models/<dir>/          a fine-tuned Hugging Face model saved with save_pretrained (has config.json)
"""
import glob
import json
import os

import numpy as np

HEAD_TOKENS, TAIL_TOKENS = 128, 382  # DistilBERT head+tail; change if training used another ratio


class DLModel:
    kind = "Deep learning (RNN/CNN)"

    def __init__(self, run_dir):
        from src.dl.predict import load_run
        self.name = os.path.basename(run_dir.rstrip("/"))
        self._clf = load_run(run_dir)

    def predict(self, text, k=3):
        return self._clf.predict(text, k)

    def top_words(self, text, k=15):
        return self._clf.top_words(text, k)


class SklearnModel:
    kind = "Classical baseline"

    def __init__(self, path):
        import joblib
        base = os.path.splitext(os.path.basename(path))[0]
        self.name = os.path.basename(os.path.dirname(path)) if base == "pipeline" else base
        self._pipe = joblib.load(path)

    def predict(self, text, k=3):
        classes = self._pipe.classes_
        if hasattr(self._pipe, "predict_proba"):
            p = self._pipe.predict_proba([text])[0]
        else:  # LinearSVC has no probabilities: softmax over the decision scores (a ranking, not calibrated)
            s = np.asarray(self._pipe.decision_function([text]))[0]
            p = np.exp(s - s.max()); p = p / p.sum()
        return [(str(classes[i]), float(p[i])) for i in np.argsort(-p)[:k]]

    def top_words(self, text, k=15):
        """Words in this text that push most towards the predicted class (linear models only)."""
        try:
            vec, clf = self._pipe.steps[0][1], self._pipe.steps[-1][1]
            x = self._pipe[:-1].transform([text])
            coef = clf.coef_ if hasattr(clf, "coef_") else clf.feature_log_prob_
            cls = int(np.argmax(self._pipe.decision_function([text])[0])) if hasattr(self._pipe, "decision_function") \
                else int(np.argmax(self._pipe.predict_proba([text])[0]))
            contrib = x.multiply(coef[cls]).toarray()[0]
            names = vec.get_feature_names_out()
            return [(names[i], float(contrib[i])) for i in np.argsort(-contrib)[:k] if contrib[i] > 0]
        except Exception:
            return []


class HFModel:
    kind = "Transformer"

    def __init__(self, path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._torch = torch
        self.name = os.path.basename(path.rstrip("/"))
        self._tok = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        with open(os.path.join(path, "config.json")) as f:
            id2label = json.load(f).get("id2label", {})
        self._labels = [id2label[str(i)] for i in range(len(id2label))]

    def predict(self, text, k=3):
        ids = self._tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) > HEAD_TOKENS + TAIL_TOKENS:
            ids = ids[:HEAD_TOKENS] + ids[-TAIL_TOKENS:]
        ids = [self._tok.cls_token_id] + ids + [self._tok.sep_token_id]
        with self._torch.no_grad():
            logits = self._model(input_ids=self._torch.tensor([ids])).logits[0]
        p = self._torch.softmax(logits, -1).numpy()
        return [(self._labels[i], float(p[i])) for i in np.argsort(-p)[:k]]

    def top_words(self, text, k=15):
        return []


def discover(root="."):
    """Return {display name: (loader class, path)} for every model found on disk."""
    found = {}
    for cfg in sorted(glob.glob(os.path.join(root, "runs", "*", "config.json"))):
        d = os.path.dirname(cfg)
        if os.path.exists(os.path.join(d, "best.pt")):
            found[f"{os.path.basename(d)}"] = (DLModel, d)
    for p in sorted(glob.glob(os.path.join(root, "models", "*.joblib"))):
        found[os.path.splitext(os.path.basename(p))[0]] = (SklearnModel, p)
    for cfg in sorted(glob.glob(os.path.join(root, "models", "*", "config.json"))):
        d = os.path.dirname(cfg)
        found[os.path.basename(d)] = (HFModel, d)
    # Phuc's layout: models/task_a/<name>/pipeline.joblib and models/task_a/distilbert_head_tail/runs/<run>/best
    for p in sorted(glob.glob(os.path.join(root, "models", "task_a", "*", "pipeline.joblib"))):
        name = os.path.basename(os.path.dirname(p))
        if name != "dummy":
            found[name] = (SklearnModel, p)
    for cfg in sorted(glob.glob(os.path.join(root, "models", "task_a", "*", "runs", "*", "best", "config.json"))):
        found["distilbert_head_tail"] = (HFModel, os.path.dirname(cfg))  # last (newest) run wins
    return found
