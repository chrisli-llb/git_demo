"""Simple binary classification example using logistic regression.

Run:
    python binary_classifier.py
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def main() -> None:
    # Generate a toy binary classification dataset.
    x, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    main()
