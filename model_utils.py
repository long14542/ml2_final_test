import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import learning_curve
from config import Config


class ModelUtils:
    def __init__(self, numerical_features, categorical_features):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.plots = {}
        self.model = None
        self.class_names = None

    def build_and_train_models(self, X_train, y_train):
        """Build and train a Random Forest model with anti-overfitting parameters"""
        self.class_names = sorted(np.unique(y_train))

        # Initialize a Random Forest model with optimized parameters
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=20,
            max_features='sqrt',
            random_state=42,
            oob_score=True,
            class_weight='balanced'
        )

        # Train the model
        self.model.fit(X_train, y_train)

        return self.model

    def evaluate_model(self, X_train, X_test, y_train, y_test):
        """Evaluate the model on the test set"""
        if self.model is None:
            raise ValueError("Model has not been trained. Call build_and_train_models first.")

        # Predict labels
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Compute evaluation metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision_micro': precision_score(y_test, y_test_pred, average='micro'),
            'recall_micro': recall_score(y_test, y_test_pred, average='micro'),
            'f1_micro': f1_score(y_test, y_test_pred, average='micro'),
            'classification_report': classification_report(y_test, y_test_pred)
        }

        # Compute the confusion matrix
        self.conf_matrix = confusion_matrix(y_test, y_test_pred)

        return metrics

    def plot_confusion_matrix(self, X_test, y_test):
        """Plot the confusion matrix"""
        if self.model is None:
            raise ValueError("Model has not been trained. Call build_and_train_models first.")

        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.xlabel('Predicted value')
        plt.ylabel('Actual value')
        plt.title('Confusion matrix')

        self.plots['confusion_matrix'] = plt.gcf()
        plt.close()

    def plot_roc_curve(self, X_test, y_test):
        """Plot the ROC curve"""
        if self.model is None:
            raise ValueError("Model has not been trained. Call build_and_train_models first.")

        # If it is a binary classification problem
        if len(self.class_names) == 2:
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.title('ROC curve')
            plt.legend(loc="lower right")

            self.plots['roc_curve'] = plt.gcf()
            plt.close()

    def plot_precision_recall_curve(self, X_test, y_test):
        """Plot the Precision-Recall curve"""
        if self.model is None:
            raise ValueError("Model has not been trained. Call build_and_train_models first.")

        # If it is a binary classification problem
        if len(self.class_names) == 2:
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            ap = average_precision_score(y_test, y_pred_proba)

            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2, label=f'AP = {ap:.2f}')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall curve')
            plt.legend(loc="lower left")

            self.plots['precision_recall_curve'] = plt.gcf()
            plt.close()

    def plot_feature_importance(self, feature_names):
        """Plot the importance of features"""
        if self.model is None:
            raise ValueError("Model has not been trained. Call build_and_train_models first.")

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 8))
        plt.title('Feature Importance')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()

        self.plots['feature_importance'] = plt.gcf()
        plt.close()

    def plot_learning_curve(self, X_train, y_train, cv=5):
        """Plot the learning curve"""
        if self.model is None:
            raise ValueError("Model has not been trained. Call build_and_train_models first.")

        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X_train, y_train, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.title('Learning Curve')
        plt.xlabel('Number of Training Samples')
        plt.ylabel('Score')
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.legend(loc="best")

        self.plots['learning_curve'] = plt.gcf()
        plt.close()

    def save_all_plots(self, output_dir=None):
        """Save all plots"""
        if output_dir is None:
            output_dir = Config.OUTPUT_DIR

        os.makedirs(output_dir, exist_ok=True)

        for name, fig in self.plots.items():
            fig.savefig(os.path.join(output_dir, f'{name}.png'))
