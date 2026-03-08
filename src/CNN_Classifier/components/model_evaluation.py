import os

from dotenv import load_dotenv
from dataclasses import dataclass
import joblib
import numpy as np
from seaborn import cm
from sklearn.calibration import label_binarize
from CNN_Classifier import logger
from CNN_Classifier.entity.config_entity import EvaluationConfig
import mlflow
import dagshub
from scipy.special import softmax
import matplotlib.pyplot as plt
from sklearn.metrics import (auc, classification_report, 
                             confusion_matrix,ConfusionMatrixDisplay,
                               roc_curve, roc_auc_score, 
                               f1_score, precision_score, recall_score)
from sklearn.preprocessing import label_binarize
from urllib.parse import urlparse
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

class ModelEvaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    # Building test set
    @dataclass
    class TestData:
        X: tf.data.Dataset
        y: np.ndarray
        target_names: list[str]
    
    
    @staticmethod
    def _load_keras_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    @staticmethod
    def _load_sklearn_model(path: Path):
        return joblib.load(str(path))
    
    @staticmethod
    def _evaluation_metrics(y_true, y_pred, y_score=None):
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        roc_score_ = roc_auc_score(y_true, y_score, average='weighted', multi_class='ovr') if y_score is not None else None

        return f1, precision, recall, roc_score_
    @staticmethod
    def _evaluation_reports(y_true, y_pred, target_names,output_dir: Path,y_score=None):
        report = classification_report(y_true,
                                        y_pred,
                                          output_dict=True, 
                                       target_names=target_names,
                                         zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(12, 8))  # wider figure

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(ax=ax, values_format='d', cmap="viridis", colorbar=True)

        # Fix horizontal label mess
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        cm_path = Path(output_dir) / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300)
        plt.close(fig)
        if y_score is not None:
            n_classes = y_score.shape[1]
            y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))  # (N,C)
        
            # Flatten everything (micro-average)
            fpr, tpr, _ = roc_curve(
                y_true_bin.ravel(),
                y_score.ravel()
            )

            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, label=f"Micro-average ROC (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.tight_layout()
            roc_path = Path(output_dir) / "roc_curve.png"
            plt.savefig(roc_path, dpi=300)
            plt.close()

        return report,cm_path, roc_path if y_score is not None else None
    @staticmethod
    def _prepare_test_data_for_CNN(test_data_dir: Path, image_size: tuple, batch_size:int, seed: int) -> TestData:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            directory=test_data_dir,
            label_mode='int',
            labels='inferred',
            batch_size=batch_size,
            image_size=image_size,
            seed=seed,
            shuffle=False
        )
        target_names = list(test_ds.class_names)
        test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
        y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
        return ModelEvaluation.TestData(X=test_ds, y=y_true,target_names=target_names)
    @staticmethod
    def _prepare_test_data_for_ML(test_data_dir: Path,
                                   image_size: tuple, 
                                   feature_extractor:tf.keras.Model,
                                     batch_size:int, seed: int) -> TestData:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            directory=test_data_dir,
            label_mode='int',
            labels='inferred',
            batch_size=batch_size,
            image_size=image_size,
            seed=seed,
            shuffle=False
        )
        target_names = list(test_ds.class_names)
        test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
        X_list, y_list = [], []
        for X_batch, y_batch in test_ds:
            features = feature_extractor(X_batch, training=False).numpy()
            X_list.append(features)
            y_list.append(y_batch.numpy())
        X = np.concatenate(X_list, axis=0)
        y_true = np.concatenate(y_list, axis=0)
        return ModelEvaluation.TestData(X=X, y=y_true,target_names=target_names)
    
    def _model_logger(self,model, model_name:str, X,y_true,target_names):
            
        if isinstance(model, tf.keras.Model):
            y_score = model.predict(X, verbose=0)          # (N,C)
            y_pred = y_score.argmax(axis=1)               # (N,)
        else:
            # sklearn model
            y_pred = model.predict(X)                     # (N,)
            if hasattr(model, "decision_function"):
                y_score = model.decision_function(X)      # (N,C) for multiclass 
                y_score = softmax(y_score, axis=1)

            elif hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X)          # (N,C)
            else:
                y_score = None

        f1, precision, recall, roc_score_ = self._evaluation_metrics(y_true, y_pred, y_score)
        report, cm_path, roc_path = self._evaluation_reports(y_true, y_pred,
                                            target_names=target_names, 
                                            output_dir=self.config.root_dir,
                                                y_score=y_score)
        dagshub.init(repo_owner='Shahriyar-1988', repo_name="HealthCare-Pulmonary-diagnosis", mlflow=True)
        
        mlflow.set_registry_uri(self.config.mlflow_uri)
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment(f"Pulmonary_Diagnosis_Evaluation_{model_name}")

        
        with mlflow.start_run():
            mlflow.log_param("model_name", model_name)
            try:
                mlflow.keras.log_model(model, artifact_path=model_name)
            except Exception as e:
                mlflow.sklearn.log_model(model, artifact_path=model_name)
            if roc_score_ is not None:
                mlflow.log_metric("roc_auc_score", round(roc_score_, 3))
                mlflow.log_artifact(str(roc_path))

            mlflow.log_metric("f1_score", round(f1, 3))
            mlflow.log_metric("precision", round(precision, 3))
            mlflow.log_metric("recall", round(recall, 3))
            mlflow.log_dict(report, artifact_file=f"{model_name}_classification_report.json")
            mlflow.log_artifact(str(cm_path))
    def evaluate_model(self, log_model: bool = True):
        # Load model
        cnn_model = self._load_keras_model(self.config.trained_model_dir)
        svm_model = self._load_sklearn_model(self.config.trained_model_dir_svm)
        pca_svm_model = self._load_sklearn_model(self.config.trained_model_dir_pca_svm)
        kpca_svm_model = self._load_sklearn_model(self.config.trained_model_dir_kpca_svm)

        # Prepare test data
        test_data_cnn = self._prepare_test_data_for_CNN(self.config.test_data_dir,
                                                         self.config.IMAGE_SIZE,
                                                           self.config.BATCH_SIZE,
                                                             self.config.SEED)
        feature_extractor = self._load_keras_model(self.config.feature_extract_dir)
        test_data_ml = self._prepare_test_data_for_ML(self.config.test_data_dir,
                                                         self.config.IMAGE_SIZE,
                                                           feature_extractor,
                                                             self.config.BATCH_SIZE,
                                                             self.config.SEED)
        

        # Evaluate and Log results
        if log_model:
            self._model_logger(cnn_model, "CNN", test_data_cnn.X, test_data_cnn.y, test_data_cnn.target_names)
            self._model_logger(svm_model, "SVM", test_data_ml.X, test_data_ml.y, test_data_ml.target_names)
            self._model_logger(pca_svm_model, "PCA_SVM", test_data_ml.X, test_data_ml.y, test_data_ml.target_names)
            self._model_logger(kpca_svm_model, "KPCA_SVM", test_data_ml.X, test_data_ml.y, test_data_ml.target_names)
        logger.info("Model Successfully logged into MLflow.")