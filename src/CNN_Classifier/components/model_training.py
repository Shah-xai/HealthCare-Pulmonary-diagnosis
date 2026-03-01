from xml.parsers.expat import model

from CNN_Classifier import logger
from CNN_Classifier.entity.config_entity import ModelTrainingConfig
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA, KernelPCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from CNN_Classifier.utils.common import save_json,save_object

class ModelTraining:
    def __init__(self, config:ModelTrainingConfig):
        self.config = config
    def _input_preparation_for_CNN(self):
        train_ds = tf.keras.utils.image_dataset_from_directory( directory=self.config.training_data_dir,
                                                               label_mode='categorical',
                                                               labels='inferred',
                                                                batch_size=self.config.BATCH_SIZE,
                                                                image_size=self.config.IMAGE_SIZE,
                                                                seed=self.config.SEED,
                                                                shuffle=True)
        val_ds = tf.keras.utils.image_dataset_from_directory( directory=self.config.validation_data_dir,
                                                               label_mode='categorical',
                                                               labels='inferred',
                                                                batch_size=self.config.BATCH_SIZE,
                                                                image_size=self.config.IMAGE_SIZE,
                                                                seed=self.config.SEED,
                                                                shuffle=False
        )

        augmntation_layer = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(height_factor=0.02, width_factor=0.02),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.RandomGaussianBlur(0.2)
        ])
        train_ds = train_ds.map(lambda x, y: (augmntation_layer(x, training=True), y))
        train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        logger.info("Input data prepared for CNN training.")
        return train_ds, val_ds
    def _input_preparation_for_ML(self):
        train_ds = tf.keras.utils.image_dataset_from_directory( directory=self.config.training_data_dir,
                                                               label_mode='int',
                                                                labels='inferred',
                                                                batch_size=self.config.BATCH_SIZE,
                                                                image_size=self.config.IMAGE_SIZE,
                                                                seed=self.config.SEED,
                                                                shuffle=True)
        val_ds = tf.keras.utils.image_dataset_from_directory( directory=self.config.validation_data_dir,
                                                               label_mode='int',
                                                                labels='inferred',
                                                                batch_size=self.config.BATCH_SIZE,
                                                                image_size=self.config.IMAGE_SIZE,
                                                                seed=self.config.SEED,
                                                                shuffle=False)
        train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        logger.info("Input data prepared for ML training.")
        return train_ds, val_ds
    # A function to extract features using the feature extractor model for feeding into ML models
    def _feature_extraction(self, feature_extractor, dataset):
        x, y = [], []
        for xbatch, ybatch in dataset:
            features = feature_extractor(xbatch, training=False)
            x.append(features.numpy())
            y.append(ybatch.numpy())
        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)
        return x, y
    @staticmethod
    def report_f1(clf, X_train, y_train, X_val, y_val):
        # Predictions
        pred_train = clf.predict(X_train)
        pred_val   = clf.predict(X_val)

        # F1 scores
        train_f1_macro = f1_score(y_train, pred_train, average="macro")
        val_f1_macro   = f1_score(y_val, pred_val, average="macro")
        return {
            "train_f1_macro": round(train_f1_macro,3),
            "val_f1_macro": round(val_f1_macro,3),
            "overfitting_gap": round(abs(train_f1_macro - val_f1_macro),3),
        }

    def train_model(self):
        #Transfer Learning
        train_ds, val_ds = self._input_preparation_for_CNN()
        model = tf.keras.models.load_model(self.config.updated_model_dir)
        model.fit(train_ds, validation_data=val_ds,
                   epochs=self.config.EPOCHS,
                   callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.config.PATIENCE, restore_best_weights=True)]
                   )


        model.save(self.config.trained_model_dir)
        logger.info(f"Transfer Learning model saved at {self.config.trained_model_dir}")
        # CNN-SVM(Kernel)
        feature_extractor = tf.keras.models.load_model(self.config.feature_extract_dir)
        train_ds, val_ds = self._input_preparation_for_ML()
        x_train, y_train = self._feature_extraction(feature_extractor, train_ds)
        x_val, y_val = self._feature_extraction(feature_extractor, val_ds)
        # Standardize features
        scaler = StandardScaler()
        cls_svm = SVC(kernel='rbf', random_state=self.config.SEED,
                                              class_weight='balanced',
                                              decision_function_shape='ovr')
        pipe_svc = make_pipeline(scaler, cls_svm)
        pipe_svc.fit(x_train, y_train)
        metrics = self.report_f1(pipe_svc, x_train, y_train, x_val, y_val)
        logger.info(f"SVM with RBF kernel - Train F1 Macro: {metrics['train_f1_macro']:.4f}, Validation F1 Macro: {metrics['val_f1_macro']:.4f}, Overfitting Gap: {metrics['overfitting_gap']:.4f}")
        save_json(Path(self.config.root_dir) / "svm_metrics.json", metrics)
        save_object( self.config.trained_model_dir_svm, pipe_svc)
        logger.info(f"SVM model saved at {self.config.trained_model_dir_svm}")

        # CNN-PCA-SVC(Kernel)
        pca = PCA(n_components=0.9, random_state=self.config.SEED)
        pipe_pca_svc = make_pipeline(scaler, pca,cls_svm)
        pipe_pca_svc.fit(x_train, y_train)
        metrics_pca_svc = self.report_f1(pipe_pca_svc, x_train, y_train, x_val, y_val)
        logger.info(f"PCA + SVM with RBF kernel - Train F1 Macro: {metrics_pca_svc['train_f1_macro']:.4f}, Validation F1 Macro: {metrics_pca_svc['val_f1_macro']:.4f}, Overfitting Gap: {metrics_pca_svc['overfitting_gap']:.4f}")
        save_json(Path(self.config.root_dir) / "pca_svc_metrics.json", metrics_pca_svc)
        save_object( self.config.trained_model_dir_pca_svm, pipe_pca_svc)
        logger.info(f"PCA + SVM model saved at {self.config.trained_model_dir_pca_svm}")

        # CNN-KPCA-SVC(Kernel)
        kpca= KernelPCA(n_components=100, kernel='sigmoid', random_state=self.config.SEED)
        pipe_kpca_svc = make_pipeline(scaler, kpca, cls_svm)
        pipe_kpca_svc.fit(x_train, y_train)
        metrics_kpca_svc = self.report_f1(pipe_kpca_svc, x_train, y_train, x_val, y_val)
        logger.info(f"Kernel PCA + SVM with Sigmoid kernel - Train F1 Macro: {metrics_kpca_svc['train_f1_macro']:.4f}, Validation F1 Macro: {metrics_kpca_svc['val_f1_macro']:.4f}, Overfitting Gap: {metrics_kpca_svc['overfitting_gap']:.4f}")
        save_json(Path(self.config.root_dir) / "kpca_svc_metrics.json", metrics_kpca_svc)
        save_object( self.config.trained_model_dir_kpca_svm, pipe_kpca_svc)
        logger.info(f"Kernel PCA + SVM model saved at {self.config.trained_model_dir_kpca_svm}")






        
    
        

