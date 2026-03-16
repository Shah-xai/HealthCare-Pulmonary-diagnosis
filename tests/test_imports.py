def test_package_imports():
    try:
        import CNN_Classifier
        import CNN_Classifier.components.data_ingestion
        import CNN_Classifier.components.base_model_preparation
        import CNN_Classifier.components.model_training
        import CNN_Classifier.components.model_evaluation
        import CNN_Classifier.pipeline.base_model_preparation_pipeline
        import CNN_Classifier.pipeline.model_training_pipeline
        import CNN_Classifier.pipeline.model_evaluation_pipeline
    except ImportError as e:
        raise ImportError(f"Import failed: {e}")