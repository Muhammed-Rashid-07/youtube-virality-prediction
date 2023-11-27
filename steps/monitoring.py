import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import ClassificationPreset, TargetDriftPreset, DataQualityPreset
from evidently.report import Report
from zenml import step
from sklearn.base import ClassifierMixin
from mlflow.pyfunc import PyFuncModel
from zenml.services import BaseService

#from zenml.materializers.base_materializer import BaseMaterializer


class MonitoringPipeline():
    def __init__(self, reference_data, current_data, target, prediction, numerical_features, categorical_features):
        self.reference_data = reference_data
        self.current_data = current_data
        self.target = target
        self.prediction = prediction
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.column_mapping = self.create_column_mapping()

        self.classification_performance_report = Report(
            metrics=[ClassificationPreset()]
            )
        self.target_drift_report = Report(
            metrics=[TargetDriftPreset()]
            )
        self.data_quality_report = Report(metrics=[DataQualityPreset()])

    def create_column_mapping(self):
        column_mapping = ColumnMapping()
        column_mapping.target = self.target
        column_mapping.prediction = self.prediction
        column_mapping.numerical_features = self.numerical_features
        column_mapping.categorical_features = self.categorical_features
        return column_mapping

    def run_reports(self):
        self._run_classification_performance_report()
        self._run_data_quality_report()
        self._run_target_drift_report()
        

    def _run_classification_performance_report(self):
        self.classification_performance_report.run(
            current_data=self.reference_data,
            reference_data=None,
            column_mapping=self.column_mapping
        )
        self.classification_performance_report.save_html('reports/classification_performance_report.html')

    def _run_target_drift_report(self):
        self.target_drift_report.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
            column_mapping=self.column_mapping
        )
        self.target_drift_report.save_html('reports/target_drift_report.html')

    def _run_data_quality_report(self):
        self.data_quality_report.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
            column_mapping=self.column_mapping
        )
        self.data_quality_report.save_html('reports/data_quality_report.html')      

@step(enable_cache=False)
def model_monitoring(reference_data: pd.DataFrame, current_data: pd.DataFrame, model: ClassifierMixin | BaseService):
    """
    Perform model monitoring by generating and saving classification performance,
    target drift, and data quality reports.

    Args:
        reference_data (pd.DataFrame): DataFrame with reference data for monitoring.
        current_data (pd.DataFrame): DataFrame with current data for monitoring.
        model (ClassifierMixin | BaseService): Trained machine learning model.

    Returns:
        None
    """
    # Define target column and prediction column
    target_column = "is_viral"
    prediction = 'prediction'

    # Define numerical and categorical features used in the monitoring
    numerical_features = ['likes', 'dislikes', 'comment_count', 'time_since_publish', 'tag_count',
                          'like_dislike_ratio', 'comment_view_ratio', 'title_words_count', 'description_words_count']
    categorical_features = ['comments_disabled', 'ratings_disabled', 'video_error_or_removed']

    # Define the features used for predictions
    features = ['likes', 'dislikes', 'comment_count', 'comments_disabled', 'ratings_disabled',
                'video_error_or_removed', 'time_since_publish', 'tag_count', 'like_dislike_ratio',
                'comment_view_ratio', 'title_words_count', 'description_words_count']

    # Generate predictions for reference and current data
    ref_prediction = model.predict(reference_data[features])
    current_prediction = model.predict(current_data[features])

    # Add prediction columns to reference and current data
    reference_data['prediction'] = ref_prediction
    current_data['prediction'] = current_prediction

    # Create a MonitoringPipeline instance
    monitoring_pipeline = MonitoringPipeline(
        reference_data=reference_data,
        current_data=current_data,
        target=target_column,
        prediction=prediction,
        numerical_features=numerical_features,
        categorical_features=categorical_features
    )

    # Run and save monitoring reports
    monitoring_pipeline.run_reports()

