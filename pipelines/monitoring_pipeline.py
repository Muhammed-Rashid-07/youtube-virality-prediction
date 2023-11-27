from steps.ingest_data import ingest_df
from steps.data_cleaning import cleaning_data
from steps.splitter import drift_splitting
from steps.evaluate import evaluate_model
from steps.model import test_model
from steps.model import test_model
from steps.monitoring import model_monitoring


from zenml import pipeline

@pipeline(enable_cache=False)
def monitoring_pipeline(report_name:list):
    raw_data = ingest_df()
    df = cleaning_data(df=raw_data)
    reference_data, current_data = drift_splitting(df)
    model = test_model(model_name='Best Model', stage='None')
    model_monitoring(reference_data=reference_data, current_data=current_data, model=model)
    