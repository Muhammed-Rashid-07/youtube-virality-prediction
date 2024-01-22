import json
import logging
from typing import Any, Dict, List, Optional, Union
from typing_extensions import Annotated
from pipelines.deployment_pipeline import deploy_and_predict    
from run_pipeline import main as run_pipeline
import pandas as pd
import streamlit as st
from steps.prediction_service_loader import prediction_service_loader
import datetime
import numpy as np
from steps.monitoring import update_monitoring
from steps.splitter import drift_splitting
from steps.load_test_data import load_test_data
import logging

import pandas as pd
from steps.ingest_data import ingest_df
from steps.data_cleaning import cleaning_data   




def prediction():
    st.title("Youtube Virality Prediction")
    st.markdown(
    """ 
    #### Problem Statement 
     The objective here is to predict the whether a youtube video will become viral or not based on the views, likes , dislikes ,comments, title length etc.   """
    )
    
    df = load_test_data()
    
    reference_data, current_data = drift_splitting(df)
    
    trending_date = st.date_input("Trending Date of the video: ", format="YYYY/MM/DD")
    video_title = st.text_input("Title of the video: ")
    likes = st.number_input("Number of likes: ", value=None)
    views = st.number_input("Number of views: ", value=None)
    dislikes = st.number_input("Number of dislikes: ", value=None)
    comments = st.number_input("Number of comments: ", value=None)
    tags = st.number_input("Number of tags: ", value=None)
    comments_disabled = st.radio("Comments Disabled: ", options=[True, False])
    rating_disabled = st.radio("Ratings Disabled: ", options=[True, False])
    video_error_removed = st.radio("Video Error or Removed: ", options=[True, False])
    description = st.text_input("Description of the video: ")
    publish_date = st.date_input("Publish Date of the video: ", format="YYYY/MM/DD")
    
    trending_dt = datetime.datetime.combine(trending_date, datetime.datetime.min.time())
    publishing_date = datetime.datetime.combine(publish_date, datetime.datetime.min.time())
    time_since_publish = (trending_dt - publishing_date).days
    
    
    
    
    if st.button("Predict"):
        service = prediction_service_loader('train_pipeline')
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            run_main()
        try:
            data = {
                "likes": likes,
                "dislikes": dislikes,
                "comment_count": comments,
                "comments_disabled": comments_disabled,
                "ratings_disabled": rating_disabled,
                "video_error_or_removed": video_error_removed,
                "tag_count": tags,
                "like_dislike_ratio": likes / (dislikes + 1),
                "comment_view_ratio": comments / (views + 1),
                "title_words_count": len(video_title.split()),
                "description_words_count": len(description.split()),
                "time_since_publish": time_since_publish,
            }

            # Convert the data point to a DataFrame
            data_point_df = pd.DataFrame(data, index=[0])

            # Predict and add prediction column
            pred = service.predict(data_point_df)
            data_point_df['prediction'] = pred

            # Update the reports based on the current prediction
            update_monitoring(reference_data, current_data, data_point_df, service)

            st.success(f"Youtube Virality prediction: {'Viral' if pred == 1 else 'Not Viral'}")
        except Exception as e:
            logging.error(e)
            raise e




def display_report(report_path: str):
    """Displays an Evidently HTML report in Streamlit.

    Args:
        report_path: File path of the Evidently HTML report.
    """   
    with open(report_path) as report_file:
        report_html = report_file.read()

    st.components.v1.html(report_html, height=1000, scrolling=True)
    

def run_reports():
    generate_quality_report = st.checkbox("Generate Data Quality Report")
    generate_performance_report = st.checkbox("Generate Performance Report")
    generate_target_drift_report = st.checkbox("Generate Drift Report")
    
    selected_reports = []
    if generate_quality_report:
        selected_reports.append("data_quality_report")
    if generate_performance_report:
        selected_reports.append("classification_performance_report")
    if generate_target_drift_report:  
        selected_reports.append("target_drift_report")

        
    if 'data_quality_report' in selected_reports:
        display_report("reports/data_quality_report.html")
    if 'classification_performance_report' in selected_reports:
        display_report("reports/classification_performance_report.html")
    if 'target_drift_report' in selected_reports:
        display_report("reports/target_drift_report.html")
        
        
def main():
    st.sidebar.title("Options")
    selection = st.sidebar.radio("Select an option", ["Prediction", "Reports"])
    if selection == "Prediction":
        prediction()
    if selection == "Reports":
        run_reports()
if __name__ == "__main__":
    main()