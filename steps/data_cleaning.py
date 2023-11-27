import logging
import pandas as pd
from zenml import step
from typing_extensions import Union

class DataPreprocessing:
    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame | pd.Series:
        """
        Preprocesses the input DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
        Returns:
            Union[pd.DataFrame, pd.Series]: Preprocessed DataFrame.
        """
        # Remove duplicate rows
        df.drop_duplicates(inplace=True)

        # Convert 'trending_date' to datetime format
        df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')

        # Convert 'publish_time' to datetime format
        df['publish_time'] = pd.to_datetime(df['publish_time'])

        # Convert 'trending_date' and 'publish_time' to tz-naive datetime objects
        df['trending_date'] = df['trending_date'].dt.tz_localize(None)
        df['publish_time'] = df['publish_time'].dt.tz_localize(None)

        # Calculate time since publication
        df['time_since_publish'] = (df['trending_date'] - df['publish_time']).dt.days 

        # Convert 'tags' column to strings
        df['tags'] = df['tags'].astype(str)

        # Tokenize and preprocess the tags
        df['tag_count'] = df['tags'].apply(lambda x: len(x.split('|')))
        df['tags'] = df['tags'].str.lower().str.split('|')

        # Engagement metrics
        df['like_dislike_ratio'] = df['likes'] / (df['dislikes'] + 1)
        df['comment_view_ratio'] = df['comment_count'] / (df['views'] + 1)
        df = df.fillna(0)
        threshold = 100000

        # Ensure 'title' column contains strings
        df['title'] = df['title'].astype(str)

        # Count words in 'title'
        df['title_words_count'] = df['title'].apply(lambda x: len(x.split()))

        # Convert 'description' column to strings
        df['description'] = df['description'].astype(str)

        # Count words in 'description'
        df['description_words_count'] = df['description'].apply(lambda x: len(x.split()))

        # Create 'is_viral' column based on views and time_since_publish
        df['is_viral'] = ((df['views'] > threshold) & (df['time_since_publish'] < 10)).astype(int)

        # Drop unnecessary columns
        df = df.drop(['title', 'video_id', 'publish_time', 'views', 'trending_date', 'thumbnail_link', 
                      'category_id', 'channel_title', 'tags', 'description'], axis=1)

        return df


@step(enable_cache=True)
def cleaning_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step for cleaning data by using the DataPreprocessing class.

    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    try:
        # Instantiate the DataPreprocessing class
        data_preprocessing = DataPreprocessing()
        
        # Perform data cleaning
        data = data_preprocessing.handle_data(df=df)
        
        # Log the column names of the cleaned DataFrame
        logging.info(data.columns)
        
        return data
    except Exception as e:
        # Log an error message if an exception occurs
        logging.error("Error in cleaning data")
        raise e
