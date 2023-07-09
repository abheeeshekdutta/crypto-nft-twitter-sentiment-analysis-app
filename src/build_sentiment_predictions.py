from transformers import pipeline
import pandas as pd
import torch
from dotenv import load_dotenv, find_dotenv, dotenv_values
import hopsworks
import numpy as np


def fetch_raw_data_from_hopsworks_fs(hopsworks_api_key: str) -> pd.DataFrame:
    """
    This function fetches raw tweets data from the Hopsworks feature store

    Args:
        hopsworks_api_key (str): Hopsworks API key

    Returns:
        pd.DataFrame: Tweets raw data fetched from Hopsworks feature store
    """

    project = hopsworks.login(api_key_value=hopsworks_api_key)
    fs = project.get_feature_store()

    try:
        feature_view = fs.get_feature_view(name="nft_crypto_tweets_raw_view", version=2)
        df = feature_view.get_batch_data()
    except:
        iris_fg = fs.get_feature_group(name="nft_crypto_tweets_raw", version=2)
        query = iris_fg.select_all()
        feature_view = fs.create_feature_view(
            name="nft_crypto_tweets_raw_view",
            version=2,
            description="Read from Tweets dataset",
            labels=["dummy_target"],
            query=query,
        )
        df = feature_view.get_batch_data()

    return df


def extract_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        tweets (pd.Series): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df["sentiment"] = df["tweet_text"].apply(
        lambda x: sentiment_analysis(x)[0]["label"]
    )
    df["sentiment_score"] = df["tweet_text"].apply(
        lambda x: sentiment_analysis(x)[0]["score"]
    )

    return df


def push_predictions_fg(tweets_df_predictions: pd.DataFrame):
    """
    This function pushes sentiment predictions of tweets data to the Hopsworks feature group

    Args:
        tweets_df (pd.Dataframe): Tweets dataframe that needs to be pushed to Hopsworks feature store
    """
    hopsworks_api_key = env_vars["HOPSWORKS_API_KEY"]
    project = hopsworks.login(api_key_value=hopsworks_api_key)
    fs = project.get_feature_store()

    tweets_df_predictions["dummy_target"] = np.random.randint(
        0, 100, size=len(tweets_df_predictions)
    )

    tweets_predictions_fg = fs.get_or_create_feature_group(
        name="nft_crypto_tweets_sentiment_predictions",
        version=1,
        primary_key=["tweet_id"],
        description="Feature group for storing sentiment predictions of raw tweets data",
    )
    tweets_predictions_fg.insert(
        features=tweets_df_predictions, overwrite=False, operation="upsert"
    )


if __name__ == "__main__":
    # Fetch variables for .env file
    print("Fetching API keys...")
    env_vars = dotenv_values(find_dotenv())
    hopsworks_api_key = env_vars["HOPSWORKS_API_KEY"]

    print("Fetching raw data...")
    df = fetch_raw_data_from_hopsworks_fs(hopsworks_api_key=hopsworks_api_key)

    # Set up the inference pipeline using a model from the 🤗 Hub
    sentiment_analysis = pipeline(
        model="finiteautomata/bertweet-base-sentiment-analysis"
    )

    # Extract sentiment and sentiment scores
    print("Extracting sentiment and sentiment scores...")
    tweets_df_predictions = extract_sentiment(df)

    # Push predictions to feature group
    push_predictions_fg(tweets_df_predictions)
