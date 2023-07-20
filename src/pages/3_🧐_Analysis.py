import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import hopsworks
from dotenv import load_dotenv, find_dotenv, dotenv_values


@st.cache_data(show_spinner=False)
def fetch_sentiment_data_from_hopsworks_fs() -> pd.DataFrame:
    """
    This function fetches the sentiment data for our tweets from the Hopsworks feature store

    Returns:
        pd.DataFrame: Predictions data
    """

    # Fetch variables for .env file
    print("Fetching API keys...")
    env_vars = dotenv_values(find_dotenv())
    hopsworks_api_key = env_vars["HOPSWORKS_API_KEY"]

    print("Fetching predictions data...")

    project = hopsworks.login(api_key_value=hopsworks_api_key)
    fs = project.get_feature_store()

    try:
        feature_view = fs.get_feature_view(
            name="nft_crypto_tweets_sentiments_view", version=1
        )
        df = feature_view.get_batch_data()
    except:
        iris_fg = fs.get_feature_group(
            name="nft_crypto_tweets_sentiment_predictions", version=1
        )
        query = iris_fg.select_all()
        feature_view = fs.create_feature_view(
            name="nft_crypto_tweets_sentiments_view",
            version=1,
            description="Read from Tweets sentiment dataset",
            labels=["dummy_target"],
            query=query,
        )
        df = feature_view.get_batch_data()
    return df


with st.spinner("Fetching sample prediction data from Hopsworks..."):
    df = fetch_sentiment_data_from_hopsworks_fs()


# Function to filter and resample the DataFrame based on user selection
def filter_tweets_data(data, duration, start_date=None, end_date=None):
    if duration == "Last 24 hours":
        filtered_data = data[
            data["tweet_date"] >= (pd.to_datetime("now") - pd.DateOffset(days=1))
        ]
        resampled_data = filtered_data.resample("1H", on="tweet_date").mean()
    elif duration == "Last Week":
        filtered_data = data[
            data["tweet_date"] >= (pd.to_datetime("now") - pd.DateOffset(weeks=1))
        ]
        resampled_data = filtered_data.resample("D", on="tweet_date").mean()
    else:
        filtered_data = data[
            (data["tweet_date"] >= start_date) & (data["tweet_date"] <= end_date)
        ]
        resampled_data = filtered_data.resample("D", on="tweet_date").mean()
    return resampled_data, filtered_data


# Streamlit app code
def main():
    st.header("Sentiment Analysis of Tweets")

    with st.spinner("Fetching sample prediction data from Hopsworks..."):
        tweets_df = fetch_sentiment_data_from_hopsworks_fs()
    tweets_df["tweet_id"] = tweets_df["tweet_id"].astype("str")

    # User input for time duration
    duration = st.radio(
        "Select Time Duration:", ["Last 24 hours", "Last Week", "Custom"]
    )

    if duration == "Custom":
        # Custom date range input
        start_date = st.date_input(
            "Start Date", pd.to_datetime("now") - pd.DateOffset(days=7)
        )
        end_date = st.date_input("End Date", pd.to_datetime("now"))

        # Filter and resample DataFrame based on custom date range
        resampled_df, filtered_df = filter_tweets_data(
            tweets_df, duration, start_date, end_date
        )
    else:
        # Filter and resample DataFrame based on selected duration
        resampled_df, filtered_df = filter_tweets_data(tweets_df, duration)

    if not resampled_df.empty:
        # Line plot of sentiment scores
        fig_sentiment_score = px.line(
            resampled_df,
            x=resampled_df.index,
            y="sentiment_score",
            title=f"Sentiment Scores for {duration}",
            labels={"Sentiment_Score": "Sentiment Score"},
        )
        st.plotly_chart(fig_sentiment_score)

        # Bar chart of sentiment categories
        fig_sentiment_category = px.bar(
            filtered_df,
            x="sentiment",
            title="Sentiment Categories",
            labels={"Sentiment": "Sentiment Category", "count": "Count"},
        )
        st.plotly_chart(fig_sentiment_category)


if __name__ == "__main__":
    main()
