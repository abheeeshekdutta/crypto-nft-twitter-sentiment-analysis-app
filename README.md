# Twitter sentiment analysis app for cryptocurrencies and NFTs

The objective of this application is to display an interactive dashboard which demonstrates the trend of sentiments around cryptocurrency and NFT tweets on Twitter.

# 🛠️ Tools used

The following tools were used to build the application

1. Python - programming language used for building majority of the application
2. [Streamlit](https://streamlit.io) - to develop the application user interface
3. [Hopsworks Feature Store](https://www.hopsworks.ai) - feature store for storing and retrieving tweet data
4. [RapidAPI](https://rapidapi.com/Glavier/api/twitter135/) - API for fetching new tweet data
5. [HuggingFace Hub](https://huggingface.co/models) - for utilizing the [pre-trained sentiment analysis model](https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis)
6. [Github Actions](https://github.com/features/actions) - for implementing script automation

# ⿸ Application architecture

![Project Architecture](https://github.com/abheeeshekdutta/crypto-nft-twitter-sentiment-analysis-app/blob/main/assets/project_architecture.png)


# ？How it works

1. Data is fetched from the twitter API hosted on RapidAPI - the tweet pull schedule is everyday at 10:00AM Irish Standard Time(IST)

2. The raw tweet data from the above step is pushed to a Hopsworks Feature Store

3. The data is then run through a sentiment analysis pipeline and the sentiment predictions and scores are produced, which are again stored in the Hopsworks Feature Store.

4. The web application which is hosted on Streamlit Cloud, fetches the prediction data from the Hopsworks Feature Store, and it used to plot graphs and display results.
