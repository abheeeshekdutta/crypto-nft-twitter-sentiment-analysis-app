#!/bin/bash

set -e

cd src

python feature_pipeline.py
python build_sentiment_predictions.py