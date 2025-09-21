import streamlit as st
import os
import boto3
import torch
from transformers import pipeline

s3 = boto3.client('s3')
bucket_name = 'mlops-tutorial-models'

local_path = 'tinybert-sentiment-analysis'
s3_prefix = 'ml-models/tinybert-sentiment-analysis'

def download_dir(local_path, s3_prefix):
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key = key['Key']

                if s3_key.endswith("/"):
                    continue

                print(s3_key)
                local_file = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))
                # os.makedirs(os.path.dirname(local_file), exist_ok=True)

                s3.download_file(bucket_name, s3_key, local_file)

st.title('Machine Learning Model Deployment at the Server')
button = st.button('Download Model')
if button:
    with st.spinner('Downloading... Please wait'):
        download_dir(local_path, s3_prefix)

text = st.text_area('Enter Your Review')
predict = st.button('Predict')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if predict:
    with st.spinner('Predicting...'):
        classifier = pipeline('text-classification', model='tinybert-sentiment-analysis', device=device)
        output = classifier(text)

        st.write(output)
        st.info(output)
