# M2DET Object Detection Model Deployment using AWS API gateway

### This repository contains code to create a detection API using a model trained on coco dataset. 

### How to create a public API using this repo ?
1. Please put the code in this repository in a docker container
2. Push the container to AWS ECR
3. Create a model on AWS Sageamker using the container in ECR 
3. Create an endpoint-configuration and endpoint using the model
4. Create a lamda function to send the parameters of request (Image and detection confidence thershold etc.) to the sagemaker endpoint
5. Create API using API gateway to invoke the lamda function to address client side post requests 

### To perform inference
#### Please edit the url in post.py script to the api url and run post.py 
