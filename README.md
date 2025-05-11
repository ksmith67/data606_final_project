# data606_final_project

This repository is for the files necessary to create the AI-Driven Culinary Innovation project developed by Jennifer Giolitti and Kevin Smith
for the UMBC DATA 606 Course

This project is our final capstone project for the Masters in Professional Studies in Data Science course. Our project is a Large Language Model 
(LLM) that accepts a recipe as inout, then uses a RAG framework to modify these recipes to produce a recipe that meets set criteira for 
a high protein, low carbohydrate diet. 

The project is dependent on a separate use of Ollama's deployment of Llama 3.2 3B to run locally. It is recommended you have at least 16GB of RAM
to run this project, and the processor used to develop it was an M4 Apple processor. Ollama must be deployed separately bia the terminal to be used. Ollama can be downloaded here: https://ollama.com/download

The project was further developed in two versions - a version that runs using PySpark for distributed computing and another version that runs 
using FireDucks' Pandas module, which is designed to be faster than regular pandas. The FireDucks version is more stable on our local development
so it is recommended that be used for local evaluation/deployment, but the distributed notebook was left in the repository for future use if 
those resources were ever to become available. 

Local Deployment Notebook: DATA 606 Capstone Fireducks (2).ipynb
Distributed Deployment Notebook: Data 606 Capstone.ipynb

The project was determined to be very successful as a preliminary prototype, and the results_data.csv file contains results for 3896 test cases. 
It took over 80 hours of continuous running to get these results alone, so attempt more at your own risk! 

Thank you for looking at our project! 
