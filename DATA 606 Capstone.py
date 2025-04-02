#!/usr/bin/env python
# coding: utf-8

# # Dietary Restriction AI

# ### Import Packages

# In[1]:


pip install langchain_community


# In[2]:


pip install pandas


# In[3]:


pip install pyspark


# In[36]:


from langchain_community.llms import Ollama
import os
import json
import pandas as pd 
import numpy as np
import tempfile 
from pyspark.sql import SparkSession

from pyspark.sql import functions as f
from sentence transformers import SentenceTransformer


# ### Initialize a Spark Session

# In[5]:


spark = SparkSession.builder \
    .appName("Local Spark") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()


# ### Data Ingestion
# In order to be compatible with LLM that we will be creating, the data needs to be processed to be in an efficient retrieval format and stored in a searchable index. 
# 
# ##### Recipe Data
# Our recipe data is sourced from web-scraped data containing

# In[26]:


all_files = os.listdir("./")
recipe_files = [file for file in all_files if "recipes_raw_nosource" in file]

df_list = []

for file_name in recipe_files: 
    temp_path = os.path.join(tempfile.gettempdir(), file_name)
    
    with open(file_name, "r", encoding="utf-8") as file:
        data = json.load(file)

    file_df = pd.DataFrame.from_dict(data, orient="index")
    df_list.append(file_df)  # Collect DataFrame
    
# Concatenate all dataframes
recipes_df = pd.concat(df_list)

# select only title, ingredient, instructions columns
recipes_df = recipes_df[['title', 'ingredients', 'instructions']]

# repartition the dataframe 
recipes_df = spark.createDataFrame(recipes_df)
recipes_df = recipes_df.repartition(100)

recipes_df.show()


# ##### Cooking Literature Data
# 
# The cooking literature data was pre-processed from PDF text files into a usable format in another notebook.

# In[18]:


cook_lit_files = [file for file in all_files if "chunked_data" in file]

cook_lit_df = pd.read_json(cook_lit_files[0])
cook_lit_df = spark.createDataFrame(cook_lit_df)
cook_lit_df.show()


# ### Data Chunking
# ##### Recipes Data
# The data was chunked into recipe-level chunks, since the recipes will then be able toi be referenced individually when needed. Since this use case is about modifying recipes in their entirety, we want the model to be able to reference the recipes in their entirety during its retrieval process. 

# In[38]:


recipes_df_chunk = recipes_df.withColumn("chunk_text", 
                                         f.concat_ws("\n", f.col("title"), f.col("ingredients"), f.col("instructions")))
recipes_df_chunk.show()


# ### Generate Embeddings

# In[ ]:


# load model and generate embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = np.array([embedding_model.encode(chunk["body"]) for chunk in chunked_data], dtype=np.float32)

# Store embeddings in chunked JSON
for i, chunk in enumerate(chunked_data):
    chunk["embedding"] = embeddings[i].tolist()
    
# Create and save FAISS index
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)


# ### Model Ingestion

# In[8]:


llm = Ollama(model="llama3.2")
print("Loaded Model")


# ##### Prompt Engineering
# The prompt inputted by the user should only need to contain the necessary recipe that the user wants to modify. The following prompt engineering code adds additional, consistent language that does the following: 
# - Specifies that the user wants to modify the recipe, retaining the original intention
# - Provides the dietary framework to stick to, in this case the high-protein low-carb diet. In another phase of development, this could be changed to xspecify a diet of choice
# - Requests a list of macronutrients based on the data

# In[9]:





# ##### RAG Component

# In[ ]:


def retrieve_relevant_chunks(query, k=5):
    """Retrieve top-k most relevant chunks using FAISS."""
    query_embedding = embedding_model.encode(query).reshape(1, -1)  # Convert query to embedding
    distances, indices = index.search(query_embedding, k)  # Retrieve top-k chunks

    return [chunked_data[i] for i in indices[0]]  # Get original text chunks

def query_ollama_with_context(query):
    """Retrieve relevant context and query Ollama 3.2."""
    retrieved_chunks = retrieve_relevant_chunks(query)
    context = "\n".join([chunk["body"] for chunk in retrieved_chunks])  # Combine relevant chunks

    # Formulate prompt for LLaMA
    prompt = f"Context:\n{context}\n\nQuery: {query}\nAnswer:"

    # Query Ollama
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

if __name__ == "__main__":
    query = input("Enter your recipe: ")
    query += " Modify this recipe so that it is more suited for a high-protein, low carb diet. Provide a list of macronutrients as a part of the analysis.
    answer = query_ollama_with_context(query)
    print("\nOllama's Answer:", answer)


# ### Recipe Evaluator

# In[2]:


def evaluate_recipe(protein_g, fat_g, carb_g):
    # Caloric values per gram
    PROTEIN_CAL = 4
    CARB_CAL = 4
    FAT_CAL = 9
    
    # Calculate total calories
    total_calories = (protein_g * PROTEIN_CAL) + (fat_g * FAT_CAL) + (carb_g * CARB_CAL)
    
    if total_calories == 0:
        return "Invalid recipe: Total calories cannot be zero."
    
    # Calculate macronutrient percentage
    protein_pct = (protein_g * PROTEIN_CAL / total_calories) * 100
    fat_pct = (fat_g * FAT_CAL / total_calories) * 100
    carb_pct = (carb_g * CARB_CAL / total_calories) * 100
    
    # Define healthy ranges
    protein_range = (10, 30)
    fat_range = (20, 35)
    carb_range = (45, 65)
    
    # Check if recipe meets healthy criteria
    if (protein_range[0] <= protein_pct <= protein_range[1] and
        fat_range[0] <= fat_pct <= fat_range[1] and
        carb_range[0] <= carb_pct <= carb_range[1]):
        return "Meets Criteria"
    else:
        return "Does Not Meet Criteria"
# Example usage
recipe_result = evaluate_recipe(protein_g=3, fat_g=20, carb_g=100)
print(recipe_result)


# In[ ]:




