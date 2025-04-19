## Overview

This application integrates the Spider Agent from Spider 2.0 dataset with Streamlit to provide a user-friendly interface for generating SQL queries from natural language questions. The Spider Agent comes from the Spider 2.0 dataset, which is a large-scale benchmark for evaluating text-to-SQL systems. This application allows users to interact with the Spider Agent easily to inspect the agent's step-by-step reasoning and the generated SQL queries.

## Quickstart

1. **Install Docker**. Follow the instructions in the [Docker setup guide](https://docs.docker.com/engine/install/) to install Docker on your machine. 
2. **Install conda environment**.
```
git clone https://github.com/averrie/TEXT2SQL
cd TEXT2SQL

# Optional: Create a Conda environment for Spider 2.0
# conda create -n spider2 python=3.11
# conda activate spider2

# Install required dependencies
pip install -r requirements.txt


# Optional: Modify spider_agent/agent/models.py to add your own LLM API key and inference URL.
```

3. **Run Streamlit app**. 
```
streamlit run app.py
```