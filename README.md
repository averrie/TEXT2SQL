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