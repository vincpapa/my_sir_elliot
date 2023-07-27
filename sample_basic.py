
import zipfile
import io
import requests
import os

from elliot.run import run_experiment

url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
print(f"Getting Movielens 1Million from : {url} ..")
response = requests.get(url)

ml_1m_ratings = []

print("Extracting ratings.dat ..")
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    for line in zip_ref.open("ml-1m/ratings.dat"):
        ml_1m_ratings.append(str(line, "utf-8").replace("::", "\t"))

print("Printing ratings.tsv to data/movielens1m/ ..")

os.makedirs("data/cat_dbpedia_movielens_1m_v030", exist_ok=True)
with open("data/cat_dbpedia_movielens_1m_v030/dataset.tsv", "w") as f:
    f.writelines(ml_1m_ratings)

print("Done! We are now starting the Elliot's experiment")
run_experiment("config_files/examples/basic_configuration_v030.yml")
