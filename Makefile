.PHONY: all clean

MODEL_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

all: data/raw/models.csv

clean:
	rm -rf data/interim/*

data/raw/models.csv:
	python src/data/download.py $(MODEL_URL) $@