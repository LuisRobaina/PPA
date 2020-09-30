init:
	pip install -r requirements.txt
train-model:
	python3 -m PPA.PPA_Learn
test-model:
	python3 -m PPA.PPA_Test

.PHONY: init train-model test-model
