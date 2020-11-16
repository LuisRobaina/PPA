init:
	pip3 install -r requirements.txt
train-model:
	python3 -m PPA.PPA_Learn
test-model:
	python3 -m PPA.PPA_Test
evaluate-set:
	python3 -m PPA.VerifyTrainingSet

.PHONY: init train-model test-model
