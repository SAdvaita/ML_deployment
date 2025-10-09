install:
	pip install --upgrade pip
	pip install -r requirements.txt

train:
	python3 train.py

eval:
	echo "## Model Metrics" > report.md
	cat results/metrics.txt >> report.md
	echo "\n## Confusion Matrix Plot" >> report.md
	echo "![Confusion Matrix](./results/model_results.png)" >> report.md
	cml comment create report.md

update-branch:
	git config --global user.name "SAdvaita"
	git config --global user.email "adhu1926@gmail.com"
	git add model results
	git commit -m "Update model and results"
	git push --force origin HEAD:update

hf-login:
	git pull origin update || true
	git switch update || true
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF_TOKEN) --add-to-git-credential

push-hub:
	huggingface-cli upload DRGJ2025/DRUG_CLASSIFY ./app --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload DRGJ2025/DRUG_CLASSIFY ./model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload DRGJ2025/DRUG_CLASSIFY ./results --repo-type=space --commit-message="Sync Results"

deploy:
	make hf-login
	make push-hub
