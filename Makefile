.PHONY: style check_code_quality

PYTHON=python
export PYTHONPATH = .
check_dirs := inference inference_sdk

style:
	python3 -m black $(check_dirs) --exclude '__init__\.py|node_modules|perception_encoder/vision_encoder/'
	python3 -m isort $(check_dirs) --skip-glob '**/__init__.py' --skip-glob '**/node_modules/**' --skip-glob '**/perception_encoder/vision_encoder/**'

check_code_quality:
	python3 -m black --check $(check_dirs) --exclude '__init__\.py|node_modules|perception_encoder/vision_encoder/'
	python3 -m isort --check-only $(check_dirs) --skip-glob '**/__init__.py' --skip-glob '**/node_modules/**' --skip-glob '**/perception_encoder/vision_encoder/**'
	# stop the build if there are Python syntax errors or undefined names
	flake8 $(check_dirs) --count --select=E9,F63,F7,F82 --show-source --statistics --exclude __init__.py,inference/inference/landing/node_modules
	# exit-zero treats all errors as warnings. E203 for black, E501 for docstring, W503 for line breaks before logical operators 
	flake8 $(check_dirs) --count --max-line-length=88 --exit-zero  --ignore=D --extend-ignore=E203,E501,W503  --statistics --exclude __init__.py,inference/inference/landing/node_modules


start_test_docker_cpu:
	docker run -d --rm -p $(PORT):$(PORT) -e USE_INFERENCE_MODELS=$(USE_INFERENCE_MODELS) -e PORT=$(PORT) -e MAX_BATCH_SIZE=17 --name inference-test roboflow/${INFERENCE_SERVER_REPO}:test

start_test_docker_gpu:
	docker run -d --rm -p $(PORT):$(PORT) -e USE_INFERENCE_MODELS=$(USE_INFERENCE_MODELS) -e PORT=$(PORT) -e MAX_BATCH_SIZE=17 --gpus=all --name inference-test roboflow/${INFERENCE_SERVER_REPO}:test

start_test_docker_gpu_with_roboflow_staging:
	docker run -d --rm -p $(PORT):$(PORT) -e PORT=$(PORT) -e MAX_BATCH_SIZE=17 --gpus=all -e PROJECT=roboflow-staging --name inference-test roboflow/${INFERENCE_SERVER_REPO}:test


start_test_docker_jetson:
	docker run -d --rm -p $(PORT):$(PORT) -e PORT=$(PORT) -e MAX_ACTIVE_MODELS=1 -e MAX_BATCH_SIZE=17 --runtime=nvidia --name inference-test roboflow/${INFERENCE_SERVER_REPO}:test

stop_test_docker:
	docker rm -f inference-test

create_wheels:
	python -m pip install --upgrade pip
	python -m pip install wheel twine requests -r requirements/_requirements.txt -r requirements/requirements.cpu.txt -r requirements/requirements.http.txt -r requirements/requirements.sdk.http.txt
	rm -f dist/*
	rm -rf build/*
	python .release/pypi/inference.core.setup.py bdist_wheel
	rm -rf build/*
	python .release/pypi/inference.cpu.setup.py bdist_wheel
	rm -rf build/*
	python .release/pypi/inference.gpu.setup.py bdist_wheel
	rm -rf build/*
	python .release/pypi/inference.setup.py bdist_wheel
	rm -rf build/*
	python .release/pypi/inference.sdk.setup.py bdist_wheel
	rm -rf build/*
	python .release/pypi/inference.cli.setup.py bdist_wheel

create_wheels_for_gpu_notebook:
	python -m pip install --upgrade pip
	python -m pip install wheel twine requests
	rm -f dist/*
	python .release/pypi/inference.core.setup.py bdist_wheel
	python .release/pypi/inference.gpu.setup.py bdist_wheel
	python .release/pypi/inference.sdk.setup.py bdist_wheel
	python .release/pypi/inference.cli.setup.py bdist_wheel

create_inference_cli_whl:
	${PYTHON} -m pip install --upgrade pip
	${PYTHON} -m pip install wheel twine requests
	rm -f dist/*
	${PYTHON} .release/pypi/inference.cli.setup.py bdist_wheel


upload_wheels:
	twine upload dist/*.whl
