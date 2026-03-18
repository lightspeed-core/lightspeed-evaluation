LS_EVAL_SYSTEM_CFG_PATH ?= ./dashboard/system.yaml
LS_EVAL_DATA_PATH ?= ./dashboard/eval.yaml
LS_EVAL_REPORTS_PATH ?= ./dashboard/results
LS_EVAL_DASHBOARD_RUN_ENABLED ?= true
API_KEY ?= $(shell oc whoami -t 2>/dev/null)

.PHONY: dev install build_image

install:
	npm install

dev:
	@LS_EVAL_SYSTEM_CFG_PATH="$(LS_EVAL_SYSTEM_CFG_PATH)" LS_EVAL_DATA_PATH="$(LS_EVAL_DATA_PATH)" LS_EVAL_REPORTS_PATH="$(LS_EVAL_REPORTS_PATH)" LS_EVAL_DASHBOARD_RUN_ENABLED="$(LS_EVAL_DASHBOARD_RUN_ENABLED)" API_KEY="$(API_KEY)" npx vite

build_image:
	@./build.sh $(DOCKER_IMAGE_NAME) $(DOCKER_IMAGE_TAG)
