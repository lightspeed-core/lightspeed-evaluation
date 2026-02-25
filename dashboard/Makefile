LS_EVAL_SYSTEM_CFG_PATH ?= system.yaml
LS_EVAL_DATA_PATH ?= eval.yaml
LS_EVAL_REPORTS_PATH ?= results
API_KEY ?= $(shell oc whoami -t 2>/dev/null)

.PHONY: dev install

install:
	npm install

dev:
	LS_EVAL_SYSTEM_CFG_PATH=$(LS_EVAL_SYSTEM_CFG_PATH) LS_EVAL_DATA_PATH=$(LS_EVAL_DATA_PATH) LS_EVAL_REPORTS_PATH=$(LS_EVAL_REPORTS_PATH) API_KEY=$(API_KEY) npx vite
