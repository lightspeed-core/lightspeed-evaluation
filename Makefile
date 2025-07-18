# Put targets here if there is a risk that a target name might conflict with a filename.
# this list is probably overkill right now.
# See: https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html
.PHONY: format verify

install-tools: ## Install required utilities/tools
	# OLS 1085: Service build failure issue caused by newest PDM version
	# (right now we need to stick to PDM specified in pyproject.toml file)
	@command -v pdm > /dev/null || { echo >&2 "pdm is not installed. Installing..."; pip install pdm; }
	pdm --version
	# this is quick fix for OLS-758: "Verify" CI job is broken after new Mypy 1.10.1 was released 2 days ago
	# CI job configuration would need to be updated in follow-up task
	# pip uninstall -v -y mypy 2> /dev/null || true
	# display setuptools version
	pip show setuptools
	export PIP_DEFAULT_TIMEOUT=100
	# install all dependencies, including devel ones
	@for a in 1 2 3 4 5; do pdm install --group default,dev --fail-fast -v && break || sleep 15; done
	# check that correct mypy version is installed
	# mypy --version
	pdm run mypy --version
	# check that correct Black version is installed
	pdm run black --version
	# check that correct Ruff version is installed
	pdm run ruff --version
	# check that Pytest is installed
	pdm run pytest --version

pdm-lock-check: ## Check that the pdm.lock file is in a good shape
	pdm lock --check

install-deps: install-tools pdm-lock-check ## Install all required dependencies needed to run the service, according to pdm.lock
	@for a in 1 2 3 4 5; do pdm sync && break || sleep 15; done

install-deps-test: install-tools pdm-lock-check ## Install all required dev dependencies needed to test the service, according to pdm.lock
	@for a in 1 2 3 4 5; do pdm sync --dev && break || sleep 15; done

update-deps: ## Check pyproject.toml for changes, update the lock file if needed, then sync.
	pdm install
	pdm install --dev

check-types: ## Checks type hints in sources
	pdm run mypy --explicit-package-bases --disallow-untyped-calls --disallow-untyped-defs --disallow-incomplete-defs src/ lsc_agent_eval/src/

format: install-deps-test ## Format the code into unified format
	pdm run black .
	pdm run ruff check . --fix --per-file-ignores=tests/*:S101 --per-file-ignores=scripts/*:S101 --per-file-ignores=lsc_agent_eval/tests/*:S101

verify:	install-deps-test ## Verify the code using various linters
	pdm run black . --check
	pdm run ruff check . --per-file-ignores=tests/*:S101 --per-file-ignores=scripts/*:S101 --per-file-ignores=lsc_agent_eval/tests/*:S101
	pdm run pylint src tests lsc_agent_eval/src lsc_agent_eval/tests

requirements.txt:	pyproject.toml pdm.lock ## Generate requirements.txt file containing hashes for all non-devel packages
	pdm export --prod --format requirements --output requirements.txt

verify-packages-completeness:	requirements.txt ## Verify that requirements.txt file contains complete list of packages
	pip download -d /tmp/ --use-pep517 --verbose -r requirements.txt

distribution-archives: ## Generate distribution archives to be uploaded into Python registry
	pdm run python -m build

test: install-deps-test ## Execute tests with Pytest
	pdm run pytest tests lsc_agent_eval/tests

help: ## Show this help screen
	@echo 'Usage: make <OPTIONS> ... <TARGETS>'
	@echo ''
	@echo 'Available targets are:'
	@echo ''
	@grep -E '^[ a-zA-Z0-9_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-33s\033[0m %s\n", $$1, $$2}'
	@echo ''

pylint:
	pdm run pylint src lsc_agent_eval/src

pyright:
	pdm run pyright src lsc_agent_eval/src

docstyle:
	pdm run pydocstyle -v .

ruff:
	pdm run ruff check . --per-file-ignores=tests/*:S101 --per-file-ignores=scripts/*:S101 --per-file-ignores=lsc_agent_eval/tests/*:S101
