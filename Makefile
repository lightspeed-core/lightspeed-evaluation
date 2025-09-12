# Put targets here if there is a risk that a target name might conflict with a filename.
# this list is probably overkill right now.
# See: https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html
.PHONY: format verify

install-tools: ## Install required utilities/tools
	# Install uv if not already installed
	@command -v uv > /dev/null || { echo >&2 "uv is not installed. Installing..."; pip install uv; }
	uv --version
	# this is quick fix for OLS-758: "Verify" CI job is broken after new Mypy 1.10.1 was released 2 days ago
	# CI job configuration would need to be updated in follow-up task
	# pip uninstall -v -y mypy 2> /dev/null || true
	# display setuptools version
	pip show setuptools
	export PIP_DEFAULT_TIMEOUT=100
	# install all dependencies, including devel ones
	@for a in 1 2 3 4 5; do uv sync --group dev && break || sleep 15; done
	# check that correct mypy version is installed
	uv run mypy --version
	# check that correct Black version is installed
	uv run black --version
	# check that correct Ruff version is installed
	uv run ruff --version
	# check that Pytest is installed
	uv run pytest --version

uv-lock-check: ## Check that the uv.lock file is in a good shape
	uv lock --check

install-deps: install-tools uv-lock-check ## Install all required dependencies needed to run the service, according to uv.lock
	@for a in 1 2 3 4 5; do uv sync && break || sleep 15; done

install-deps-test: install-tools uv-lock-check ## Install all required dev dependencies needed to test the service, according to uv.lock
	@for a in 1 2 3 4 5; do uv sync --group dev && break || sleep 15; done

update-deps: ## Check pyproject.toml for changes, update the lock file if needed, then sync.
	uv sync
	uv sync --group dev

check-types: ## Checks type hints in sources
	uv run mypy --explicit-package-bases --disallow-untyped-calls --disallow-untyped-defs --disallow-incomplete-defs src/ lsc_agent_eval/src/

format: install-deps-test ## Format the code into unified format
	uv run black .
	uv run ruff check . --fix --per-file-ignores=tests/*:S101 --per-file-ignores=scripts/*:S101 --per-file-ignores=lsc_agent_eval/tests/*:S101

verify:	install-deps-test ## Verify the code using various linters
	uv run black . --check
	uv run ruff check . --per-file-ignores=tests/*:S101 --per-file-ignores=scripts/*:S101 --per-file-ignores=lsc_agent_eval/tests/*:S101
	uv run pylint src tests lsc_agent_eval/src lsc_agent_eval/tests

requirements.txt:	pyproject.toml uv.lock ## Generate requirements.txt file containing hashes for all non-devel packages
	uv export --no-dev --format requirements-txt --output-file requirements.txt

verify-packages-completeness:	requirements.txt ## Verify that requirements.txt file contains complete list of packages
	pip download -d /tmp/ --use-pep517 --verbose -r requirements.txt

distribution-archives: ## Generate distribution archives to be uploaded into Python registry
	uv run python -m build

test: install-deps-test ## Execute tests with Pytest
	uv run pytest tests lsc_agent_eval/tests

help: ## Show this help screen
	@echo 'Usage: make <OPTIONS> ... <TARGETS>'
	@echo ''
	@echo 'Available targets are:'
	@echo ''
	@grep -E '^[ a-zA-Z0-9_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-33s\033[0m %s\n", $$1, $$2}'
	@echo ''

pylint:
	uv run pylint src
	uv run pylint lsc_agent_eval/src

pyright:
	uv run pyright src lsc_agent_eval/src

docstyle:
	uv run pydocstyle -v .

ruff:
	uv run ruff check . --per-file-ignores=tests/*:S101 --per-file-ignores=scripts/*:S101 --per-file-ignores=lsc_agent_eval/tests/*:S101
