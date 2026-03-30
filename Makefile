# Put targets here if there is a risk that a target name might conflict with a filename.
# this list is probably overkill right now.
# See: https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html
.PHONY: help test

git-hooks:  ## Install git hooks
	@echo "Installing git hooks"
	cd .git/hooks && ln -sf ../../githooks/* ./

install-tools: git-hooks  ## Install required utilities/tools
	@command -v uv > /dev/null || { echo >&2 "uv is not installed. Installing..."; pip install uv; }
	@uv --version

uv-lock-check: ## Check that the uv.lock file is in a good shape
	uv lock --check


install-deps-test: ## Install all required dev dependencies needed to test the service, according to uv.lock
	uv sync --group dev

update-deps: ## Check pyproject.toml for changes, update the lock file if needed, then sync.
	uv lock
	uv sync --group dev

check-types: ## Checks type hints in sources
	uv run mypy --explicit-package-bases --disallow-untyped-calls --disallow-untyped-defs --disallow-incomplete-defs src/ lsc_agent_eval/src/ tests

black-check:
	uv run black src tests script lsc_agent_eval --check

black-format:
	uv run black src tests script lsc_agent_eval

uv-lock-regenerate: ## Regenerate both CPU and GPU lock files from pyproject.toml
	@echo "Regenerating CPU lock file (uv.lock)..."
	uv lock
	@echo "Regenerating GPU lock file (uv-gpu.lock)..."
	@# Use mktemp for safe temporary files
	@( \
		set -e; \
		BACKUP_FILE=$$(mktemp "$${TMPDIR:-/tmp}/pyproject-backup.XXXXXX"); \
		TEMP_FILE=$$(mktemp "$${TMPDIR:-/tmp}/pyproject-temp.XXXXXX"); \
		trap "rm -f $$TEMP_FILE; [ -f $$BACKUP_FILE ] && mv $$BACKUP_FILE pyproject.toml; echo '❌ Error: GPU lock generation failed, pyproject.toml restored'; exit 1" EXIT; \
		cp pyproject.toml $$BACKUP_FILE; \
		sed '/^\[tool\.uv\.sources\]/,/^torch = /d' pyproject.toml > $$TEMP_FILE; \
		mv $$TEMP_FILE pyproject.toml; \
		uv lock --locked 2>/dev/null || uv lock; \
		mv uv.lock uv-gpu.lock; \
		mv $$BACKUP_FILE pyproject.toml; \
		trap - EXIT; \
	)
	@echo "Restoring CPU lock file (uv.lock)..."
	uv lock
	@echo "✅ Done! Created uv.lock (CPU) and uv-gpu.lock (GPU)"

generate-requirements: ## Generate pinned requirements-*.txt from uv.lock (no -e ., safe without clone)
	@echo "Generating requirements.txt (runtime only, no optional extras)..."
	uv export --frozen --no-hashes --no-dev --no-emit-project -o requirements.txt
	@echo "Generating requirements-nlp-metrics.txt (base + nlp-metrics)..."
	uv export --frozen --no-hashes --no-dev --no-emit-project --extra nlp-metrics -o requirements-nlp-metrics.txt
	@echo "Generating requirements-local-embeddings.txt (base + local-embeddings, excludes torch)..."
	uv export --frozen --no-hashes --no-dev --no-emit-project --extra local-embeddings --no-emit-package torch -o requirements-local-embeddings.txt
	@echo "Generating requirements-all-extras.txt (base + all optional extras, excludes torch)..."
	uv export --frozen --no-hashes --no-dev --no-emit-project --all-extras --no-emit-package torch -o requirements-all-extras.txt
	@TORCH_VERSION=$$(grep 'torch.*version.*2\.' uv.lock | head -1 | sed 's/.*version = "\([^"]*\)".*/\1/'); \
	echo ""; \
	echo "========================================"; \
	echo "Requirements files generated:"; \
	echo "  - requirements.txt"; \
	echo "  - requirements-nlp-metrics.txt"; \
	echo "  - requirements-local-embeddings.txt (torch excluded - install separately)"; \
	echo "  - requirements-all-extras.txt (torch excluded - install separately)"; \
	echo ""; \
	echo "For local-embeddings extras, install torch separately:"; \
	echo "  CPU: pip install torch==$$TORCH_VERSION --index-url https://download.pytorch.org/whl/cpu"; \
	echo "  GPU: pip install torch==$$TORCH_VERSION"; \
	echo "Note: Dev dependencies use uv.lock for local development"; \
	echo "========================================"

sync-lock-and-requirements: uv-lock-regenerate generate-requirements ## Regenerate lock files and requirements-*.txt

verify-packages-completeness: requirements-all-extras.txt ## Verify pinned requirements resolve (full optional set)
	uv pip download -d /tmp/ --use-pep517 --verbose -r requirements-all-extras.txt

distribution-archives: ## Generate distribution archives to be uploaded into Python registry
	uv run python -m build

test: install-deps-test ## Execute tests with Pytest
	uv run pytest tests lsc_agent_eval/tests

e2e_tests: install-deps-test
	uv run pytest tests/integration -v -m integration

e2e_tests_lcore: e2e_tests
	# May be changed in the future to different test suite
	echo "LCORE e2e tests done"

pre-commit: black-check docstyle pyright pylint ruff check-types bandit
	@echo "All checks successful"

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
	uv run pylint --disable=R0801 lsc_agent_eval/src tests

pyright:
	uv run pyright src lsc_agent_eval/src tests

docstyle:
	uv run pydocstyle -v src tests script lsc_agent_eval

ruff:
	uv run ruff check src tests script lsc_agent_eval

bandit: ## Security scanning with Bandit
	uv run bandit -r src/lightspeed_evaluation -ll
