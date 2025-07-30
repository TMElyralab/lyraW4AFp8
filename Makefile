.PHONY: help check-deps install-deps tree ln submodule install build clean rebuild test format update

# Show help for each target
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

check-deps: ## Check and install required Python formatting dependencies
	@command -v isort >/dev/null 2>&1 || (echo "Installing isort..." && pip install isort)
	@command -v black >/dev/null 2>&1 || (echo "Installing black..." && pip install black)

install-deps: ## Install Python formatting tools (isort and black)
	pip install scikit-build-core isort black

tree: ## Show project directory structure
	@tree --prune -I "__pycache__|*.egg-info|*.so|build|3rdparty|dist"

submodule: ## Initialize and update git submodules
	@git submodule update --init --recursive

ln: submodule ## Create compilation database
	@rm -rf build && mkdir build && cd build && cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=YES -DCMAKE_POLICY_VERSION_MINIMUM=3.5


install: submodule ## Install package in development mode
	@pip install -e . --no-build-isolation

build: install-deps submodule ## Build and install wheel package
	@rm -rf dist/* || true && export MAX_JOBS=$(nproc) && CMAKE_POLICY_VERSION_MINIMUM=3.5 CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) uv build --wheel -Cbuild-dir=build . --verbose --color=always --no-build-isolation && pip3 install dist/*whl --force-reinstall --no-deps

clean: ## Remove build artifacts
	@rm -rf build dist *.egg-info

rebuild: clean submodule build ## Clean and rebuild the project
	@echo "Succeed to rebuild"

test: ## Run all tests
	@find tests -name "test_*.py" | xargs -n 1 python3

