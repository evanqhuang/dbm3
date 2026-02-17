-include .env
export CLOUDFLARE_API_TOKEN
export CLOUDFLARE_ACCOUNT_ID
export TF_VAR_account_id = $(CLOUDFLARE_ACCOUNT_ID)

INFRA_DIR := infra
.DEFAULT_GOAL := help

##@ Development

.PHONY: sync
sync: ## Sync Python model files for Pyodide
	bash scripts/sync_planner_py.sh

.PHONY: serve
serve: sync ## Run local development server at http://localhost:8080
	python3 -m http.server -d docs 8080

##@ Deploy

.PHONY: deploy
deploy: sync ## Deploy to Cloudflare Pages
	python3 scripts/compact_data.py
	npx wrangler pages deploy docs --project-name=planner --commit-dirty=true

##@ Infrastructure

.PHONY: tf-init
tf-init: ## Initialize Terraform
	terraform -chdir=$(INFRA_DIR) init

.PHONY: tf-plan
tf-plan: ## Show Terraform execution plan
	terraform -chdir=$(INFRA_DIR) plan

.PHONY: tf-apply
tf-apply: ## Apply Terraform changes
	terraform -chdir=$(INFRA_DIR) apply

.PHONY: tf-destroy
tf-destroy: ## Destroy Terraform-managed infrastructure
	terraform -chdir=$(INFRA_DIR) destroy

.PHONY: tf-fmt
tf-fmt: ## Format Terraform files
	terraform fmt $(INFRA_DIR)

.PHONY: tf-validate
tf-validate: ## Validate Terraform configuration
	terraform -chdir=$(INFRA_DIR) validate

##@ Help

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } \
		/^[a-zA-Z_-]+:.*?## / { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
