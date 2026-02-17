variable "account_id" {
  description = "Cloudflare account ID (set via TF_VAR_account_id or terraform.tfvars)"
  type        = string
}

variable "domain" {
  description = "Primary domain for the site"
  type        = string
  default     = "camazotzdiving.com"
}

variable "project_name" {
  description = "Cloudflare Pages project name"
  type        = string
  default     = "planner"
}

variable "subdomain" {
  description = "Subdomain for the dive planner"
  type        = string
  default     = "planner"
}
