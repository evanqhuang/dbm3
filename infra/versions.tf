# State is stored locally. This project has a single operator.
# If multi-user access is needed, migrate to a remote backend.
terraform {
  required_version = ">= 1.0"
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 5.0"
    }
  }
}

provider "cloudflare" {
  # Reads CLOUDFLARE_API_TOKEN from environment automatically
}
