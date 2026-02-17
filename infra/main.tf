locals {
  zone_id = data.cloudflare_zones.site.result[0].id
}

# Look up the existing Cloudflare zone for the domain
data "cloudflare_zones" "site" {
  account = {
    id = var.account_id
  }
  name = var.domain
}

# Create the Cloudflare Pages project (direct upload, no git integration)
resource "cloudflare_pages_project" "site" {
  account_id        = var.account_id
  name              = var.project_name
  production_branch = "main"
}

# Attach the planner subdomain to the Pages project
resource "cloudflare_pages_domain" "planner" {
  account_id   = var.account_id
  project_name = cloudflare_pages_project.site.name
  name         = "${var.subdomain}.${var.domain}"
}

# Create CNAME record for planner subdomain pointing to Pages
resource "cloudflare_dns_record" "planner" {
  zone_id = local.zone_id
  name    = var.subdomain
  content = cloudflare_pages_project.site.subdomain
  type    = "CNAME"
  proxied = true
  ttl     = 1
  comment = "Dive planner on Cloudflare Pages"
}
