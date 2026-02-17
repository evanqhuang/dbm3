output "pages_url" {
  description = "The default Pages deployment URL"
  value       = "https://${var.project_name}.pages.dev"
}

output "custom_domain_url" {
  description = "The custom subdomain URL"
  value       = "https://${var.subdomain}.${var.domain}"
}

output "zone_id" {
  description = "The Cloudflare zone ID"
  value       = local.zone_id
}

output "project_id" {
  description = "The Cloudflare Pages project ID"
  value       = cloudflare_pages_project.site.id
}
