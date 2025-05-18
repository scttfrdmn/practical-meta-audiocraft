source "https://rubygems.org"

# GitHub Pages
gem "github-pages", "~> 228", group: :jekyll_plugins
gem "minima", "~> 2.5"

group :jekyll_plugins do
  gem "jekyll-feed", "~> 0.12"
  gem "jekyll-seo-tag", "~> 2.8"
  gem "jekyll-sitemap", "~> 1.4"
  gem "jekyll-relative-links", "~> 0.7.0"
end

# JRuby does not include zoneinfo files
platforms :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end