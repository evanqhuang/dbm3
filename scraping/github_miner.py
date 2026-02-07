"""
GitHub mining module for extracting dive log file attachments from issue trackers.

This module implements a GitHubMiner class that:
- Fetches issues and comments from GitHub repositories
- Extracts file attachment URLs using regex patterns
- Downloads dive log files (XML, UDDF, BIN, ZIP, etc.)
- Maintains an index of downloaded files to avoid duplicates
- Handles GitHub API rate limiting and authentication
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests

logger = logging.getLogger(__name__)

# File extensions we want to download
DIVE_FILE_EXTENSIONS = {'.xml', '.uddf', '.bin', '.zip', '.db', '.sde', '.dmp', '.ssrf'}

# Regex patterns for GitHub attachment URLs
ATTACHMENT_PATTERNS = [
    re.compile(r'https://github\.com/user-attachments/(?:assets|files)/[a-f0-9-]+[^\s\)\]"\']*'),
    re.compile(r'https://github\.com/[^/]+/[^/]+/files/\d+/[^\s\)\]"\']+'),
    re.compile(r'https://(?:private-)?user-images\.githubusercontent\.com/[^\s\)\]"\']+'),
]


def _load_dotenv_token() -> str | None:
    """Load GITHUB_TOKEN from .env file in project root."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("GITHUB_TOKEN=") and not line.startswith("#"):
            return line.split("=", 1)[1].strip()
    return None


class GitHubMiner:
    """Mines dive log file attachments from GitHub issue trackers."""

    def __init__(self, token: str | None = None, output_dir: str = "data/raw"):
        """
        Initialize the GitHub miner.

        Args:
            token: GitHub Personal Access Token. If None, reads from GITHUB_TOKEN env var.
                   Falls back to unauthenticated (60 req/hr).
            output_dir: Directory to save downloaded files.
        """
        self.token = token or os.environ.get("GITHUB_TOKEN") or _load_dotenv_token()
        self.output_dir = Path(output_dir)
        self.session = requests.Session()
        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"
            logger.info("Using authenticated GitHub API (5000 req/hr)")
        else:
            logger.warning("No GITHUB_TOKEN set - using unauthenticated API (60 req/hr)")
        self.session.headers["Accept"] = "application/vnd.github.v3+json"
        self.session.headers["User-Agent"] = "DBM3-DataMiner/1.0"

        # Index for tracking downloads
        self.index_path = self.output_dir.parent / "index.json"
        self.index = self._load_index()

    def mine_repo(self, owner: str, repo: str, max_pages: int = 0) -> list[dict]:
        """
        Mine all issues (and their comments) from a repo for dive log attachments.

        Args:
            owner: GitHub repo owner
            repo: GitHub repo name
            max_pages: Max pages of issues to fetch (0 = unlimited)

        Returns:
            List of download metadata dicts
        """
        repo_dir = self.output_dir / repo
        repo_dir.mkdir(parents=True, exist_ok=True)

        downloads = []
        page = 1

        while True:
            if max_pages and page > max_pages:
                break

            issues = self._fetch_issues(owner, repo, page)
            if not issues:
                break

            for issue in issues:
                # Skip pull requests (they appear in the issues endpoint)
                if "pull_request" in issue:
                    continue

                issue_num = issue["number"]

                # Extract from issue body
                attachments = self._extract_attachments(issue.get("body") or "")

                # Extract from comments
                if issue.get("comments", 0) > 0:
                    comments = self._fetch_comments(owner, repo, issue_num)
                    for comment in comments:
                        attachments.extend(self._extract_attachments(comment.get("body") or ""))

                # Download each attachment
                for att in attachments:
                    filename = f"issue_{issue_num}_{att['filename']}"
                    dest = repo_dir / filename

                    if dest.exists():
                        logger.debug(f"Skipping existing: {filename}")
                        continue

                    if self._download_file(att["url"], str(dest)):
                        meta = {
                            "source": f"{owner}/{repo}",
                            "issue": issue_num,
                            "issue_title": issue.get("title", ""),
                            "url": att["url"],
                            "filename": filename,
                            "path": str(dest),
                            "extension": att["extension"],
                        }
                        downloads.append(meta)
                        self.index.setdefault("downloads", []).append(meta)

            logger.info(f"Processed page {page} of {owner}/{repo} ({len(issues)} issues)")
            page += 1

        self._save_index()
        logger.info(f"Mining complete: {len(downloads)} new files from {owner}/{repo}")
        return downloads

    def _fetch_issues(self, owner: str, repo: str, page: int, per_page: int = 100) -> list[dict]:
        """Fetch a page of issues from the GitHub API."""
        url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        params = {
            "state": "all",
            "page": page,
            "per_page": per_page,
            "sort": "created",
            "direction": "desc"
        }

        resp = self._request_with_retry(url, params=params)
        if resp is None:
            return []
        return resp.json()

    def _fetch_comments(self, owner: str, repo: str, issue_number: int) -> list[dict]:
        """Fetch all comments for a specific issue."""
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
        params = {"per_page": 100}

        resp = self._request_with_retry(url, params=params)
        if resp is None:
            return []
        return resp.json()

    def _extract_attachments(self, text: str) -> list[dict]:
        """Extract file attachment URLs from markdown text."""
        attachments = []
        seen_urls = set()

        for pattern in ATTACHMENT_PATTERNS:
            for match in pattern.finditer(text):
                url = match.group(0).rstrip('.,;:')
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                # Try to extract filename from URL
                parsed = urlparse(url)
                path_parts = parsed.path.rstrip('/').split('/')
                filename = unquote(path_parts[-1]) if path_parts else "unknown"

                # Only download files with recognized dive log extensions
                ext = Path(filename).suffix.lower()
                if ext in DIVE_FILE_EXTENSIONS:
                    attachments.append({"url": url, "filename": filename, "extension": ext})

        return attachments

    def _download_file(self, url: str, dest: str) -> bool:
        """Download a file, following redirects (GitHub redirects to S3)."""
        try:
            resp = self.session.get(url, stream=True, timeout=30, allow_redirects=True)
            resp.raise_for_status()

            with open(dest, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded: {Path(dest).name} ({os.path.getsize(dest)} bytes)")
            return True
        except requests.RequestException as e:
            logger.warning(f"Failed to download {url}: {e}")
            return False

    def _request_with_retry(self, url: str, params: dict | None = None, max_retries: int = 3) -> requests.Response | None:
        """Make a GET request with retry and rate limit handling."""
        for attempt in range(max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=30)

                # Check rate limit
                self._handle_rate_limit(resp)

                if resp.status_code == 200:
                    return resp
                elif resp.status_code == 403 and "rate limit" in resp.text.lower():
                    # Already handled by _handle_rate_limit, retry
                    continue
                elif resp.status_code == 404:
                    logger.debug(f"Not found: {url}")
                    return None
                else:
                    logger.warning(f"HTTP {resp.status_code} for {url}")

            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        return None

    def _handle_rate_limit(self, response: requests.Response) -> None:
        """Sleep if approaching rate limit."""
        remaining = int(response.headers.get("X-RateLimit-Remaining", 999))
        if remaining < 50:
            reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
            sleep_seconds = max(reset_time - int(time.time()), 1) + 5
            logger.warning(f"Rate limit low ({remaining} remaining). Sleeping {sleep_seconds}s...")
            time.sleep(sleep_seconds)

    def _load_index(self) -> dict:
        """Load the download index from disk."""
        if self.index_path.exists():
            with open(self.index_path) as f:
                return json.load(f)
        return {"downloads": []}

    def _save_index(self) -> None:
        """Save the download index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)
