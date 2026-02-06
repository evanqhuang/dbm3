#!/usr/bin/env python3
"""
Dive data scraping CLI for mining and parsing dive logs from GitHub.

Usage:
    python run_scraper.py mine                          # Mine default repos
    python run_scraper.py mine --repo owner/repo        # Mine specific repo
    python run_scraper.py mine --max-pages 5            # Limit pages fetched
    python run_scraper.py parse                         # Parse all raw files
    python run_scraper.py stats                         # Print dataset statistics
    python run_scraper.py all                           # Mine + Parse + Stats
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scraping import GitHubMiner, parse_raw_files, load_profiles


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    log_dir = Path("data")
    log_dir.mkdir(exist_ok=True)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Console handler (INFO level)
    console_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=console_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "scraper.log")
        ]
    )


def mine_repos(repo_list: list[tuple[str, str]], max_pages: int = 0):
    """Mine dive logs from GitHub repositories."""
    print("\n" + "=" * 60)
    print("MINING GITHUB REPOSITORIES")
    print("=" * 60)

    miner = GitHubMiner()
    total_downloads = 0

    for owner, repo in repo_list:
        print(f"\nMining {owner}/{repo}...")
        try:
            downloads = miner.mine_repo(owner, repo, max_pages=max_pages)
            total_downloads += len(downloads)
            print(f"  Downloaded {len(downloads)} files")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print(f"Mining complete: {total_downloads} total files downloaded")
    print("=" * 60)


def parse_files():
    """Parse all raw dive log files."""
    print("\n" + "=" * 60)
    print("PARSING RAW FILES")
    print("=" * 60)

    try:
        count = parse_raw_files(raw_dir="data/raw", parsed_dir="data/parsed")
        print("\n" + "=" * 60)
        print(f"Parsing complete: {count} profiles parsed")
        print("=" * 60)
    except Exception as e:
        print(f"ERROR: {e}")


def show_stats():
    """Show dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    try:
        profiles = load_profiles(data_dir="data/parsed", min_points=10, max_depth=200.0)

        if not profiles:
            print("\nNo profiles loaded. Run 'mine' and 'parse' first.")
            return

        # Calculate statistics
        depths = [p.max_depth for p in profiles]
        times = [p.bottom_time for p in profiles]
        point_counts = [len(p.points) for p in profiles]

        print(f"\nTotal profiles: {len(profiles)}")
        print("\n--- Depth Statistics (m) ---")
        print(f"  Min: {min(depths):.1f}")
        print(f"  Max: {max(depths):.1f}")
        print(f"  Mean: {sum(depths)/len(depths):.1f}")

        print("\n--- Bottom Time Statistics (min) ---")
        print(f"  Min: {min(times):.1f}")
        print(f"  Max: {max(times):.1f}")
        print(f"  Mean: {sum(times)/len(times):.1f}")

        print("\n--- Sample Points Statistics ---")
        print(f"  Min: {min(point_counts)}")
        print(f"  Max: {max(point_counts)}")
        print(f"  Mean: {sum(point_counts)/len(point_counts):.1f}")

        # Depth distribution
        print("\n--- Depth Distribution ---")
        depth_bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 200)]
        for low, high in depth_bins:
            count = sum(1 for d in depths if low <= d < high)
            print(f"  {low:3d}-{high:3d}m: {count:5d} profiles")

        print("\n" + "=" * 60)

    except Exception as e:
        print(f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Mine and parse dive log files from GitHub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_scraper.py mine
  python run_scraper.py mine --repo subsurface/subsurface --max-pages 3
  python run_scraper.py parse
  python run_scraper.py stats
  python run_scraper.py all
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Mine subcommand
    mine_parser = subparsers.add_parser("mine", help="Mine dive logs from GitHub")
    mine_parser.add_argument(
        "--repo",
        type=str,
        help="Specific repo to mine (format: owner/repo)"
    )
    mine_parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Maximum pages to fetch (0 = unlimited)"
    )
    mine_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    # Parse subcommand
    parse_parser = subparsers.add_parser("parse", help="Parse raw dive log files")
    parse_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    # Stats subcommand
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")
    stats_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    # All subcommand
    all_parser = subparsers.add_parser("all", help="Run mine + parse + stats")
    all_parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Maximum pages to fetch (0 = unlimited)"
    )
    all_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Set up logging
    verbose = getattr(args, "verbose", False)
    setup_logging(verbose)

    # Default repos
    default_repos = [
        ("subsurface", "subsurface"),
        ("libdivecomputer", "libdivecomputer")
    ]

    # Execute command
    if args.command == "mine":
        if args.repo:
            # Parse owner/repo format
            parts = args.repo.split("/")
            if len(parts) != 2:
                print("ERROR: --repo must be in format 'owner/repo'")
                return
            repos = [(parts[0], parts[1])]
        else:
            repos = default_repos

        mine_repos(repos, max_pages=args.max_pages)

    elif args.command == "parse":
        parse_files()

    elif args.command == "stats":
        show_stats()

    elif args.command == "all":
        mine_repos(default_repos, max_pages=args.max_pages)
        parse_files()
        show_stats()


if __name__ == "__main__":
    main()
