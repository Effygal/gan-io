#!/usr/bin/env python3
"""Legacy entry point that now proxies to the packaged CLI."""

from gan_io.cli import main


if __name__ == "__main__":
    main()
