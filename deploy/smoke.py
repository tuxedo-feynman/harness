#!/usr/bin/env python3
"""Black-box smoke tests for the packaged hyh Docker image.

Usage:
    python deploy/smoke.py --image hyh:latest
    python deploy/smoke.py --image hyh:latest --ssh user@host
    python deploy/smoke.py --image hyh:latest --live

The app is exercised only through `docker run` (optionally via ssh).
Stdlib only; no imports from the application packages.

Telegram never starts in any of these tests: single-shot mode never arms
listeners, and the fake-thinking config used by the chat test declares no
telegram action. The --live test is also single-shot.
"""

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import time
import uuid

GREETING = "Hello! How can I help you?"
TIMEOUT = 180

# Fake-thinking config mounted over the image's /app/config.yaml.
FAKE_CONFIG = """\
actions:
  terminal:
  fake:
default_thinking_action: fake
"""


class Failure(Exception):
    def __init__(self, message, proc=None):
        super().__init__(message)
        self.proc = proc


class Host:
    """Runs commands locally, or on a remote host when ssh is set."""

    def __init__(self, ssh=None):
        self.ssh = ssh

    def wrap(self, argv):
        if self.ssh:
            return ["ssh", self.ssh, " ".join(shlex.quote(a) for a in argv)]
        return argv

    def run(self, argv, stdin=""):
        # stdin defaults to an empty pipe (immediate EOF) so no run can
        # block on the caller's inherited stdin.
        try:
            return subprocess.run(
                self.wrap(argv),
                input=stdin,
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            raise Failure("timed out after %ds: %s" % (TIMEOUT, " ".join(argv))) from exc


def stage_config(host, tmpdir):
    """Write the fake config; return (path usable in -v mount, cleanup fn)."""
    local = os.path.join(tmpdir, "config.yaml")
    with open(local, "w") as f:
        f.write(FAKE_CONFIG)
    if not host.ssh:
        return local, lambda: None
    remote = "/tmp/hyh-smoke-%s.yaml" % uuid.uuid4().hex
    proc = subprocess.run(
        ["scp", "-q", local, "%s:%s" % (host.ssh, remote)],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )
    if proc.returncode != 0:
        raise Failure("scp of fake config to %s failed" % host.ssh, proc)

    def cleanup():
        subprocess.run(
            ["ssh", host.ssh, "rm -f %s" % shlex.quote(remote)],
            capture_output=True,
            timeout=TIMEOUT,
        )

    return remote, cleanup


def expect(proc, must_contain=()):
    if proc.returncode != 0:
        raise Failure("exit code %d, expected 0" % proc.returncode, proc)
    for needle in must_contain:
        if needle not in proc.stdout:
            raise Failure("stdout missing %r" % needle, proc)


def test_chat(host, image, cfg):
    proc = host.run(
        ["docker", "run", "-i", "--rm",
         "-v", "%s:/app/config.yaml" % cfg,
         image, "python", "-m", "hyh.main", "--chat"],
        stdin="hi\nsecond\nquit\n",
    )
    expect(proc, [GREETING, "[fake] second"])


def test_single_shot(host, image, cfg):
    proc = host.run(
        ["docker", "run", "--rm",
         "-v", "%s:/app/config.yaml" % cfg,
         image, "python", "-m", "hyh.main", "hello there"],
    )
    expect(proc, [GREETING])


def test_live(host, image):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise Failure("ANTHROPIC_API_KEY is not set in the environment (required for --live)")
    # No config mount: the baked /app/config.yaml defaults to claude.
    proc = host.run(
        ["docker", "run", "--rm",
         "-e", "ANTHROPIC_API_KEY=%s" % key,
         image, "python", "-m", "hyh.main", "Reply with exactly: OK"],
    )
    expect(proc)
    if not proc.stdout.strip():
        raise Failure("stdout is empty", proc)


def check(name, fn):
    start = time.monotonic()
    try:
        fn()
    except Failure as exc:
        elapsed = time.monotonic() - start
        print("FAIL %s (%.2fs)" % (name, elapsed))
        print("smoke: %s failed: %s" % (name, exc), file=sys.stderr)
        if exc.proc is not None:
            print("--- stdout ---\n%s" % exc.proc.stdout, file=sys.stderr)
            print("--- stderr ---\n%s" % exc.proc.stderr, file=sys.stderr)
        sys.exit(1)
    elapsed = time.monotonic() - start
    print("PASS %s (%.2fs)" % (name, elapsed))


def main():
    parser = argparse.ArgumentParser(
        description="Black-box smoke tests for the hyh docker image."
    )
    parser.add_argument("--image", required=True, help="docker image, e.g. hyh:latest")
    parser.add_argument("--ssh", metavar="USER@HOST",
                        help="run docker commands on a remote host via ssh")
    parser.add_argument("--live", action="store_true",
                        help="also run one real Claude turn (needs ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    host = Host(args.ssh)
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            cfg, cleanup = stage_config(host, tmpdir)
        except Failure as exc:
            print("smoke: setup failed: %s" % exc, file=sys.stderr)
            if exc.proc is not None:
                print(exc.proc.stderr, file=sys.stderr)
            sys.exit(1)
        try:
            check("chat-loop", lambda: test_chat(host, args.image, cfg))
            check("single-shot", lambda: test_single_shot(host, args.image, cfg))
            if args.live:
                check("live-claude", lambda: test_live(host, args.image))
        finally:
            cleanup()


if __name__ == "__main__":
    main()
