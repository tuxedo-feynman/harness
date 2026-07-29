"""Build and ship the hyh docker image. Stdlib only, fail fast.

    python deploy/deploy.py package [--platform linux/amd64]
    python deploy/deploy.py deploy USER@HOST
    python deploy/deploy.py cloud
"""

import argparse
import os
import shlex
import subprocess
import sys
from typing import NoReturn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPLOY_DIR = os.path.join(REPO_ROOT, "deploy")
DOCKERFILE = os.path.join(DEPLOY_DIR, "Dockerfile")
SMOKE = os.path.join(DEPLOY_DIR, "smoke.py")
ENV_FILE = os.path.join(DEPLOY_DIR, "hyh.env")
IMAGE = "hyh:latest"

# macOS runs non-login ssh commands without /usr/local/bin on PATH
# (path_helper only runs for login shells), so remote docker needs this.
REMOTE_PATH = "PATH=/usr/local/bin:$PATH "

ARCH_TO_PLATFORM = {
    "x86_64": "linux/amd64",
    "amd64": "linux/amd64",
    "aarch64": "linux/arm64",
    "arm64": "linux/arm64",
}


def die(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    sys.exit(1)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+ " + shlex.join(cmd), file=sys.stderr)
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        die(f"command failed (exit {result.returncode}): {shlex.join(cmd)}")
    return result


def build(platform: str) -> None:
    run([
        "docker", "buildx", "build",
        "--platform", platform,
        "--load",
        "-t", IMAGE,
        "-f", DOCKERFILE,
        REPO_ROOT,
    ])


def smoke(ssh: str | None = None) -> None:
    cmd = [sys.executable, SMOKE, "--image", IMAGE]
    if ssh:
        cmd += ["--ssh", ssh]
    run(cmd)


def cmd_package(args: argparse.Namespace) -> None:
    build(args.platform)
    smoke()
    print(f"packaged {IMAGE} for {args.platform}, smoke test passed")


def cmd_deploy(args: argparse.Namespace) -> None:
    target = args.target

    probe = subprocess.run(
        ["ssh", target, REMOTE_PATH + "docker info"], capture_output=True, text=True
    )
    if probe.returncode != 0:
        die(
            f"cannot run docker on {target} via ssh:\n{probe.stderr.strip()}\n"
            "Check ssh connectivity and that docker is installed and running "
            "on the target (on older macOS, colima is an option)."
        )

    uname = subprocess.run(
        ["ssh", target, "uname -m"], capture_output=True, text=True
    )
    if uname.returncode != 0:
        die(f"cannot detect target arch via `ssh {target} uname -m`:\n{uname.stderr.strip()}")
    arch = uname.stdout.strip()
    platform = ARCH_TO_PLATFORM.get(arch)
    if platform is None:
        die(f"unsupported target arch {arch!r} (expected one of {sorted(ARCH_TO_PLATFORM)})")

    build(platform)

    print(f"+ docker save {IMAGE} | ssh {target} docker load", file=sys.stderr)
    save = subprocess.Popen(["docker", "save", IMAGE], stdout=subprocess.PIPE)
    load = subprocess.run(["ssh", target, REMOTE_PATH + "docker load"], stdin=save.stdout)
    save.stdout.close()
    if save.wait() != 0:
        die(f"docker save {IMAGE} failed")
    if load.returncode != 0:
        die(f"docker load on {target} failed")

    env_uploaded = os.path.exists(ENV_FILE)
    if env_uploaded:
        run(["ssh", target, "mkdir -p ~/.hyh"])
        run(["scp", ENV_FILE, f"{target}:.hyh/env"])
        run(["ssh", target, "chmod 600 ~/.hyh/env"])
    else:
        print(f"no {ENV_FILE}; skipping env file upload", file=sys.stderr)

    smoke(ssh=target)

    env_flag = "--env-file ~/.hyh/env " if env_uploaded else ""
    print(f"deployed {IMAGE} to {target} ({platform}). Run it there with:")
    print(f"  docker run -d {env_flag}-v ~/hyh-reports:/app/reports --name hyh hyh:latest")
    print("No --restart policy on purpose: keeping it always-on is the target machine's concern.")


def cmd_cloud(args: argparse.Namespace) -> None:
    die(
        "cloud: not implemented. Intended shape: push the image to a container "
        "registry and provision ANTHROPIC_API_KEY / TELEGRAM_BOT_TOKEN from a "
        "secret store instead of an env file."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_package = sub.add_parser("package", help="build the image locally and smoke test it")
    p_package.add_argument("--platform", default="linux/amd64")
    p_package.set_defaults(func=cmd_package)

    p_deploy = sub.add_parser("deploy", help="build for a remote host and ship it over ssh")
    p_deploy.add_argument("target", metavar="USER@HOST")
    p_deploy.set_defaults(func=cmd_deploy)

    p_cloud = sub.add_parser("cloud", help="not implemented")
    p_cloud.set_defaults(func=cmd_cloud)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
