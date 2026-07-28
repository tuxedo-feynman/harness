# deploy

Docker-based deploy tooling for hyh. Stdlib only, fail fast.

## Commands

    python deploy/deploy.py package              # build hyh:latest locally (default linux/amd64), smoke test it
    python deploy/deploy.py deploy USER@HOST     # build for the target's arch, ship over ssh (no registry), smoke test it
    python deploy/deploy.py cloud                # not implemented; prints the intended shape

`deploy` prints the run command instead of executing it:

    docker run -d --env-file ~/.hyh/env --name hyh hyh:latest

Deliberately no `--restart` policy: the process stays fragile, and keeping it
always-on is the target machine's concern.

## Secrets

The image and `deploy/config.yaml` contain no secrets. Put them in
`deploy/hyh.env` (gitignored; `deploy` copies it to `~/.hyh/env` on the target):

    ANTHROPIC_API_KEY=sk-ant-...
    TELEGRAM_BOT_TOKEN=123456:...
