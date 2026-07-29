"""Process-monitoring runner.

Spawns a command as a black-box subprocess, samples its process tree and
system-wide metrics on an interval, and writes a CSV plus a markdown report.
Standalone: imports only the standard library and psutil.

Usage:
    python -m runner.monitor [--interval SEC] [--duration SEC] [--report-every SEC] [--out DIR] -- <command...>
"""

import argparse
import csv
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

FIELDS = [
    "elapsed",
    "rss_bytes",
    "cpu_percent",
    "threads",
    "fds",
    "inet_sockets",
    "sys_cpu_percent",
    "sys_mem_percent",
    "net_bytes_sent",
    "net_bytes_recv",
]

# (label, field, scale divisor, format); the first three are also the timeline charts.
SUMMARY_SPECS = [
    ("Process RSS (MB)", "rss_bytes", 2**20, "{:.1f}"),
    ("Process CPU %", "cpu_percent", 1, "{:.1f}"),
    ("Threads", "threads", 1, "{:.0f}"),
    ("Open FDs", "fds", 1, "{:.0f}"),
    ("Inet sockets", "inet_sockets", 1, "{:.0f}"),
    ("System CPU %", "sys_cpu_percent", 1, "{:.1f}"),
    ("System memory %", "sys_mem_percent", 1, "{:.1f}"),
]

CHART_CAP = 512


class Stats:
    """Bounded aggregate state so memory stays flat over an arbitrarily long run:
    exact min/mean/peak per metric, first/last network counters, and a decimated
    subset of samples for the timeline charts."""

    def __init__(self):
        self.count = 0
        self.agg = {}  # field -> [min, max, total]
        self.chart = []  # every stride-th sample, halved when it hits CHART_CAP
        self.stride = 1
        self.first_net = self.last_net = None

    def add(self, sample):
        for _, field, _, _ in SUMMARY_SPECS:
            value = sample[field]
            entry = self.agg.setdefault(field, [value, value, 0.0])
            entry[0] = min(entry[0], value)
            entry[1] = max(entry[1], value)
            entry[2] += value
        net = (sample["net_bytes_sent"], sample["net_bytes_recv"])
        if self.first_net is None:
            self.first_net = net
        self.last_net = net
        if self.count % self.stride == 0:
            self.chart.append(sample)
            if len(self.chart) >= CHART_CAP:
                self.chart = self.chart[::2]
                self.stride *= 2
        self.count += 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m runner.monitor",
        description="Run a command and sample its process tree until it exits.",
    )
    parser.add_argument("--interval", type=float, default=1.0, help="sampling interval in seconds (default: 1.0)")
    parser.add_argument("--duration", type=float, default=None, help="terminate the command after this many seconds")
    parser.add_argument("--report-every", type=float, default=60.0, help="rewrite the report this often while running (default: 60)")
    parser.add_argument("--out", default="reports", help="reports directory (default: reports)")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command to run, after --")
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error("--interval must be > 0")
    if args.report_every <= 0:
        parser.error("--report-every must be > 0")
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("no command given (put it after --)")
    return args


def inet_sockets(proc):
    # Process.net_connections() replaced connections() in psutil 6.0.
    if hasattr(proc, "net_connections"):
        return proc.net_connections(kind="inet")
    return proc.connections(kind="inet")


def collect_sample(root, cache, elapsed):
    """Sample the process tree; returns None if the whole tree is gone.

    ``cache`` maps pid -> psutil.Process and persists across samples, so each
    cpu_percent() measures since the previous sample instead of resetting to a
    fresh 0.0 baseline on a brand-new Process object every time.
    """
    procs = {root.pid: root}
    try:
        for proc in root.children(recursive=True):
            cached = cache.get(proc.pid)
            # psutil equality includes creation time, guarding against pid reuse.
            procs[proc.pid] = cached if cached == proc else proc
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass
    cache.clear()
    cache.update(procs)
    rss = cpu = threads = fds = sockets = seen = 0
    for proc in procs.values():
        # Read into locals first, commit only on full success, so a process
        # failing partway through doesn't leave a half-counted sample.
        try:
            with proc.oneshot():
                p_rss = proc.memory_info().rss
                # First cpu_percent() call per process is a 0.0 baseline.
                p_cpu = proc.cpu_percent(interval=None)
                p_threads = proc.num_threads()
                p_fds = proc.num_fds()
                p_sockets = len(inet_sockets(proc))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue  # exited mid-sample: normal lifecycle, skip this process
        rss += p_rss
        cpu += p_cpu
        threads += p_threads
        fds += p_fds
        sockets += p_sockets
        seen += 1
    if seen == 0:
        return None
    net = psutil.net_io_counters()
    return {
        "elapsed": round(elapsed, 2),
        "rss_bytes": rss,
        "cpu_percent": round(cpu, 1),
        "threads": threads,
        "fds": fds,
        "inet_sockets": sockets,
        "sys_cpu_percent": psutil.cpu_percent(interval=None),
        "sys_mem_percent": psutil.virtual_memory().percent,
        "net_bytes_sent": net.bytes_sent,
        "net_bytes_recv": net.bytes_recv,
    }


def stop_child(child, root):
    """Terminate the whole tree; the child may not forward SIGTERM to its own."""
    if child.poll() is not None:
        return
    try:
        descendants = root.children(recursive=True)
    except psutil.Error:
        descendants = []
    child.terminate()
    for proc in descendants:
        try:
            proc.terminate()
        except psutil.Error:
            pass
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
    for proc in psutil.wait_procs(descendants, timeout=5)[1]:
        try:
            proc.kill()
        except psutil.Error:
            pass


def human_bytes(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def summary_table(stats):
    lines = ["| Metric | Min | Mean | Peak |", "| --- | --- | --- | --- |"]
    for label, field, scale, fmt in SUMMARY_SPECS:
        lo, hi, total = stats.agg[field]
        cells = (fmt.format(v / scale) for v in (lo, total / stats.count, hi))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def line_chart(points, fmt="{:.1f}", height=8, width=60):
    """Text line chart of (elapsed, value) points, one column per time bucket.
    The series is traced with box-drawing glyphs — runs, corners, and vertical
    connectors — so it reads as a line, not bars. Fenced as a code block so it
    renders monospaced in any markdown viewer."""
    if len(points) > width:
        last = len(points) - 1
        points = [points[round(i * last / (width - 1))] for i in range(width)]
    values = [v for _, v in points]
    lo, hi = min(values), max(values)
    span = hi - lo
    levels = [round((v - lo) / span * (height - 1)) if span else 0 for v in values]
    grid = [[" "] * len(levels) for _ in range(height)]  # row 0 = bottom
    previous = levels[0]
    for column, level in enumerate(levels):
        if level == previous:
            grid[level][column] = "─"
        else:
            rising = level > previous
            grid[level][column] = "╭" if rising else "╰"
            grid[previous][column] = "╯" if rising else "╮"
            for row in range(min(previous, level) + 1, max(previous, level)):
                grid[row][column] = "│"
        previous = level
    lines = ["```"]
    for row in range(height - 1, -1, -1):
        if row == height - 1:
            label = f"{fmt.format(hi):>8} ┤"
        elif row == 0:
            label = f"{fmt.format(lo):>8} ┤"
        else:
            label = " " * 9 + "│"
        lines.append(label + "".join(grid[row]))
    lines.append(" " * 9 + "└" + "─" * len(levels))
    end = f"{points[-1][0]:.0f}s"
    lines.append(" " * 10 + "0s" + end.rjust(max(len(levels) - 2, len(end))))
    lines.append("```")
    return lines


def timeline_charts(samples):
    """One line chart per timeline metric."""
    specs = [
        ("Process RSS (MB)", lambda s: s["rss_bytes"] / 2**20, "{:.1f}"),
        ("Process CPU %", lambda s: s["cpu_percent"], "{:.1f}"),
        ("Threads", lambda s: s["threads"], "{:.0f}"),
    ]
    lines = []
    for title, get, fmt in specs:
        lines += [f"### {title}", ""]
        lines += line_chart([(s["elapsed"], get(s)) for s in samples], fmt)
        lines.append("")
    return lines[:-1]


def write_report(path, command, started, ended, exit_code, interval, stats):
    running = exit_code is None
    lines = [
        "# Monitor report",
        "",
        f"- Command: `{shlex.join(command)}`",
        f"- Started: {started}",
        f"- {'Report generated' if running else 'Ended'}: {ended}",
        f"- Wall duration: {(datetime.fromisoformat(ended) - datetime.fromisoformat(started)).total_seconds():.1f} s",
        f"- Child exit code: {'still running' if running else exit_code}",
        f"- Samples: {stats.count} at {interval} s interval",
        "",
    ]
    if stats.count:
        sent = stats.last_net[0] - stats.first_net[0]
        recv = stats.last_net[1] - stats.first_net[1]
        lines += ["## Summary", ""]
        lines += summary_table(stats)
        lines += [
            "",
            "## Network",
            "",
            f"- Bytes sent: {human_bytes(sent)}",
            f"- Bytes received: {human_bytes(recv)}",
            "- Caveat: network counters are host-wide, not scoped to the monitored process.",
            "",
            "## Timeline",
            "",
        ]
        lines += timeline_charts(stats.chart)
    else:
        lines += ["No samples collected (child exited before it could be sampled).", ""]
    lines += [
        "",
        "## Notes",
        "",
        "- RSS (resident set size) is the physical RAM the process tree holds —",
        "  excluding swapped-out pages and reserved-but-unused virtual memory.",
        "- The first sample's CPU figures (per-process and system-wide) are 0.0 baselines,",
        "  which skews the Summary Min/Mean for the CPU rows low.",
        "- Network byte counts are host-wide, not process-scoped.",
        "- FD and socket counts are point-in-time snapshots at each sample.",
        "",
    ]
    Path(path).write_text("\n".join(lines))


def main(argv=None):
    args = parse_args(argv)
    # Microsecond timestamp avoids two runs in the same second sharing a
    # directory; exist_ok=False makes any residual collision an error, not a
    # silent clobber.
    run_dir = Path(args.out) / f"run-{datetime.now():%Y%m%d-%H%M%S-%f}"
    run_dir.mkdir(parents=True, exist_ok=False)

    started = datetime.now().isoformat(timespec="seconds")
    child = subprocess.Popen(args.command)
    root = psutil.Process(child.pid)
    print(f"monitor: pid {child.pid}, sampling every {args.interval} s", file=sys.stderr)

    start = time.monotonic()
    stats = Stats()
    cache = {}
    csv_file = open(run_dir / "samples.csv", "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
    writer.writeheader()

    def take_sample():
        sample = collect_sample(root, cache, time.monotonic() - start)
        if sample:
            stats.add(sample)
            writer.writerow(sample)

    take_sample()  # immediate t=0 sample, so short-lived children still get one
    last_flush = 0.0
    try:
        while child.poll() is None:
            remaining = None
            if args.duration is not None:
                remaining = args.duration - (time.monotonic() - start)
                if remaining <= 0:
                    take_sample()  # final sample before tearing the tree down
                    stop_child(child, root)
                    break
            try:
                child.wait(timeout=min(args.interval, remaining) if remaining else args.interval)
            except subprocess.TimeoutExpired:
                take_sample()
                elapsed = time.monotonic() - start
                if elapsed - last_flush >= args.report_every:
                    csv_file.flush()
                    now = datetime.now().isoformat(timespec="seconds")
                    write_report(run_dir / "report.md", args.command, started, now, None, args.interval, stats)
                    last_flush = elapsed
    except KeyboardInterrupt:
        stop_child(child, root)
    ended = datetime.now().isoformat(timespec="seconds")

    csv_file.close()
    write_report(run_dir / "report.md", args.command, started, ended, child.returncode, args.interval, stats)
    print(f"monitor: report written to {run_dir}", file=sys.stderr)
    # Map signal deaths (negative returncode) to the shell convention 128+signum,
    # so scripts don't see e.g. -15 wrap around to exit status 241.
    return 128 - child.returncode if child.returncode < 0 else child.returncode


if __name__ == "__main__":
    sys.exit(main())
