#!/usr/bin/env python3
"""Run terminal Graff as a coding agent on Inkling 2 Mini, on an owned loopback adapter.

Run under tools/guarded_run.py --lock <inference lock> --timeout <budget> --
<python with transformers (and mlx-lm for --engine mlx)> tools/inkling_agent.py
CHECKPOINT --engine native|mlx --graff GRAFF --workspace DIR [options] -- [Graff arguments]

Like tools/graff_native.py (whose helpers this uses): an ephemeral-port adapter
(tools/inkling_adapter.py) the launcher owns, an ephemeral API key only in the
child environment, no global Graff configuration changed, owned children
stopped on exit. Graff reaches the adapter through its built-in `nanoattention`
provider (route id `flint`); the model behind it is Inkling 2 Mini.

--sandbox runs Graff under macOS sandbox-exec: files may be written only in the
workspace, temporary folders and Graff's own state folders; credentials folders
cannot be read; the only network is loopback. Graff's tool calls (a shell among
them) run inside it. The adapter and engine run outside.
"""
import argparse
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import tempfile

import graff_native as native


def sandbox_profile(workspace):
    home = Path.home()
    return "\n".join([
        "(version 1)",
        "(allow default)",
        "(deny file-write*)",
        '(allow file-write* (subpath "{}"))'.format(workspace),
        '(allow file-write* (subpath "/private/tmp") (subpath "/private/var/folders") (subpath "/dev"))',
        '(allow file-write* (subpath "{}") (subpath "{}") (subpath "{}") (subpath "{}"))'.format(
            home / ".graff", home / ".harness", home / ".codegraff", home / "Library/Caches"),
        "".join('(deny file-read* (subpath "{}"))'.format(home / sub) for sub in (
            ".ssh", ".aws", ".config", ".gnupg", ".netrc", "Library/Preferences/.wrangler", ".npmrc")),
        "(deny network-outbound)",
        '(allow network-outbound (remote ip "localhost:*"))',
        '(allow network-outbound (remote unix-socket))',
        ""])


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--engine", choices=("native", "mlx"), required=True)
    parser.add_argument("--binary", type=Path, help="nano, for --engine native")
    parser.add_argument("--mlx-python", type=Path, help="a Python with mlx-lm, for --engine mlx")
    parser.add_argument("--graff", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--sandbox", action="store_true")
    args = parser.parse_args(argv[:split])
    args.graff_args = argv[split + 1:]
    if not 0 < args.max_tokens < args.context:
        parser.error("require 0 < --max-tokens < --context")
    for name in ("checkpoint", "graff", "workspace"):
        setattr(args, name, getattr(args, name).resolve())
    if not args.workspace.is_dir() or not args.checkpoint.is_dir():
        parser.error("workspace and checkpoint must be existing directories")
    return args


def run(args):
    built_in = native.supports_builtin_flint(args.graff)
    lease = native.RouterLease(args.workspace)
    if not built_in:
        lease.check()
    server = agent = profile = None
    env = dict(os.environ, NANOATTENTION_API_KEY=secrets.token_urlsafe(32), GRAFF_CONTEXT=str(args.context),
               GRAFF_NO_TELEMETRY="1")
    previous = {}

    def interrupted(signum, frame):
        raise native.Interrupted(signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, interrupted)
        adapter = [sys.executable, "-B", str(Path(__file__).with_name("inkling_adapter.py")), str(args.checkpoint),
                   "--engine", args.engine, "--port", "0", "--context", str(args.context),
                   "--max-tokens", str(args.max_tokens), "--timeout", str(args.timeout)]
        if args.engine == "native":
            adapter += ["--binary", str(args.binary)]
        else:
            adapter += ["--mlx-python", str(args.mlx_python)]
        if args.record:
            adapter += ["--record", str(args.record)]
        server = subprocess.Popen(adapter, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        port = native.receive_ready(server, min(120, args.timeout), args.context)
        if built_in:
            env["GRAFF_NANOATTENTION_URL"] = "http://127.0.0.1:{}/v1/chat/completions".format(port)
        else:
            lease.acquire(port)
        command = [str(args.graff), "--model", "nanoattention/flint", "--no-subagent-tier", "--subagent-provider",
                   "nanoattention", "--subagent-model", "flint", "--no-telemetry"] + args.graff_args
        if args.sandbox:
            profile = tempfile.NamedTemporaryFile("w", suffix=".sb", delete=False)
            profile.write(sandbox_profile(args.workspace))
            profile.close()
            command = ["/usr/bin/sandbox-exec", "-f", profile.name] + command
        agent = subprocess.Popen(command, cwd=args.workspace, env=env, start_new_session=False)
        return agent.wait(timeout=args.timeout)
    finally:
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        try:
            try:
                native.stop_owned(agent)
            finally:
                try:
                    native.stop_owned(server)
                finally:
                    lease.release()
                    if profile is not None:
                        try:
                            os.unlink(profile.name)
                        except OSError:
                            pass
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def main(argv=None):
    try:
        return run(parse_args(argv))
    except native.Interrupted as error:
        return 128 + error.signum
    except KeyboardInterrupt:
        return 130
    except native.LaunchError as error:
        print("inkling_agent: " + str(error), file=sys.stderr)
        return 1
    except (OSError, subprocess.TimeoutExpired):
        print("inkling_agent: local process failed or exceeded its time budget", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
