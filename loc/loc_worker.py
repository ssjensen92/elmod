#!/usr/bin/env python
import contextlib
import io
import json
import os
import runpy
import sys
import traceback

os.environ.setdefault("PYOPENCL_NO_CACHE", "1")

INSTALL_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(INSTALL_DIR)

import LOC_aux


_orig_initcl = LOC_aux.InitCL
_orig_program = LOC_aux.cl.Program
_context_cache = {}
_program_cache = {}


def _verbose():
    return os.environ.get("ELMOD_VERBOSE", "").lower() in ("1", "true", "yes", "on")


def _cache_key(value):
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def cached_initcl(GPU=0, platforms=[], idevice=0, sub=0, verbose=True):
    key = (GPU, _cache_key(platforms), idevice, sub)
    if key not in _context_cache:
        _context_cache[key] = _orig_initcl(GPU, platforms, idevice, sub, verbose)
    elif verbose and _verbose():
        platform, device, context, queue, mf = _context_cache[key]
        print("  Reusing OpenCL context")
        print("  Platform: ", platform)
        print("  Device:   ", device)
    return _context_cache[key]


class CachedProgram:
    def __init__(self, context, source):
        self.context = context
        self.source = source

    def build(self, options=None, devices=None, cache_dir=None):
        key = (id(self.context), self.source, options, _cache_key(devices))
        if key not in _program_cache:
            program = _orig_program(self.context, self.source)
            _program_cache[key] = program.build(options, devices, cache_dir)
        elif _verbose():
            print("  Reusing compiled OpenCL program")
        return _program_cache[key]


LOC_aux.InitCL = cached_initcl
LOC_aux.cl.Program = CachedProgram


def run_loc1d(ini_file):
    old_argv = sys.argv[:]
    old_cwd = os.getcwd()
    ini_file = os.path.abspath(ini_file)
    sys.argv = ["LOC1D.py", ini_file]
    try:
        os.chdir(os.path.dirname(ini_file))
        runpy.run_path(os.path.join(INSTALL_DIR, "LOC1D.py"), run_name="__main__")
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv


def respond(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    try:
        request = json.loads(line)
        if request.get("cmd") == "stop":
            respond({"ok": True})
            break
        ini_file = request["ini_file"]
        capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                run_loc1d(ini_file)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            if code != 0:
                raise RuntimeError(
                    "LOC1D.py exited with code %s\n%s" %
                    (code, capture.getvalue()))
            respond({"ok": True})
        except Exception:
            raise RuntimeError(capture.getvalue() + traceback.format_exc())
        else:
            respond({"ok": True})
    except Exception:
        respond({"ok": False, "output": traceback.format_exc()})
