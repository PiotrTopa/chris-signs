#!/usr/bin/env python3
"""List every GLOBAL referenced by the pickles inside torch .pt zip files, without loading them.

usage: scan_pt.py FILE.pt [...]   (exit 1 if any global is outside the allowlist)
Run before torch.load / YOLO(...) touches an untrusted .pt. The pinned MTSD files import only
torch.*, ultralytics.*, collections.*, dill._dill._load_type and builtins (docs/SIGN_MODELS.md).
"""
import io
import pickletools
import sys
import zipfile

OK = ('torch', 'ultralytics', 'collections', 'numpy', 'builtins.set', 'builtins.slice', 'builtins.getattr',
      'builtins.dict', 'builtins.list', 'builtins.tuple', 'builtins.int', 'builtins.float', '_codecs.encode',
      'datetime', 'pathlib', 'functools.partial', 'types.SimpleNamespace', 'argparse.Namespace',
      '__builtin__.set', 'copyreg._reconstructor', 'builtins.object', 're._compile', 'typing', 'enum',
      'dill._dill._load_type')
rc = 0
for path in sys.argv[1:]:
    z = zipfile.ZipFile(path)
    globs = set()
    for n in z.namelist():
        if n.endswith('.pkl'):
            data = z.read(n)
            strs = []
            for op, arg, pos in pickletools.genops(io.BytesIO(data)):
                if op.name in ('SHORT_BINUNICODE', 'BINUNICODE', 'UNICODE', 'STRING', 'BINSTRING', 'SHORT_BINSTRING'):
                    strs.append(arg)
                elif op.name == 'GLOBAL':
                    globs.add(arg.replace(' ', '.'))
                elif op.name == 'STACK_GLOBAL':
                    globs.add(strs[-2] + '.' + strs[-1])
    bad = sorted(g for g in globs if not g.startswith(OK))
    print(path, 'globals:', len(globs), 'NON-ALLOWLISTED:', bad)
    if bad:
        rc = 1
sys.exit(rc)
