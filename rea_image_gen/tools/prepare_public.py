"""Export an allowlisted source snapshot; no credentials, SDK binaries or models."""
from pathlib import Path
import shutil,json,hashlib,re,subprocess
src=Path('/root/rea_image_gen');dst=Path('/root/rngd-model-forge')
if dst.exists(): raise SystemExit('Destination exists; refusing to overwrite')
dst.mkdir()
selected=[]
for name in ['run_image_gen_npu.py','web_app.py','verify_color_diversity.py','aurasr_upscaler.py','requirements-furiosa-image-gen.txt']:
 selected.append(src/name)
for name in ['compiler_engine','tests','tools','rust_kernels']:
 for p in (src/name).rglob('*'):
  if p.is_file() and p.suffix in {'.py','.sh','.rs','.toml','.md','.lock'} and not any(x in {'target','__pycache__','.git'} or x.startswith('._') for x in p.parts) and p.name!='prepare_public.py': selected.append(p)
for p in selected:
 rel=p.relative_to(src);out=dst/rel;out.parent.mkdir(parents=True,exist_ok=True)
 text=p.read_text()
 # Repository-relative paths; credentials are discovered from the user's home.
 text=text.replace('/root/rea_image_gen',str(dst)).replace('/root/.local/share/huggingface/token','~/.local/share/huggingface/token')
 text=text.replace('Path("~/.local/share/huggingface/token")','Path("~/.local/share/huggingface/token").expanduser()')
 if p.name=='web_app.py': text=text.replace('Path("/root/rngd-model-forge/web_outputs")','Path(__file__).resolve().parent / "web_outputs"')
 out.write_text(text)
 shutil.copymode(p,out)
shutil.copytree(src/'codex_skill_staging/rngd-model-porting',dst/'skills/rngd-model-porting',ignore=shutil.ignore_patterns('._*','__pycache__'))
(dst/'docs').mkdir()
for name in ['WEB_RUNBOOK.md','FOLLOWUP_R16.md']:
 p=src/'reports/rngd_porting'/name
 if p.exists(): (dst/'docs'/name).write_text(p.read_text().replace('/root/rea_image_gen','/root/rngd-model-forge').replace('/Users/dong-jin/Documents/GitHub/furiosa/.ssh/config','~/.ssh/config'))
(dst/'reports/rngd_porting').mkdir(parents=True)
shutil.copy2(src/'reports/rngd_porting/TCL_DEVELOPMENT_LEDGER.json',dst/'reports/rngd_porting/TCL_DEVELOPMENT_LEDGER.json')
(dst/'.gitignore').write_text('''__pycache__/
*.py[cod]
._*
.DS_Store
.env
.env.*
*token*.env
hf_token.env
.ssh/
*.pem
*.key
.cache/
.venv/
web_outputs/
reports/**
!reports/rngd_porting/
!reports/rngd_porting/TCL_DEVELOPMENT_LEDGER.json
*.fxb
*.so
*.safetensors
*.bin
*.pt
*.pth
*.log
rust_kernels/target/
''')
(dst/'README.md').write_text('''# RNGD Model Forge

A practical PyTorch model-porting toolkit for Furiosa RNGD: discover model features, reproduce unsupported operations, implement public TCL kernels, validate real NPU execution, and serve a persistent SDXL Turbo pipeline.

**Status: experimental.** SDXL Turbo 512×512, 1 step produces images with official RNGD Task traces. Warm inference measured about 2.25 seconds in the development environment. BF16 numerical discrepancies and complete CPU event attribution remain open. These timings do not promise performance on other machines or prove every model operation is NPU-only.

## Requirements

Linux RNGD server, Python3.10+, an already configured official Furiosa SDK/driver. Tested packages are pinned in requirements-furiosa-image-gen.txt. Install SDK packages only through Furiosa's official distribution; this repository does not redistribute SDK libraries, model weights or firmware. Do not upgrade an existing working SDK blindly. Model licenses and Hugging Face access requirements apply independently.

## Quick start

```bash
cd /root/rngd-model-forge
# Use an existing compatible environment; inspect the pinned requirements first.
# Token: HF_TOKEN environment variable or ~/.local/share/huggingface/token
# Existing model cache is retained; no HF_HOME override is required.
bash tests/run_sdxl_tcl_warm.sh reports/rngd_porting/runs/manual_test
```

This command compiles, generates an image with profiling, and measures three warm forwards in the same process. Cold preparation takes minutes. Images, metrics, official traces and summaries go to the selected run directory. Do not run NPU tests concurrently on a fusion=8 card.

## Persistent web UI

```bash
bash tests/start_sdxl_web.sh
```

The first startup loads and compiles SDXL and generates a startup image before opening http://127.0.0.1:7860. Subsequent requests reuse the loaded pipeline. From your laptop create a tunnel with your own SSH configuration:

```bash
ssh -N -L 127.0.0.1:7860:127.0.0.1:7860 YOUR_RNGD_SSH_HOST
```

Web requests are queued with concurrency1. Strict NPU model fallback rejection is enabled. CPU scheduler and host control work remain intentional. SDXL is the available model choice; experimental candidates are not presented as supported models. Each generated PNG has a neighboring .metrics.json report.

## Operator-first workflow

1. Extract operator/shape/dtype/layout features through framework graphs, never checkpoint binary inspection.
2. Follow official SDK → public decomposition → existing/new TCL → public Rust/vISA with a verified Python bridge.
3. Compare small CPU references with real NPU Task traces, then integrate blocks and the full model.
4. Group numerical failures by implementation family; violating tensor elements are not independent bug counts.
5. Record failures, measured work time, test attempts, edits and reproducible commands. Missing evidence stays unknown.

REA and proprietary binary reverse engineering are outside this project's executable workflow.

## Layout

- compiler_engine/: compiler adapter, graph transforms, generic TCL windows, execution auditor and work metrics.
- tests/: unit tests, real-shape feature tests, SDXL integration and web acceptance.
- tools/: SMI monitor, measured development work wrapper.
- rust_kernels/: experimental public Rust/vISA bridge experiments; not a universal supported backend.
- skills/rngd-model-porting/: reusable Codex skill and reference workflow.
- docs/: runbook and honest current limitations.

## Telemetry and development history

```bash
python3.10 tools/npu_watch.py --interval 0.5 --duration 120 --output reports/npu_watch/example
python3.10 tools/record_porting_work.py --kind test --feature window_pack -- python3.10 tests/run_tcl_windows_feature.py
```

SMI memory can report zero during occupied execution; retain raw telemetry and use official Task trace for execution evidence. Time ranges can overlap; do not sum nested compile/preparation and inference times. Historical unrecorded development times/counts are null rather than guessed.

## Public release

This is a source snapshot prepared for review, not an uploaded GitHub repository. See RELEASE_CHECKLIST.md. A project license must be selected by the rights holder before claiming an open-source license. Third-party SDKs/models retain their own licenses.
''')
(dst/'RELEASE_CHECKLIST.md').write_text('''# Release checklist

- [x] Source-only allowlist export; no tokens, SSH keys, model checkpoints, SDK binaries, compiled artifacts or raw traces.
- [x] Explicit experimental status and unresolved numerical/CPU attribution gates.
- [ ] Rights holder selects project LICENSE and copyright attribution; no license is inferred.
- [ ] Review first-party and third-party provenance, especially any Rust examples.
- [ ] Run compatibility tests on the intended release environment; exported source syntax is checked separately from NPU acceptance.
- [ ] Review repository diff and scan for secrets again before uploading.
- [ ] Create the intended GitHub repository and upload only after authorization.

Never commit access tokens or examples containing credentials. Use your own SSH host and Hugging Face credentials.
''')
# Bounded secret pattern scan; print file/line only, never matching values.
issues=[]
for p in dst.rglob('*'):
 if p.is_file():
  for i,line in enumerate(p.read_text().splitlines(),1):
   if re.search(r'hf_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}',line):issues.append({'file':str(p.relative_to(dst)),'line':i})
if issues: raise SystemExit(json.dumps({'secret_scan_failed':issues}))
manifest=[{'path':str(p.relative_to(dst)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in dst.rglob('*') if p.is_file()]
(dst/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
subprocess.run(['git','init','-b','main',str(dst)],check=True,capture_output=True)
subprocess.run(['python3.10','-m','compileall','-q',str(dst/'compiler_engine'),str(dst/'tests'),str(dst/'tools'),str(dst/'web_app.py'),str(dst/'run_image_gen_npu.py')],check=True)
print(json.dumps({'path':str(dst),'files':len(manifest),'source_bytes':sum(r['bytes'] for r in manifest),'secret_pattern_hits':0,'syntax':'passed','github_upload':False}))
