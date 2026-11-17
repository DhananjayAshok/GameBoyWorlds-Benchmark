"""
Build-time only: install prime-agent's bundled Python skills into the kernel venv, and record
them in its .bootstrap-version the way prime-agent itself does.

    python seed_kernel_skills.py <venv> <skills.json> <runtime source dir>

The installer's kernel pre-warm (`prime-agent --prime-agent-bootstrap`) builds the venv with no
skills. prime-agent installs them on the first session's first tool call instead -- over the
network, and outside the image's version pins. So the image does it here, with the same command
prime-agent runs (pa-core kernel/bootstrap/venv.rs, sync_python_skills) under UV_CONSTRAINT, and
writes the same records (venv/skills.rs, to_bootstrap_skill): a session then finds every skill
covered and installs nothing.

skills.json lists import_name and package_path per skill, copied from a .bootstrap-version that
prime-agent 0.10.3 wrote for these bundled skills. The hashes are recomputed from the files here.
If a prime-agent upgrade changes the skill set, the first session reinstalls and says so --
verify_sandbox.sh's end-to-end check looks for exactly that.
"""
import hashlib
import json
import os
import subprocess
import sys


def pyproject_hash(path: str) -> str:
    with open(path, "rb") as handle:
        return "sha256:" + hashlib.sha256(handle.read()).hexdigest()


def main() -> None:
    venv, skills_file, runtime_source = sys.argv[1:4]
    python = os.path.join(venv, "bin", "python")
    with open(skills_file) as handle:
        skills = json.load(handle)

    command = ["uv", "--no-config", "pip", "install", "--python", python, runtime_source]
    for skill in skills:
        command += ["--editable", skill["package_path"]]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)

    records = []
    for skill in skills:
        pyproject = os.path.join(skill["package_path"], "pyproject.toml")
        if not os.path.isfile(pyproject):
            sys.exit(f"bundled skill has no pyproject.toml: {pyproject}")
        records.append({"import_name": skill["import_name"], "package_path": skill["package_path"],
                        "pyproject_path": pyproject, "pyproject_hash": pyproject_hash(pyproject)})
    records.sort(key=lambda r: (r["package_path"], r["import_name"]))

    for record in records:
        subprocess.run([python, "-c", f"import {record['import_name']}"], check=True)

    version_path = os.path.join(venv, ".bootstrap-version")
    with open(version_path) as handle:
        version = json.load(handle)
    version["python_skills"] = records
    with open(version_path, "w") as handle:
        handle.write(json.dumps(version, separators=(",", ":")) + "\n")
    print(f"recorded {len(records)} skills in {version_path}", flush=True)


if __name__ == "__main__":
    main()
