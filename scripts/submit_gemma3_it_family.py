#!/usr/bin/env python3
"""
Submit only gemma3-it family jobs to SLURM.
Uses fp16_short configs for larger models (12B, 27B) to mitigate NaN issues.
"""

import subprocess
import sys
from pathlib import Path

# IT configs to submit (excluding PT variants)
IT_CONFIGS = [
    "gemma3_270m_it_full.yaml",
    "gemma3_1b_it_full.yaml",
    "gemma3_4b_it_full.yaml",
    "gemma3_12b_it_fp16_short.yaml",  # NaN mitigation
    "gemma3_27b_it_fp16_short.yaml",  # NaN mitigation
]

def main():
    project_root = Path(__file__).parent.parent

    print("Submitting gemma3-it family jobs...")

    # Create a temporary config directory with only IT configs
    temp_dir = project_root / "configs" / "tmp_it_only"
    temp_dir.mkdir(exist_ok=True)

    try:
        # Symlink only IT configs to temp directory
        for config in IT_CONFIGS:
            src = project_root / "configs" / config
            dst = temp_dir / config
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src)

        # Submit using submit_family.py with temp directory
        cmd = [
            "python", "scripts/submit_family.py",
            "--family", "gemma3",
            "--config-dir", str(temp_dir),
            "--project-root", str(project_root),
            "--template", "slurm/job_template.sh"
        ]

        subprocess.run(cmd, cwd=project_root, check=True)
        print("Done submitting gemma3-it family jobs!")

    finally:
        # Clean up temp directory
        if temp_dir.exists():
            for file in temp_dir.iterdir():
                file.unlink()
            temp_dir.rmdir()

if __name__ == "__main__":
    sys.exit(main())
