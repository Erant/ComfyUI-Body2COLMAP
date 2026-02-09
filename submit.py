#!/usr/bin/env python3
"""Submit Body2COLMAP workflows to ComfyUI.

Workflows must be exported in API format (not the default Save format).
In ComfyUI: Settings > enable Dev Mode, then File > Export (API Format).

The pipeline is defined as a YAML config with global settings and an
ordered list of steps:

    # pipeline.yaml
    settings:
      brush_path: /opt/brush/bin/brush

    steps:
      - workflow: segment.json
        input: source
        output: control

      - workflow: refine.json
        input: control
        output: refined

      - workflow: segment.json      # workflows can repeat
        input: refined
        output: final

Step arguments (input/output) are prefixed with the dataset name at
runtime, so for dataset_00001 "control" becomes "dataset_00001/control".

Usage:
    python submit.py pipeline.yaml dataset_00001
    python submit.py pipeline.yaml dataset_00001 dataset_00002
    python submit.py --server 192.168.1.100:8188 pipeline.yaml dataset_00001
"""

import argparse
import copy
import json
import os
import sys
import time
import urllib.error
import urllib.request

import yaml

# Node class_type -> input field name, split by role
INPUT_NODES = {
    "Body2COLMAP_LoadDataset": "directory",
}

OUTPUT_NODES = {
    "Body2COLMAP_SaveDataset": "output_directory",
    "Body2COLMAP_ExportCOLMAP": "output_directory",
}

# Global settings: config key -> (class_type, input field name)
GLOBAL_SETTINGS = {
    "brush_path": ("Body2COLMAP_RunBrush", "brush_path"),
}


def apply_settings(prompt, settings):
    """Apply global settings to matching nodes in a prompt.

    Modifies prompt in-place.
    """
    for key, value in settings.items():
        if key not in GLOBAL_SETTINGS:
            continue
        class_type, field = GLOBAL_SETTINGS[key]
        for node_id, node_def in prompt.items():
            if node_def.get("class_type") == class_type:
                node_def["inputs"][field] = value
                title = node_def.get("_meta", {}).get("title", class_type)
                print(f"    [{node_id}] {title}: {field} = {value!r}")


def patch_prompt(prompt, dataset, step_args, settings):
    """Apply step arguments and global settings to a workflow prompt.

    Prefixes input/output directory fields with the dataset name and
    applies any global settings to matching nodes.
    Modifies prompt in-place. Returns the number of nodes patched.
    """
    patched = 0
    input_dir = step_args.get("input")
    output_dir = step_args.get("output")

    for node_id, node_def in prompt.items():
        class_type = node_def.get("class_type")

        if class_type in INPUT_NODES and input_dir is not None:
            field = INPUT_NODES[class_type]
            value = os.path.join(dataset, input_dir)
            node_def["inputs"][field] = value
            title = node_def.get("_meta", {}).get("title", class_type)
            print(f"    [{node_id}] {title}: {field} = {value!r}")
            patched += 1

        if class_type in OUTPUT_NODES and output_dir is not None:
            field = OUTPUT_NODES[class_type]
            value = os.path.join(dataset, output_dir)
            node_def["inputs"][field] = value
            title = node_def.get("_meta", {}).get("title", class_type)
            print(f"    [{node_id}] {title}: {field} = {value!r}")
            patched += 1

    apply_settings(prompt, settings)

    return patched


def validate_api_format(prompt):
    """Check that the JSON looks like an API-format workflow."""
    if not isinstance(prompt, dict):
        return False
    sample = next(iter(prompt.values()), None)
    if not isinstance(sample, dict):
        return False
    return "class_type" in sample


def load_pipeline(config_path):
    """Load and validate a pipeline YAML config.

    Returns (settings, steps) where settings is a dict of global options
    and each step is a dict with at least a 'workflow' key plus any
    arguments like 'input' and 'output'.
    """
    base_dir = os.path.dirname(os.path.abspath(config_path))

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict) or "steps" not in config:
        print(
            "Error: Pipeline config must be a YAML mapping with a 'steps' key.",
            file=sys.stderr,
        )
        sys.exit(1)

    settings = config.get("settings", {})
    steps = config["steps"]

    if not isinstance(steps, list) or not steps:
        print("Error: 'steps' must be a non-empty array.", file=sys.stderr)
        sys.exit(1)

    # Warn on unrecognised settings
    for key in settings:
        if key not in GLOBAL_SETTINGS:
            print(
                f"Warning: Unknown setting {key!r} (known: {', '.join(GLOBAL_SETTINGS)})",
                file=sys.stderr,
            )

    # Load and validate each workflow, cache to avoid re-reading duplicates
    workflow_cache = {}
    for i, step in enumerate(steps):
        if not isinstance(step, dict) or "workflow" not in step:
            print(
                f"Error: Step {i + 1} must be a mapping with at least a 'workflow' key.",
                file=sys.stderr,
            )
            sys.exit(1)

        name = step["workflow"]
        if name not in workflow_cache:
            path = os.path.join(base_dir, name)
            try:
                with open(path) as f:
                    prompt = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error loading {path}: {e}", file=sys.stderr)
                sys.exit(1)

            if not validate_api_format(prompt):
                print(
                    f"Error: {name} doesn't look like an API-format workflow.\n"
                    "In ComfyUI, enable Dev Mode in Settings, then use\n"
                    "File > Export (API Format) to get the correct format.",
                    file=sys.stderr,
                )
                sys.exit(1)

            workflow_cache[name] = prompt

        step["_prompt"] = workflow_cache[name]

    return settings, steps


def queue_prompt(server, prompt):
    """Submit an API-format prompt to ComfyUI. Returns prompt_id."""
    payload = json.dumps({"prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(
        f"http://{server}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read())
    if "error" in result:
        raise RuntimeError(
            f"ComfyUI rejected prompt: {result['error']}\n"
            f"Node errors: {json.dumps(result.get('node_errors', {}), indent=2)}"
        )
    return result["prompt_id"]


def wait_for_completion(server, prompt_id, poll_interval=2.0):
    """Poll /history until the prompt appears (meaning it finished)."""
    while True:
        time.sleep(poll_interval)
        try:
            url = f"http://{server}/history/{prompt_id}"
            resp = urllib.request.urlopen(url)
            history = json.loads(resp.read())
        except urllib.error.URLError:
            print("!", end="", flush=True)
            continue

        if prompt_id in history:
            print(" done.")
            return history[prompt_id]
        print(".", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Submit Body2COLMAP workflows to ComfyUI",
        epilog=(
            "Workflows must be in API format. In ComfyUI, enable Dev Mode "
            "in Settings, then use File > Export (API Format)."
        ),
    )
    parser.add_argument(
        "pipeline",
        help="Path to pipeline YAML config",
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        help="Dataset name(s) to prefix onto node directories",
    )
    parser.add_argument(
        "--server",
        default="127.0.0.1:8188",
        help="ComfyUI server address (default: %(default)s)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between completion checks (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be submitted without actually sending",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit all workflows without waiting for completion",
    )
    args = parser.parse_args()

    # Load and validate pipeline up front
    settings, steps = load_pipeline(args.pipeline)
    if settings:
        print("Settings:")
        for key, val in settings.items():
            print(f"  {key}: {val}")
    print(f"Pipeline ({len(steps)} steps):")
    for i, step in enumerate(steps, 1):
        parts = [step["workflow"]]
        if "input" in step:
            parts.append(f"input={step['input']}")
        if "output" in step:
            parts.append(f"output={step['output']}")
        print(f"  {i}. {', '.join(parts)}")

    # Process each dataset through the full pipeline
    for di, dataset in enumerate(args.datasets, 1):
        print(f"\n{'='*60}")
        print(f"[{di}/{len(args.datasets)}] Dataset: {dataset}")
        print(f"{'='*60}")

        for si, step in enumerate(steps, 1):
            print(f"\n  Step {si}/{len(steps)}: {step['workflow']}")

            prompt = copy.deepcopy(step["_prompt"])
            patched = patch_prompt(prompt, dataset, step, settings)

            if patched == 0:
                print(
                    "  Warning: No nodes to patch in this workflow.",
                    file=sys.stderr,
                )

            if args.dry_run:
                print("    (dry run, not submitting)")
                continue

            # Submit
            try:
                prompt_id = queue_prompt(args.server, prompt)
            except urllib.error.URLError as e:
                print(
                    f"  Error connecting to ComfyUI at {args.server}: {e}",
                    file=sys.stderr,
                )
                sys.exit(1)
            except RuntimeError as e:
                print(f"  {e}", file=sys.stderr)
                continue

            print(f"    Queued: {prompt_id}")

            if args.no_wait:
                continue

            print(f"    Waiting for completion", end="", flush=True)

            result = wait_for_completion(
                args.server, prompt_id, args.poll_interval
            )

            # Report outputs
            outputs = result.get("outputs", {})
            for node_id, out in outputs.items():
                if out:
                    class_type = step["_prompt"].get(node_id, {}).get(
                        "class_type", "?"
                    )
                    print(f"    Output [{node_id}] ({class_type}):")
                    for key, val in out.items():
                        print(f"      {key}: {val}")

    action = "Would process" if args.dry_run else "Processed"
    print(
        f"\nDone. {action} {len(args.datasets)} dataset(s) "
        f"x {len(steps)} step(s)."
    )


if __name__ == "__main__":
    main()
