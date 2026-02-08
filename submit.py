#!/usr/bin/env python3
"""Submit Body2COLMAP workflows to ComfyUI.

Workflows must be exported in API format (not the default Save format).
In ComfyUI: Settings > enable Dev Mode, then File > Export (API Format).

The workflow directory should contain numbered JSON files that will be
executed in sorted order for each dataset:

    workflows/
        1_load_and_segment.json
        2_refine_masks.json
        3_export_colmap.json

Usage:
    python submit.py workflows/ my_dataset
    python submit.py workflows/ dir1 dir2 dir3
    python submit.py --server 192.168.1.100:8188 workflows/ my_dataset
"""

import argparse
import copy
import json
import os
import sys
import time
import urllib.error
import urllib.request

# Node class_type -> input field name for the directory parameter
DIRECTORY_NODES = {
    "Body2COLMAP_LoadDataset": "directory",
    "Body2COLMAP_SaveDataset": "output_directory",
    "Body2COLMAP_ExportCOLMAP": "output_directory",
}


def find_directory_nodes(prompt):
    """Find all nodes whose directory fields should be patched.

    Returns list of (node_id, class_type, field_name) tuples.
    """
    results = []
    for node_id, node_def in prompt.items():
        class_type = node_def.get("class_type")
        if class_type in DIRECTORY_NODES:
            results.append((node_id, class_type, DIRECTORY_NODES[class_type]))
    return results


def patch_directory(prompt, directory):
    """Replace directory fields in all Load/Save/Export Dataset nodes.

    Modifies prompt in-place. Returns the number of nodes patched.
    """
    nodes = find_directory_nodes(prompt)
    for node_id, class_type, field in nodes:
        old = prompt[node_id]["inputs"].get(field)
        prompt[node_id]["inputs"][field] = directory
        title = prompt[node_id].get("_meta", {}).get("title", class_type)
        print(f"  [{node_id}] {title}: {field} = {old!r} -> {directory!r}")
    return len(nodes)


def validate_api_format(prompt):
    """Check that the JSON looks like an API-format workflow."""
    if not isinstance(prompt, dict):
        return False
    sample = next(iter(prompt.values()), None)
    if not isinstance(sample, dict):
        return False
    return "class_type" in sample


def load_workflows(workflow_dir):
    """Load all .json workflows from a directory, sorted by filename.

    Returns list of (filename, prompt_dict) tuples.
    """
    files = sorted(
        f for f in os.listdir(workflow_dir) if f.endswith(".json")
    )
    if not files:
        print(f"Error: No .json files found in {workflow_dir}", file=sys.stderr)
        sys.exit(1)

    workflows = []
    for filename in files:
        path = os.path.join(workflow_dir, filename)
        try:
            with open(path) as f:
                prompt = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading {path}: {e}", file=sys.stderr)
            sys.exit(1)

        if not validate_api_format(prompt):
            print(
                f"Error: {filename} doesn't look like an API-format workflow.\n"
                "In ComfyUI, enable Dev Mode in Settings, then use\n"
                "File > Export (API Format) to get the correct format.",
                file=sys.stderr,
            )
            sys.exit(1)

        workflows.append((filename, prompt))

    return workflows


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
        "workflow_dir",
        help="Directory containing workflow JSON files (API format), "
        "executed in sorted order (e.g. 1_load.json, 2_refine.json)",
    )
    parser.add_argument(
        "directories",
        nargs="+",
        help="Dataset directory name(s) to substitute into Load/Save nodes",
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
    args = parser.parse_args()

    # Load and validate all workflows up front
    workflows = load_workflows(args.workflow_dir)
    print(f"Loaded {len(workflows)} workflow(s) from {args.workflow_dir}:")
    for filename, _ in workflows:
        print(f"  {filename}")

    # Process each directory through the full workflow sequence
    for i, directory in enumerate(args.directories, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(args.directories)}] Dataset: {directory}")
        print(f"{'='*60}")

        for step, (filename, template) in enumerate(workflows, 1):
            print(f"\n  Step {step}/{len(workflows)}: {filename}")

            prompt = copy.deepcopy(template)
            patched = patch_directory(prompt, directory)

            if patched == 0:
                print(
                    "  Warning: No directory nodes to patch in this workflow.",
                    file=sys.stderr,
                )

            if args.dry_run:
                print("  (dry run, not submitting)")
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

            print(f"  Queued: {prompt_id}")
            print(f"  Waiting for completion", end="", flush=True)

            result = wait_for_completion(
                args.server, prompt_id, args.poll_interval
            )

            # Report outputs
            outputs = result.get("outputs", {})
            for node_id, out in outputs.items():
                if out:
                    class_type = template.get(node_id, {}).get("class_type", "?")
                    print(f"  Output [{node_id}] ({class_type}):")
                    for key, val in out.items():
                        print(f"    {key}: {val}")

    action = "Would process" if args.dry_run else "Processed"
    print(
        f"\nDone. {action} {len(args.directories)} dataset(s) "
        f"x {len(workflows)} workflow(s)."
    )


if __name__ == "__main__":
    main()
