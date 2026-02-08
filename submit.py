#!/usr/bin/env python3
"""Submit Body2COLMAP workflows to ComfyUI.

Workflows must be exported in API format (not the default Save format).
In ComfyUI: Settings > enable Dev Mode, then File > Export (API Format).

Usage:
    python submit.py workflow_api.json my_dataset
    python submit.py workflow_api.json dir1 dir2 dir3
    python submit.py --server 192.168.1.100:8188 workflow_api.json my_dataset
"""

import argparse
import copy
import json
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
        "workflow",
        help="Path to workflow JSON file (API format)",
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

    # Load workflow template
    try:
        with open(args.workflow) as f:
            template = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading workflow: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate format
    if not validate_api_format(template):
        print(
            "Error: This doesn't look like an API-format workflow.\n"
            "In ComfyUI, enable Dev Mode in Settings, then use\n"
            "File > Export (API Format) to get the correct format.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check for relevant nodes
    nodes = find_directory_nodes(template)
    if not nodes:
        print(
            "Warning: No Body2COLMAP Load/Save/Export nodes found in workflow.",
            file=sys.stderr,
        )

    # Process each directory
    for i, directory in enumerate(args.directories, 1):
        print(f"\n[{i}/{len(args.directories)}] {directory}")

        prompt = copy.deepcopy(template)
        patched = patch_directory(prompt, directory)

        if patched == 0:
            print("  No directory nodes to patch, skipping.", file=sys.stderr)
            continue

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

        result = wait_for_completion(args.server, prompt_id, args.poll_interval)

        # Report outputs
        outputs = result.get("outputs", {})
        for node_id, out in outputs.items():
            if out:
                class_type = template.get(node_id, {}).get("class_type", "?")
                print(f"  Output [{node_id}] ({class_type}):")
                for key, val in out.items():
                    print(f"    {key}: {val}")

    action = "Would process" if args.dry_run else "Processed"
    print(f"\n{action} {len(args.directories)} workflow(s).")


if __name__ == "__main__":
    main()
