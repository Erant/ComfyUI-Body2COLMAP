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
        paths:
          source: source
          control: control

      - workflow: refine.json
        paths:
          control: control       # Load from previous step's output
          masks: segmentation
          refined: refined

      - workflow: segment.json   # workflows can repeat
        paths:
          source: refined
          control: final

The 'paths' dict maps directory values found in the workflow JSON
to subdirectory names. Each is prefixed with the dataset name at
runtime, so for dataset_00001 "control" becomes "dataset_00001/control".

Usage:
    python submit.py pipeline.yaml dataset_00001
    python submit.py pipeline.yaml dataset_00001 dataset_00002
    python submit.py --server 192.168.1.100:8188 pipeline.yaml dataset_00001
"""

import argparse
import copy
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.request

import yaml

# class_type -> input field that holds a directory path
DIRECTORY_FIELDS = {
    "Body2COLMAP_LoadDataset": "directory",
    "Body2COLMAP_SaveDataset": "output_directory",
    "Body2COLMAP_ExportCOLMAP": "output_directory",
}

# Keys in a step that are directives, not node field overrides
STEP_KEYS = {"workflow", "paths", "fixup_models", "_prompt"}


def apply_settings(prompt, settings):
    """Apply global settings to nodes by matching input field names.

    Any setting key that matches an input field name on a node will
    override that field's value. Modifies prompt in-place.
    """
    for node_id, node_def in prompt.items():
        inputs = node_def.get("inputs", {})
        for key, value in settings.items():
            if key in inputs and not isinstance(inputs[key], list):
                inputs[key] = value
                title = node_def.get("_meta", {}).get("title", node_def.get("class_type", "?"))
                print(f"    [{node_id}] {title}: {key} = {value!r}")


def patch_prompt(prompt, dataset, step_args, settings):
    """Apply path mappings and global settings to a workflow prompt.

    For each Load/Save/Export Dataset node, reads its current directory
    value from the workflow JSON and uses it as a key into the step's
    'paths' dict. The mapped value is then prefixed with the dataset
    name to produce the final directory.

    Modifies prompt in-place. Returns the number of directory nodes patched.
    """
    paths = step_args.get("paths", {})
    patched = 0

    for node_id, node_def in prompt.items():
        class_type = node_def.get("class_type")
        if class_type not in DIRECTORY_FIELDS:
            continue

        field = DIRECTORY_FIELDS[class_type]
        current = node_def["inputs"].get(field, "")
        title = node_def.get("_meta", {}).get("title", class_type)

        # Skip if the field is a node connection (placeholder or otherwise)
        if isinstance(current, list):
            continue

        if current not in paths:
            print(
                f"    [{node_id}] {title}: {field} = {current!r} "
                f"(not in paths, skipping)",
                file=sys.stderr,
            )
            continue

        value = os.path.join(dataset, paths[current])
        node_def["inputs"][field] = value
        print(f"    [{node_id}] {title}: {field} = {current!r} -> {value!r}")
        patched += 1

    # Merge settings: step-level overrides take precedence over globals
    step_overrides = {k: v for k, v in step_args.items() if k not in STEP_KEYS}
    merged = {**settings, **step_overrides}
    apply_settings(prompt, merged)

    return patched


def validate_api_format(prompt):
    """Check that the JSON looks like an API-format workflow."""
    if not isinstance(prompt, dict):
        return False
    sample = next(iter(prompt.values()), None)
    if not isinstance(sample, dict):
        return False
    return "class_type" in sample


def load_pipeline(config_path, workflow_dir=None):
    """Load and validate a pipeline YAML config.

    Returns (settings, steps) where settings is a dict of global options
    and each step is a dict with at least a 'workflow' key plus a 'paths'
    mapping and optional field overrides.

    Workflow JSON files are resolved relative to workflow_dir, which
    defaults to 'workflows/api/' next to the config file.
    """
    base_dir = os.path.dirname(os.path.abspath(config_path))
    if workflow_dir is None:
        workflow_dir = os.path.join(base_dir, "workflows", "api")
    workflow_dir = os.path.abspath(workflow_dir)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict) or "steps" not in config:
        raise ValueError(
            "Pipeline config must be a YAML mapping with a 'steps' key."
        )

    settings = config.get("settings", {})
    steps = config["steps"]

    if not isinstance(steps, list) or not steps:
        raise ValueError("'steps' must be a non-empty array.")

    # Load and validate each workflow, cache to avoid re-reading duplicates
    workflow_cache = {}
    for i, step in enumerate(steps):
        if not isinstance(step, dict) or "workflow" not in step:
            raise ValueError(
                f"Step {i + 1} must be a mapping with at least a 'workflow' key."
            )

        name = step["workflow"]
        if name not in workflow_cache:
            path = os.path.join(workflow_dir, name)
            try:
                with open(path, encoding="utf-8") as f:
                    prompt = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                raise ValueError(f"Error loading {path}: {e}") from e

            if not validate_api_format(prompt):
                raise ValueError(
                    f"{name} doesn't look like an API-format workflow.\n"
                    "In ComfyUI, enable Dev Mode in Settings, then use\n"
                    "File > Export (API Format) to get the correct format."
                )

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


def get_server_address():
    """Get the ComfyUI server address from CLI args (when running as a node)."""
    try:
        import comfy.cli_args

        host = comfy.cli_args.args.listen
        port = comfy.cli_args.args.port
    except (ImportError, AttributeError):
        raise RuntimeError(
            "Cannot auto-detect server address outside of ComfyUI. "
            "Use the --server argument when running from the command line."
        )
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    return f"{host}:{port}"


def expand_datasets(raw_lines):
    """Expand dataset lines, supporting trailing wildcards.

    Lines whose last path component contains a ``*`` are glob-expanded and
    filtered to directories only.  Other lines are passed through as-is.

    Example::

        datasets/*          -> all directories inside datasets/
        datasets/batch_*    -> directories matching the prefix
        dataset_00001       -> passed through verbatim
    """
    result = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        if "*" in os.path.basename(line):
            matches = sorted(glob.glob(line))
            result.extend(m for m in matches if os.path.isdir(m))
        else:
            result.append(line)
    return result


def extract_ancestor_subgraph(prompt, node_id, input_name):
    """Extract the full ancestor subgraph feeding into a node's input.

    Walks backwards from the node connected to ``input_name`` on
    ``node_id``, collecting every ancestor node via BFS.

    Returns (subgraph, root_node_id, root_output_index) where *subgraph*
    is ``{node_id: node_def}`` for all ancestor nodes.  Returns
    ``(None, None, None)`` if the input is not connected.
    """
    node_def = prompt.get(str(node_id))
    if node_def is None:
        return None, None, None

    connection = node_def.get("inputs", {}).get(input_name)
    if not isinstance(connection, list) or len(connection) != 2:
        return None, None, None

    root_node_id = str(connection[0])
    root_output_index = connection[1]

    # BFS backwards through the graph
    subgraph = {}
    queue = [root_node_id]
    visited = set()

    while queue:
        nid = queue.pop(0)
        if nid in visited:
            continue
        visited.add(nid)

        ndef = prompt.get(nid)
        if ndef is None:
            continue

        subgraph[nid] = copy.deepcopy(ndef)

        for val in ndef.get("inputs", {}).values():
            if (
                isinstance(val, list)
                and len(val) == 2
                and isinstance(val[0], str)
                and isinstance(val[1], int)
            ):
                queue.append(val[0])

    return subgraph, root_node_id, root_output_index


def _is_connection(value):
    """Return True if *value* looks like an API-format node connection."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


def fixup_placeholders(target_workflow, prompt, unique_id, settings=None):
    """Replace Body2COLMAP_Placeholder nodes in *target_workflow*.

    Each Placeholder has a ``key`` that is resolved against two sources
    (Composer node inputs take precedence over *settings*):

    * **Connection inputs** (e.g. ``model_high`` wired to a checkpoint
      loader chain): the full ancestor subgraph is extracted, remapped to
      avoid ID collisions, injected into *target_workflow*, and every
      consumer of the placeholder is rewired to the subgraph root.

    * **Literal inputs** (e.g. ``brush_path = "brush"``, or a value from
      the pipeline YAML settings): every consumer of the placeholder has
      its reference replaced with the literal value.

    Modifies *target_workflow* in-place.
    """
    # Find all placeholder nodes
    placeholders = {}
    for nid, ndef in list(target_workflow.items()):
        if ndef.get("class_type") == "Body2COLMAP_Placeholder":
            key = ndef["inputs"]["key"]
            placeholders[nid] = key

    if not placeholders:
        return

    # Look up the Composer node's inputs in the live prompt
    composer = prompt.get(str(unique_id))
    if composer is None:
        raise ValueError(
            f"WorkflowComposer node {unique_id} not found in prompt"
        )
    composer_inputs = composer.get("inputs", {})

    # Build merged lookup: pipeline settings as base, Composer inputs win
    lookup = dict(settings) if settings else {}
    lookup.update(composer_inputs)

    # Classify each placeholder as subgraph (connection) or literal
    subgraph_phs = {}   # placeholder_id -> key
    literal_phs = {}    # placeholder_id -> (key, value)

    for placeholder_id, key in placeholders.items():
        if key not in lookup:
            raise ValueError(
                f"Placeholder '{key}' has no matching input on the "
                f"WorkflowComposer node or in the pipeline settings."
            )
        value = lookup[key]
        if _is_connection(value):
            subgraph_phs[placeholder_id] = key
        else:
            literal_phs[placeholder_id] = (key, value)

    # ---- Subgraph placeholders: extract, remap, inject, rewire ----------

    if subgraph_phs:
        merged_subgraph = {}
        roots = {}  # placeholder_id -> (root_node_id, root_output_index)

        for placeholder_id, key in subgraph_phs.items():
            subgraph, root_id, root_slot = extract_ancestor_subgraph(
                prompt, unique_id, key
            )
            if subgraph is None:
                raise ValueError(
                    f"Placeholder '{key}' is connected on the Composer but "
                    f"the ancestor subgraph could not be extracted."
                )
            merged_subgraph.update(subgraph)
            roots[placeholder_id] = (root_id, root_slot)

        # Compute ID offset to avoid collisions
        existing_int_ids = [int(nid) for nid in target_workflow if nid.isdigit()]
        offset = max(existing_int_ids, default=0) + 1

        id_map = {}
        for old_id in merged_subgraph:
            id_map[old_id] = (
                str(int(old_id) + offset) if old_id.isdigit()
                else f"{old_id}_{offset}"
            )

        # Inject remapped subgraph nodes
        for old_id, ndef in merged_subgraph.items():
            new_def = copy.deepcopy(ndef)
            for _key, val in new_def.get("inputs", {}).items():
                if _is_connection(val) and val[0] in id_map:
                    val[0] = id_map[val[0]]
            target_workflow[id_map[old_id]] = new_def

        # Rewire consumers, then remove placeholders
        for placeholder_id, (root_id, root_slot) in roots.items():
            remapped_root = id_map[root_id]
            for nid, ndef in target_workflow.items():
                if nid == placeholder_id:
                    continue
                for _key, val in ndef.get("inputs", {}).items():
                    if _is_connection(val) and str(val[0]) == placeholder_id:
                        val[0] = remapped_root
                        val[1] = root_slot
            del target_workflow[placeholder_id]

    # ---- Literal placeholders: replace references with the value --------

    for placeholder_id, (key, value) in literal_phs.items():
        for nid, ndef in target_workflow.items():
            if nid == placeholder_id:
                continue
            inputs = ndef.get("inputs", {})
            for input_key, input_val in inputs.items():
                if _is_connection(input_val) and str(input_val[0]) == placeholder_id:
                    inputs[input_key] = value
        del target_workflow[placeholder_id]


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
        "--workflow-dir",
        default=None,
        help="Directory containing workflow JSON files "
        "(default: workflows/api/ next to the pipeline YAML)",
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
    try:
        settings, steps = load_pipeline(args.pipeline, args.workflow_dir)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    if settings:
        print("Settings:")
        for key, val in settings.items():
            print(f"  {key}: {val}")
    print(f"Pipeline ({len(steps)} steps):")
    for i, step in enumerate(steps, 1):
        paths = step.get("paths", {})
        path_str = " ".join(f"{k}->{v}" for k, v in paths.items())
        print(f"  {i}. {step['workflow']}  {path_str}")

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
