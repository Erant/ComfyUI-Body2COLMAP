"""WorkflowComposer node - submits a pipeline of workflows to ComfyUI.

Loads a YAML pipeline config, patches directory paths for each dataset,
optionally injects model subgraphs from the current workflow into steps
that use Body2COLMAP_Placeholder nodes, and queues all workflows for
execution.

Workflows are submitted without waiting for completion to avoid deadlocking
the ComfyUI execution queue.
"""

import copy
import logging
import os
import time

from ..submit import (
    STEP_KEYS,
    expand_datasets,
    fixup_placeholders,
    get_server_address,
    load_pipeline,
    patch_prompt,
    queue_prompt,
)

logger = logging.getLogger(__name__)

_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PIPELINE_DIR = os.path.join(_PACKAGE_ROOT, "workflows", "pipeline")
_WORKFLOW_DIR = os.path.join(_PACKAGE_ROOT, "workflows", "api")


def _list_pipelines():
    """Enumerate pipeline YAML files, returning display names without extension."""
    if not os.path.isdir(_PIPELINE_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(_PIPELINE_DIR)
        if f.endswith((".yaml", ".yml"))
    )


class Body2COLMAP_WorkflowComposer:
    """Submit a pipeline of workflows to ComfyUI."""

    CATEGORY = "Body2COLMAP"
    FUNCTION = "submit"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    @classmethod
    def INPUT_TYPES(cls):
        pipelines = _list_pipelines()
        return {
            "required": {
                "pipeline": (pipelines, {
                    "tooltip": "Pipeline config from workflows/pipeline/",
                }),
                "datasets": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "Dataset paths, one per line. Trailing wildcards are "
                        "expanded to matching directories (e.g. datasets/*)"
                    ),
                }),
                "brush_path": ("STRING", {
                    "default": "brush",
                    "tooltip": (
                        "Path to the brush executable (or 'brush' if in PATH)"
                    ),
                }),
                "width": ("INT", {
                    "default": 720,
                    "min": 1,
                    "max": 8192,
                    "step": 16,
                    "tooltip": "Output width for workflow nodes with a width input",
                }),
                "height": ("INT", {
                    "default": 1280,
                    "min": 1,
                    "max": 8192,
                    "step": 16,
                    "tooltip": "Output height for workflow nodes with a height input",
                }),
                "model_high": ("MODEL", {
                    "tooltip": (
                        "High-detail model chain. Not used directly - its "
                        "ancestor subgraph is injected into workflow steps "
                        "that contain a Placeholder with key 'model_high'."
                    ),
                }),
                "model_low": ("MODEL", {
                    "tooltip": (
                        "Low-detail model chain. Not used directly - its "
                        "ancestor subgraph is injected into workflow steps "
                        "that contain a Placeholder with key 'model_low'."
                    ),
                }),
                "clip": ("CLIP", {
                    "tooltip": (
                        "CLIP model. Not used directly - its ancestor "
                        "subgraph is injected into workflow steps that "
                        "contain a Placeholder with key 'clip'."
                    ),
                }),
                "vae": ("VAE", {
                    "tooltip": (
                        "VAE model. Not used directly - its ancestor "
                        "subgraph is injected into workflow steps that "
                        "contain a Placeholder with key 'vae'."
                    ),
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute: pipeline config or workflow JSONs may have
        # changed on disk since the last run.
        return time.time()

    def submit(
        self,
        pipeline,
        datasets,
        brush_path,
        width,
        height,
        model_high,
        model_low,
        clip,
        vae,
        prompt=None,
        unique_id=None,
    ):
        server = get_server_address()

        # Resolve dropdown name (no extension) back to the YAML file on disk.
        # Try .yaml first, then .yml.
        pipeline_path = os.path.join(_PIPELINE_DIR, pipeline + ".yaml")
        if not os.path.isfile(pipeline_path):
            pipeline_path = os.path.join(_PIPELINE_DIR, pipeline + ".yml")

        settings, steps = load_pipeline(pipeline_path, workflow_dir=_WORKFLOW_DIR)

        raw_lines = datasets.strip().splitlines()
        dataset_list = expand_datasets(raw_lines)
        if not dataset_list:
            raise ValueError("No datasets specified (or wildcard matched nothing)")

        prompt_ids = []
        total = len(dataset_list) * len(steps)
        submitted = 0

        for di, dataset in enumerate(dataset_list, 1):
            logger.info(
                "[WorkflowComposer] [%d/%d] Dataset: %s",
                di, len(dataset_list), dataset,
            )

            for si, step in enumerate(steps, 1):
                workflow = copy.deepcopy(step["_prompt"])
                patch_prompt(workflow, dataset, step, settings)

                if prompt is not None:
                    # Merge global settings with step-level overrides so
                    # placeholders can resolve keys from the pipeline YAML.
                    step_overrides = {
                        k: v for k, v in step.items() if k not in STEP_KEYS
                    }
                    merged = {**settings, **step_overrides}
                    fixup_placeholders(workflow, prompt, unique_id, settings=merged)

                pid = queue_prompt(server, workflow)
                prompt_ids.append(pid)
                submitted += 1
                logger.info(
                    "[WorkflowComposer]   Step %d/%d (%s) queued: %s  [%d/%d]",
                    si, len(steps), step["workflow"], pid, submitted, total,
                )

        logger.info(
            "[WorkflowComposer] Submitted %d workflow(s) for %d dataset(s)",
            len(prompt_ids), len(dataset_list),
        )
        return {"ui": {"prompt_ids": prompt_ids}}
