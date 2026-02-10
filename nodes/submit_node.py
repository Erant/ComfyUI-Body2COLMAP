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
import time

from ..submit import (
    fixup_placeholders,
    get_server_address,
    load_pipeline,
    patch_prompt,
    queue_prompt,
)

logger = logging.getLogger(__name__)


class Body2COLMAP_WorkflowComposer:
    """Submit a pipeline of workflows to ComfyUI."""

    CATEGORY = "Body2COLMAP"
    FUNCTION = "submit"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("STRING", {
                    "default": "pipeline.yaml",
                    "tooltip": "Path to the pipeline YAML config file",
                }),
                "datasets": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Dataset names, one per line",
                }),
            },
            "optional": {
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
        model_high=None,
        model_low=None,
        prompt=None,
        unique_id=None,
    ):
        server = get_server_address()
        settings, steps = load_pipeline(pipeline)

        dataset_list = [
            d.strip() for d in datasets.strip().splitlines() if d.strip()
        ]
        if not dataset_list:
            raise ValueError("No datasets specified")

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
                    fixup_placeholders(workflow, prompt, unique_id)

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
