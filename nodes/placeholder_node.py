"""Placeholder node for workflow composition.

This node acts as a stand-in for model/clip/vae connections in API-format
workflow JSONs.  The WorkflowComposer node replaces these with subgraphs
extracted from the live workflow before submission.

The ``key`` input must match an optional input name on the
WorkflowComposer node (e.g. ``"model_high"``, ``"model_low"``).
"""


class Body2COLMAP_Placeholder:
    """Stand-in for a connection that will be injected at submission time."""

    CATEGORY = "Body2COLMAP"
    FUNCTION = "passthrough"
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    OUTPUT_TOOLTIPS = ("Replaced at submission time by WorkflowComposer",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key": ("STRING", {
                    "default": "model_high",
                    "tooltip": (
                        "Must match an optional input name on the "
                        "WorkflowComposer node (e.g. model_high, model_low)"
                    ),
                }),
            },
        }

    def passthrough(self, key):
        raise RuntimeError(
            f"Body2COLMAP_Placeholder '{key}' was not replaced by "
            "WorkflowComposer before execution. This node is only valid "
            "inside workflow JSON files used with the WorkflowComposer node."
        )
