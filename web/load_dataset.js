import { app } from "../../scripts/app.js";

/**
 * Extension for Body2COLMAP_LoadDataset node.
 *
 * Updates the index widget after node execution to reflect the next index value
 * when using increment/decrement mode. This ensures the UI stays in sync with
 * the Python-side state tracking for both single-run and batch queue modes.
 */
app.registerExtension({
    name: "Body2COLMAP.LoadDataset",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass === "Body2COLMAP_LoadDataset") {
            const onExecuted = nodeType.prototype.onExecuted;

            nodeType.prototype.onExecuted = function(message) {
                onExecuted?.apply(this, arguments);

                // Python side handles increment/decrement logic and returns the next index.
                // Update the widget to reflect the new value for single-run mode.
                const indexWidget = this.widgets?.find(w => w.name === "index");
                const controlWidget = this.widgets?.find(w => w.name === "index_control");

                if (indexWidget && controlWidget && message.index) {
                    const control = controlWidget.value;
                    const newIndex = message.index[0];

                    if (control !== "fixed") {
                        indexWidget.value = newIndex;
                        // Trigger callback to notify ComfyUI of the change
                        if (indexWidget.callback) {
                            indexWidget.callback(newIndex);
                        }
                    }
                }
            };
        }
    }
});
