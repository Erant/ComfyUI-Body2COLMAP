import { app } from "../../scripts/app.js";

// Extension for Body2COLMAP_LoadDataset to handle index increment/decrement
app.registerExtension({
    name: "Body2COLMAP.LoadDataset",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass === "Body2COLMAP_LoadDataset") {
            const onExecuted = nodeType.prototype.onExecuted;

            nodeType.prototype.onExecuted = function(message) {
                onExecuted?.apply(this, arguments);

                // Python side now handles the increment/decrement logic and returns
                // the next index value in message.index[0]. We just update the widget.
                const indexWidget = this.widgets?.find(w => w.name === "index");
                const controlWidget = this.widgets?.find(w => w.name === "index_control");

                if (indexWidget && controlWidget && message.index) {
                    const control = controlWidget.value;
                    const newIndex = message.index[0];
                    console.log("[Body2COLMAP LoadDataset] Received index from Python:", newIndex, "Control:", control);

                    if (control !== "fixed") {
                        const oldValue = indexWidget.value;
                        indexWidget.value = newIndex;
                        console.log("[Body2COLMAP LoadDataset] Updated widget from", oldValue, "to", newIndex);
                        // Trigger callback to notify ComfyUI of the change
                        if (indexWidget.callback) {
                            console.log("[Body2COLMAP LoadDataset] Calling widget callback with:", newIndex);
                            indexWidget.callback(newIndex);
                        } else {
                            console.log("[Body2COLMAP LoadDataset] No callback found on widget!");
                        }
                    }
                }
            };
        }
    }
});
