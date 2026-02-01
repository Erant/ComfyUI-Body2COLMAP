import { app } from "../../scripts/app.js";

// Extension for Body2COLMAP_LoadDataset to handle index increment/decrement
app.registerExtension({
    name: "Body2COLMAP.LoadDataset",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass === "Body2COLMAP_LoadDataset") {
            const onExecuted = nodeType.prototype.onExecuted;

            nodeType.prototype.onExecuted = function(message) {
                onExecuted?.apply(this, arguments);

                // Find the index and index_control widgets
                const indexWidget = this.widgets?.find(w => w.name === "index");
                const controlWidget = this.widgets?.find(w => w.name === "index_control");

                if (indexWidget && controlWidget) {
                    const control = controlWidget.value;
                    console.log("[Body2COLMAP LoadDataset] Current index:", indexWidget.value, "Control:", control);

                    if (control === "increment") {
                        const oldValue = indexWidget.value;
                        indexWidget.value = indexWidget.value + 1;
                        console.log("[Body2COLMAP LoadDataset] Incremented from", oldValue, "to", indexWidget.value);
                        // Trigger callback to notify ComfyUI of the change
                        if (indexWidget.callback) {
                            console.log("[Body2COLMAP LoadDataset] Calling widget callback with:", indexWidget.value);
                            indexWidget.callback(indexWidget.value);
                        } else {
                            console.log("[Body2COLMAP LoadDataset] No callback found on widget!");
                        }
                    } else if (control === "decrement") {
                        const oldValue = indexWidget.value;
                        indexWidget.value = indexWidget.value - 1;
                        console.log("[Body2COLMAP LoadDataset] Decremented from", oldValue, "to", indexWidget.value);
                        // Trigger callback to notify ComfyUI of the change
                        if (indexWidget.callback) {
                            console.log("[Body2COLMAP LoadDataset] Calling widget callback with:", indexWidget.value);
                            indexWidget.callback(indexWidget.value);
                        } else {
                            console.log("[Body2COLMAP LoadDataset] No callback found on widget!");
                        }
                    }
                    // "fixed" does nothing
                }
            };
        }
    }
});
