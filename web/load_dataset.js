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

                    if (control === "increment") {
                        indexWidget.value = indexWidget.value + 1;
                    } else if (control === "decrement") {
                        indexWidget.value = indexWidget.value - 1;
                    }
                    // "fixed" does nothing
                }
            };
        }
    }
});
