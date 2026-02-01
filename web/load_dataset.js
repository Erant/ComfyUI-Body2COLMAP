import { app } from "../../scripts/app.js";

// Extension for Body2COLMAP_LoadDataset to customize control_after_generate options
app.registerExtension({
    name: "Body2COLMAP.LoadDataset",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Body2COLMAP_LoadDataset") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);

                // Find the control_after_generate widget
                const controlWidget = this.widgets?.find(w => w.name === "control_after_generate");

                if (controlWidget) {
                    // Customize options to remove "random" - only keep fixed, increment, decrement
                    controlWidget.options.values = ["fixed", "increment", "decrement"];

                    // If current value is "random", reset to "fixed"
                    if (controlWidget.value === "random") {
                        controlWidget.value = "fixed";
                    }
                }

                return result;
            };
        }
    }
});
