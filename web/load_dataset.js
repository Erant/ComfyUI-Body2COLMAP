import { app } from "../../scripts/app.js";

// Extension for Body2COLMAP_LoadDataset to handle index increment/decrement
app.registerExtension({
    name: "Body2COLMAP.LoadDataset",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        console.log("[Body2COLMAP LoadDataset] beforeRegisterNodeDef called");
        console.log("[Body2COLMAP LoadDataset] nodeType.comfyClass:", nodeType.comfyClass);
        console.log("[Body2COLMAP LoadDataset] nodeData.name:", nodeData.name);

        if (nodeType.comfyClass === "Body2COLMAP_LoadDataset") {
            console.log("[Body2COLMAP LoadDataset] Matched! Setting up onExecuted hook");
            const onExecuted = nodeType.prototype.onExecuted;

            nodeType.prototype.onExecuted = function(message) {
                console.log("[Body2COLMAP LoadDataset] onExecuted called!");
                console.log("[Body2COLMAP LoadDataset] message:", message);
                console.log("[Body2COLMAP LoadDataset] this.widgets:", this.widgets);

                onExecuted?.apply(this, arguments);

                // Find the index and index_control widgets
                const indexWidget = this.widgets?.find(w => w.name === "index");
                const controlWidget = this.widgets?.find(w => w.name === "index_control");

                console.log("[Body2COLMAP LoadDataset] indexWidget:", indexWidget);
                console.log("[Body2COLMAP LoadDataset] controlWidget:", controlWidget);

                if (indexWidget && controlWidget) {
                    const control = controlWidget.value;
                    console.log("[Body2COLMAP LoadDataset] control value:", control);
                    console.log("[Body2COLMAP LoadDataset] current index:", indexWidget.value);

                    if (control === "increment") {
                        indexWidget.value = indexWidget.value + 1;
                        console.log("[Body2COLMAP LoadDataset] incremented to:", indexWidget.value);
                    } else if (control === "decrement") {
                        indexWidget.value = indexWidget.value - 1;
                        console.log("[Body2COLMAP LoadDataset] decremented to:", indexWidget.value);
                    }
                    // "fixed" does nothing
                } else {
                    console.log("[Body2COLMAP LoadDataset] Widgets not found!");
                }
            };
        }
    }
});
