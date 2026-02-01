import { app } from "../../scripts/app.js";

/**
 * Extension for Body2COLMAP_LoadDataset node.
 *
 * Synchronizes the index widget display with the Python-side state.
 * Python maintains all state and sends the current index via UI updates.
 * JavaScript simply updates the widget display - no state tracking.
 */
app.registerExtension({
    name: "Body2COLMAP.LoadDataset",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass === "Body2COLMAP_LoadDataset") {
            const onExecuted = nodeType.prototype.onExecuted;

            nodeType.prototype.onExecuted = function(message) {
                onExecuted?.apply(this, arguments);

                // Python sends the next index value in message.index
                // We just update the widget to display it
                if (message?.index) {
                    const indexWidget = this.widgets?.find(w => w.name === "index");
                    if (indexWidget) {
                        indexWidget.value = message.index[0];
                    }
                }
            };
        }
    }
});
