import { app } from "../../scripts/app.js";

// Extension for Body2COLMAP_MergeDatasets dynamic inputs
app.registerExtension({
    name: "Body2COLMAP.MergeDatasets",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Body2COLMAP_MergeDatasets") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);

                // Function to update inputs based on connections
                const updateInputs = () => {
                    // Find all dataset input triplets
                    const datasetInputs = this.inputs.filter(i =>
                        i.name.match(/^(b2c_data|images|masks)_\d+$/)
                    );

                    // Group by dataset number
                    const datasets = {};
                    for (const input of datasetInputs) {
                        const match = input.name.match(/^(b2c_data|images|masks)_(\d+)$/);
                        if (match) {
                            const datasetNum = parseInt(match[2]);
                            if (!datasets[datasetNum]) {
                                datasets[datasetNum] = { b2c_data: null, images: null, masks: null };
                            }
                            datasets[datasetNum][match[1]] = input;
                        }
                    }

                    // Get highest dataset number
                    const datasetNumbers = Object.keys(datasets).map(n => parseInt(n)).sort((a, b) => a - b);
                    const maxDataset = datasetNumbers.length > 0 ? Math.max(...datasetNumbers) : 1;

                    // Check if last dataset has any connections
                    const lastDataset = datasets[maxDataset];
                    const hasAnyConnection =
                        (lastDataset.b2c_data?.link !== null && lastDataset.b2c_data?.link !== undefined) ||
                        (lastDataset.images?.link !== null && lastDataset.images?.link !== undefined) ||
                        (lastDataset.masks?.link !== null && lastDataset.masks?.link !== undefined);

                    // Add next dataset triplet if last one has connections
                    if (hasAnyConnection) {
                        const nextNum = maxDataset + 1;

                        // Check if next dataset already exists
                        const hasNextDataset = this.inputs.some(i => i.name === `b2c_data_${nextNum}`);

                        if (!hasNextDataset) {
                            this.addInput(`b2c_data_${nextNum}`, "B2C_COLMAP_METADATA");
                            this.addInput(`images_${nextNum}`, "IMAGE");
                            this.addInput(`masks_${nextNum}`, "MASK");
                        }
                    }

                    // Remove empty trailing dataset triplets (keep at least dataset 2 available)
                    if (maxDataset > 2) {
                        const secondLast = datasets[maxDataset - 1];
                        const secondLastHasConnection =
                            (secondLast.b2c_data?.link !== null && secondLast.b2c_data?.link !== undefined) ||
                            (secondLast.images?.link !== null && secondLast.images?.link !== undefined) ||
                            (secondLast.masks?.link !== null && secondLast.masks?.link !== undefined);

                        if (!hasAnyConnection && !secondLastHasConnection) {
                            // Remove the last triplet
                            const toRemove = [
                                this.inputs.findIndex(i => i.name === `b2c_data_${maxDataset}`),
                                this.inputs.findIndex(i => i.name === `images_${maxDataset}`),
                                this.inputs.findIndex(i => i.name === `masks_${maxDataset}`)
                            ].filter(idx => idx >= 0).sort((a, b) => b - a); // Remove in reverse order

                            for (const idx of toRemove) {
                                this.removeInput(idx);
                            }
                        }
                    }
                };

                // Override onConnectionsChange to update inputs
                const originalOnConnectionsChange = this.onConnectionsChange;
                this.onConnectionsChange = function(type, index, connected, link_info) {
                    const result = originalOnConnectionsChange?.apply(this, arguments);
                    updateInputs();
                    return result;
                };

                // Add initial second dataset triplet
                setTimeout(() => {
                    const hasDataset2 = this.inputs.some(i => i.name === "b2c_data_2");
                    if (!hasDataset2) {
                        this.addInput("b2c_data_2", "B2C_COLMAP_METADATA");
                        this.addInput("images_2", "IMAGE");
                        this.addInput("masks_2", "MASK");
                    }
                }, 10);

                return result;
            };
        }
    }
});
