const fs = require('fs');
const path = require('path');

class MLStorage {
    constructor(config = {}) {
        this.baseDir = config.baseDir || path.join(process.cwd(), 'data', 'ml');
        this.consolidatedDir = path.join(this.baseDir, 'consolidated'); // NEW: Single directory
        
        this.saveInterval = config.saveInterval || 300000; // 5 minutes
        this.maxAgeHours = config.maxAgeHours || 168; // 7 days
        this.enableCache = config.enableCache !== false;
        
        // In-memory cache for consolidated data
        this.assetDataCache = new Map(); // Cache complete asset data
        
        this.initializeDirectories();
        this.startPeriodicSave();
        
        console.log('🔧 CONSOLIDATED MLStorage initialized - SINGLE FILE PER ASSET', {
            baseDir: this.baseDir,
            consolidatedDir: this.consolidatedDir,
            enableCache: this.enableCache,
            saveInterval: this.saveInterval,
            storageFormat: 'SINGLE_FILE_PER_ASSET'
        });
    }
    
    initializeDirectories() {
        const dirs = [this.baseDir, this.consolidatedDir];
        
        dirs.forEach(dir => {
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
                console.log('Created consolidated storage directory:', dir);
            }
        });
    }
    
    // Get the file path for an asset's consolidated data
    getAssetFilePath(pair) {
        return path.join(this.consolidatedDir, `${pair.toLowerCase()}_complete.json`);
    }
    
    // Atomic file writing with corruption prevention
    async writeFileAtomic(filePath, data) {
        const tempPath = `${filePath}.tmp`;
        const backupPath = `${filePath}.backup`;
        
        try {
            // Create backup if original exists
            if (fs.existsSync(filePath)) {
                fs.copyFileSync(filePath, backupPath);
            }
            
            // Write to temporary file
            const jsonData = JSON.stringify(data, null, 2);
            fs.writeFileSync(tempPath, jsonData, 'utf8');
            
            // Verify written data
            const verification = JSON.parse(fs.readFileSync(tempPath, 'utf8'));
            if (!verification || !verification.timestamp) {
                throw new Error('Data verification failed');
            }
            
            // Atomic rename
            fs.renameSync(tempPath, filePath);
            
            // Clean up backup after successful write
            if (fs.existsSync(backupPath)) {
                fs.unlinkSync(backupPath);
            }
            
            console.log('✅ Atomic consolidated write completed:', filePath);
            return true;
            
        } catch (error) {
            console.error('❌ Atomic consolidated write failed:', filePath, error.message);
            
            // Clean up temp file
            if (fs.existsSync(tempPath)) {
                fs.unlinkSync(tempPath);
            }
            
            // Restore from backup if needed
            if (fs.existsSync(backupPath)) {
                if (fs.existsSync(filePath)) {
                    fs.unlinkSync(filePath);
                }
                fs.renameSync(backupPath, filePath);
                console.log('🔄 Restored from backup:', filePath);
            }
            
            throw error;
        }
    }
    
    // Load complete asset data from consolidated file
    loadAssetData(pair) {
        const cacheKey = pair.toUpperCase();
        
        // Check cache first
        if (this.enableCache && this.assetDataCache.has(cacheKey)) {
            const cached = this.assetDataCache.get(cacheKey);
            // Return cached data if less than 1 minute old
            if (Date.now() - cached.cacheTime < 60000) {
                return cached.data;
            }
        }
        
        const filePath = this.getAssetFilePath(pair);
        
        try {
            if (!fs.existsSync(filePath)) {
                // Return empty structure if file doesn't exist
                const emptyData = this.createEmptyAssetData(pair);
                if (this.enableCache) {
                    this.assetDataCache.set(cacheKey, {
                        data: emptyData,
                        cacheTime: Date.now()
                    });
                }
                return emptyData;
            }
            
            const data = fs.readFileSync(filePath, 'utf8');
            const parsed = JSON.parse(data);
            
            // Validate structure
            if (!parsed || !parsed.timestamp || !parsed.pair) {
                console.warn('❌ Invalid consolidated file structure:', filePath);
                return this.createEmptyAssetData(pair);
            }
            
            // Cache the data
            if (this.enableCache) {
                this.assetDataCache.set(cacheKey, {
                    data: parsed,
                    cacheTime: Date.now()
                });
            }
            
            console.log(`📂 Loaded consolidated data for ${pair}:`, {
                modelsCount: Object.keys(parsed.models || {}).length,
                trainingHistoryCount: (parsed.training?.history || []).length,
                predictionsCount: (parsed.predictions?.history || []).length,
                lastUpdated: new Date(parsed.timestamp).toLocaleString()
            });
            
            return parsed;
            
        } catch (error) {
            console.error(`❌ Failed to load consolidated data for ${pair}:`, error.message);
            
            // Try to recover from backup
            const backupPath = `${filePath}.backup`;
            if (fs.existsSync(backupPath)) {
                try {
                    const backupData = fs.readFileSync(backupPath, 'utf8');
                    const parsed = JSON.parse(backupData);
                    
                    if (parsed && parsed.timestamp) {
                        console.log(`🔄 Recovered consolidated data from backup for ${pair}`);
                        return parsed;
                    }
                } catch (backupError) {
                    console.error(`❌ Backup recovery failed for ${pair}:`, backupError.message);
                }
            }
            
            return this.createEmptyAssetData(pair);
        }
    }
    
    // Create empty asset data structure
    createEmptyAssetData(pair) {
        return {
            pair: pair.toUpperCase(),
            version: '2.0.0',
            storageFormat: 'SINGLE_FILE_PER_ASSET',
            timestamp: Date.now(),
            lastUpdated: Date.now(),
            models: {}, // Will contain lstm, gru, cnn, transformer data
            training: {
                history: [],
                lastTraining: null,
                totalTrainingSessions: 0
            },
            predictions: {
                history: [],
                lastPrediction: null,
                totalPredictions: 0
            },
            features: {
                cache: null,
                lastExtraction: null,
                featureCount: 0
            },
            metadata: {
                createdAt: Date.now(),
                totalModelsSaved: 0,
                totalPredictionsMade: 0,
                totalTrainingHours: 0
            }
        };
    }
    
    // Save complete asset data to consolidated file
    async saveAssetData(pair, assetData) {
        const filePath = this.getAssetFilePath(pair);
        
        // Update timestamps
        assetData.timestamp = Date.now();
        assetData.lastUpdated = Date.now();
        
        await this.writeFileAtomic(filePath, assetData);
        
        // Update cache
        if (this.enableCache) {
            this.assetDataCache.set(pair.toUpperCase(), {
                data: assetData,
                cacheTime: Date.now()
            });
        }
        
        console.log(`💾 Saved consolidated data for ${pair}:`, {
            modelsCount: Object.keys(assetData.models || {}).length,
            trainingHistoryCount: (assetData.training?.history || []).length,
            predictionsCount: (assetData.predictions?.history || []).length,
            fileSize: JSON.stringify(assetData).length + ' bytes'
        });
    }
    
    // 🔧 CONSOLIDATED: Save model weights into single asset file
    async saveModelWeights(pair, modelType, modelWrapper) {
        try {
            console.log(`🔧 STARTING consolidated weight save for ${pair}:${modelType}`);
            
            // Load existing asset data
            const assetData = this.loadAssetData(pair);
            
            // Extract weights using tensor extraction
            const model = modelWrapper.model;
            if (!model || !model.getWeights) {
                throw new Error(`Invalid model object for ${pair}:${modelType} - no getWeights method`);
            }
            
            console.log(`🔧 Extracting weights from ${pair}:${modelType} model...`);
            const weights = model.getWeights();
            console.log(`📊 Found ${weights.length} weight tensors for ${pair}:${modelType}`);
            
            const weightsData = [];
            
            for (let i = 0; i < weights.length; i++) {
                const weightTensor = weights[i];
                console.log(`🔧 Processing weight tensor ${i + 1}/${weights.length} - shape: [${weightTensor.shape.join(', ')}]`);
                
                try {
                    const weightArray = await weightTensor.data();
                    const shape = weightTensor.shape;
                    
                    weightsData.push({
                        data: Array.from(weightArray),
                        shape: shape,
                        dtype: weightTensor.dtype,
                        index: i
                    });
                    
                    console.log(`✅ Weight tensor ${i + 1} extracted: ${weightArray.length} values`);
                } catch (tensorError) {
                    console.error(`❌ Failed to extract tensor ${i + 1}:`, tensorError.message);
                    throw new Error(`Failed to extract weight tensor ${i}: ${tensorError.message}`);
                }
            }
            
            // Initialize models section if it doesn't exist
            if (!assetData.models) {
                assetData.models = {};
            }
            
            // Save model data in consolidated structure
            assetData.models[modelType] = {
                weights: {
                    data: weightsData,
                    count: weightsData.length,
                    savedAt: Date.now(),
                    version: '2.0.0'
                },
                config: modelWrapper.getModelSummary ? modelWrapper.getModelSummary().config : {},
                architecture: modelWrapper.getModelSummary ? modelWrapper.getModelSummary().architecture : {},
                metadata: {
                    modelType: modelType,
                    pair: pair.toUpperCase(),
                    totalParams: modelWrapper.model?.countParams?.() || 0,
                    isCompiled: modelWrapper.isCompiled || false,
                    savedAt: Date.now(),
                    storageFormat: 'CONSOLIDATED_SINGLE_FILE'
                }
            };
            
            // Update asset metadata
            assetData.metadata.totalModelsSaved++;
            assetData.metadata.lastModelSaved = {
                modelType: modelType,
                savedAt: Date.now(),
                weightsCount: weightsData.length
            };
            
            // Save the complete asset data
            await this.saveAssetData(pair, assetData);
            
            console.log(`✅ COMPLETED consolidated weight save for ${pair}:${modelType}`, {
                weightsCount: weightsData.length,
                totalModelsInFile: Object.keys(assetData.models).length,
                storageFormat: 'CONSOLIDATED_SINGLE_FILE'
            });
            
            return true;
            
        } catch (error) {
            console.error(`❌ FAILED consolidated weight save for ${pair}:${modelType}`, { 
                error: error.message,
                stack: error.stack
            });
            throw error;
        }
    }
    
    // 🔧 CONSOLIDATED: Load model weights from single asset file
    async loadModelWeights(pair, modelType, ModelClass, config) {
        try {
            console.log(`🔧 STARTING consolidated weight load for ${pair}:${modelType}`);
            
            // Load asset data
            const assetData = this.loadAssetData(pair);
            
            // Check if model data exists
            if (!assetData.models || !assetData.models[modelType] || !assetData.models[modelType].weights) {
                console.log(`❌ No weights found for ${pair}:${modelType} in consolidated file`);
                return null;
            }
            
            const modelData = assetData.models[modelType];
            const weightsInfo = modelData.weights;
            
            console.log(`📊 Found weights for ${pair}:${modelType}:`, {
                weightsCount: weightsInfo.count,
                savedAt: new Date(weightsInfo.savedAt).toLocaleString(),
                storageFormat: modelData.metadata?.storageFormat
            });
            
            // Check feature count compatibility
            if (modelData.config && config) {
                const savedFeatureCount = modelData.config.features;
                const currentFeatureCount = config.features;
                
                if (savedFeatureCount !== currentFeatureCount) {
                    console.warn(`❌ Feature count mismatch for ${pair}:${modelType}. Saved: ${savedFeatureCount}, Current: ${currentFeatureCount}`);
                    return null;
                }
            }
            
            // Create new model
            console.log(`🔧 Creating new ${modelType} model for ${pair}...`);
            const modelWrapper = new ModelClass(config);
            modelWrapper.buildModel();
            modelWrapper.compileModel();
            
            console.log(`✅ New model created and compiled for ${pair}:${modelType}`);
            
            // Reconstruct tensors from consolidated data
            const tf = require('@tensorflow/tfjs');
            const weightTensors = [];
            
            console.log(`🔧 Reconstructing ${weightsInfo.data.length} tensors from consolidated data...`);
            
            for (let i = 0; i < weightsInfo.data.length; i++) {
                const weightInfo = weightsInfo.data[i];
                console.log(`🔧 Reconstructing tensor ${i + 1}/${weightsInfo.data.length}: shape [${weightInfo.shape.join(', ')}]`);
                
                try {
                    const tensor = tf.tensor(weightInfo.data, weightInfo.shape, weightInfo.dtype);
                    weightTensors.push(tensor);
                    console.log(`✅ Tensor ${i + 1} reconstructed successfully`);
                } catch (tensorError) {
                    console.error(`❌ Failed to reconstruct tensor ${i + 1}:`, tensorError.message);
                    
                    // Clean up any tensors we've created so far
                    weightTensors.forEach(t => t.dispose());
                    throw new Error(`Failed to reconstruct tensor ${i}: ${tensorError.message}`);
                }
            }
            
            console.log(`🔧 Setting ${weightTensors.length} weights on model...`);
            
            // Set the weights on the model
            try {
                modelWrapper.model.setWeights(weightTensors);
                console.log(`✅ Weights set successfully on ${pair}:${modelType} model`);
            } catch (setWeightsError) {
                console.error(`❌ Failed to set weights on model:`, setWeightsError.message);
                // Clean up weight tensors
                weightTensors.forEach(tensor => tensor.dispose());
                throw new Error(`Failed to set weights: ${setWeightsError.message}`);
            }
            
            // Clean up weight tensors (model has its own copies now)
            weightTensors.forEach(tensor => tensor.dispose());
            console.log(`🧹 Cleaned up temporary tensors`);
            
            console.log(`✅ COMPLETED consolidated weight load for ${pair}:${modelType}`, {
                weightsCount: weightsInfo.data.length,
                savedAt: new Date(weightsInfo.savedAt).toLocaleString(),
                storageFormat: 'CONSOLIDATED_SINGLE_FILE'
            });
            
            return modelWrapper;
            
        } catch (error) {
            console.error(`❌ FAILED consolidated weight load for ${pair}:${modelType}`, { 
                error: error.message,
                stack: error.stack
            });
            return null;
        }
    }
    
    // Check if trained weights exist in consolidated file
    hasTrainedWeights(pair, modelType) {
        try {
            const assetData = this.loadAssetData(pair);
            const hasWeights = !!(assetData.models && 
                                 assetData.models[modelType] && 
                                 assetData.models[modelType].weights &&
                                 assetData.models[modelType].weights.data &&
                                 assetData.models[modelType].weights.data.length > 0);
            
            console.log(`🔍 Weight check for ${pair}:${modelType}: ${hasWeights ? 'EXISTS' : 'NOT FOUND'}`);
            return hasWeights;
        } catch (error) {
            console.error(`❌ Weight check failed for ${pair}:${modelType}:`, error.message);
            return false;
        }
    }
    
    // 🔧 CONSOLIDATED: Save training history to single asset file
    async saveTrainingHistory(pair, trainingResults) {
        try {
            console.log(`💾 Saving training history to consolidated file for ${pair}`);
            
            // Load existing asset data
            const assetData = this.loadAssetData(pair);
            
            // Initialize training section if needed
            if (!assetData.training) {
                assetData.training = {
                    history: [],
                    lastTraining: null,
                    totalTrainingSessions: 0
                };
            }
            
            // Add new training session
            const trainingSession = {
                ...trainingResults,
                timestamp: Date.now(),
                sessionId: `${pair}_${trainingResults.modelType}_${Date.now()}`
            };
            
            assetData.training.history.push(trainingSession);
            assetData.training.lastTraining = trainingSession;
            assetData.training.totalTrainingSessions++;
            
            // Keep only last 100 training sessions
            if (assetData.training.history.length > 100) {
                assetData.training.history = assetData.training.history.slice(-100);
            }
            
            // Update metadata
            assetData.metadata.totalTrainingHours += (trainingSession.duration || 0) / 1000 / 60 / 60;
            
            // Save the complete asset data
            await this.saveAssetData(pair, assetData);
            
            console.log(`✅ Training history saved to consolidated file for ${pair}:${trainingResults.modelType}`);
            
        } catch (error) {
            console.error(`❌ Failed to save training history for ${pair}:`, error.message);
            throw error;
        }
    }
    
    // 🔧 CONSOLIDATED: Save prediction history to single asset file
    async savePredictionHistory(pair, predictions) {
        try {
            // Load existing asset data
            const assetData = this.loadAssetData(pair);
            
            // Initialize predictions section if needed
            if (!assetData.predictions) {
                assetData.predictions = {
                    history: [],
                    lastPrediction: null,
                    totalPredictions: 0
                };
            }
            
            // Add new predictions
            const predictionData = Array.isArray(predictions) ? predictions : [predictions];
            
            predictionData.forEach(pred => {
                const predictionEntry = {
                    ...pred,
                    timestamp: pred.timestamp || Date.now(),
                    predictionId: `${pair}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
                };
                
                assetData.predictions.history.push(predictionEntry);
                assetData.predictions.lastPrediction = predictionEntry;
                assetData.predictions.totalPredictions++;
            });
            
            // Keep only last 1000 predictions
            if (assetData.predictions.history.length > 1000) {
                assetData.predictions.history = assetData.predictions.history.slice(-1000);
            }
            
            // Update metadata
            assetData.metadata.totalPredictionsMade = assetData.predictions.totalPredictions;
            
            // Save periodically (every 10 predictions) to avoid too frequent writes
            if (assetData.predictions.totalPredictions % 10 === 0) {
                await this.saveAssetData(pair, assetData);
                console.log(`💾 Prediction history saved to consolidated file for ${pair} (${assetData.predictions.totalPredictions} total)`);
            }
            
        } catch (error) {
            console.error(`❌ Failed to save prediction history for ${pair}:`, error.message);
        }
    }
    
    // 🔧 CONSOLIDATED: Save feature cache to single asset file
    async saveFeatureCache(pair, features) {
        try {
            // Load existing asset data
            const assetData = this.loadAssetData(pair);
            
            // Update features section
            assetData.features = {
                cache: features,
                lastExtraction: Date.now(),
                featureCount: features.count || 0,
                extractedAt: Date.now()
            };
            
            // Save the complete asset data
            await this.saveAssetData(pair, assetData);
            
            console.log(`💾 Feature cache saved to consolidated file for ${pair}`);
            
        } catch (error) {
            console.error(`❌ Failed to save feature cache for ${pair}:`, error.message);
        }
    }
    
    // Get list of all trained models from consolidated files
    getTrainedModelsList() {
        try {
            const trainedModels = [];
            
            if (!fs.existsSync(this.consolidatedDir)) {
                return trainedModels;
            }
            
            const consolidatedFiles = fs.readdirSync(this.consolidatedDir);
            
            for (const file of consolidatedFiles) {
                if (file.endsWith('_complete.json')) {
                    try {
                        const pair = file.replace('_complete.json', '').toUpperCase();
                        const assetData = this.loadAssetData(pair);
                        
                        if (assetData.models) {
                            Object.entries(assetData.models).forEach(([modelType, modelData]) => {
                                if (modelData.weights && modelData.weights.data) {
                                    trainedModels.push({
                                        pair: pair,
                                        modelType: modelType,
                                        savedAt: modelData.weights.savedAt,
                                        modelParams: modelData.metadata?.totalParams || 0,
                                        featureCount: modelData.config?.features || 0,
                                        weightsCount: modelData.weights.count || 0,
                                        storageFormat: 'CONSOLIDATED_SINGLE_FILE'
                                    });
                                }
                            });
                        }
                    } catch (error) {
                        console.warn(`Failed to read consolidated file ${file}:`, error.message);
                    }
                }
            }
            
            // Sort by saved date (newest first)
            trainedModels.sort((a, b) => b.savedAt - a.savedAt);
            
            return trainedModels;
            
        } catch (error) {
            console.error('Failed to get trained models list:', error.message);
            return [];
        }
    }
    
    // Get storage statistics for consolidated files
    getStorageStats() {
        const stats = {
            storageFormat: 'CONSOLIDATED_SINGLE_FILE',
            consolidatedFiles: {
                count: 0,
                totalSizeBytes: 0,
                files: []
            },
            cache: {
                assetData: this.assetDataCache.size
            },
            trainedModels: this.getTrainedModelsList(),
            summary: {
                totalAssets: 0,
                totalModels: 0,
                totalTrainingSessions: 0,
                totalPredictions: 0
            },
            timestamp: Date.now()
        };
        
        try {
            if (fs.existsSync(this.consolidatedDir)) {
                const files = fs.readdirSync(this.consolidatedDir);
                
                files.forEach(file => {
                    if (file.endsWith('_complete.json')) {
                        const filePath = path.join(this.consolidatedDir, file);
                        const fileStat = fs.statSync(filePath);
                        
                        stats.consolidatedFiles.count++;
                        stats.consolidatedFiles.totalSizeBytes += fileStat.size;
                        stats.consolidatedFiles.files.push({
                            name: file,
                            sizeBytes: fileStat.size,
                            lastModified: fileStat.mtime.toISOString(),
                            asset: file.replace('_complete.json', '').toUpperCase()
                        });
                        
                        // Try to get detailed stats from each file
                        try {
                            const pair = file.replace('_complete.json', '');
                            const assetData = this.loadAssetData(pair);
                            
                            stats.summary.totalAssets++;
                            stats.summary.totalModels += Object.keys(assetData.models || {}).length;
                            stats.summary.totalTrainingSessions += assetData.training?.totalTrainingSessions || 0;
                            stats.summary.totalPredictions += assetData.predictions?.totalPredictions || 0;
                        } catch (error) {
                            console.warn(`Failed to read stats from ${file}:`, error.message);
                        }
                    }
                });
            }
        } catch (error) {
            console.error('Failed to get consolidated storage stats:', error.message);
        }
        
        return stats;
    }
    
    // Cleanup old prediction/training data within consolidated files
    async cleanup(maxAgeHours = this.maxAgeHours) {
        const cutoffTime = Date.now() - (maxAgeHours * 60 * 60 * 1000);
        let cleanedItems = 0;
        
        try {
            if (!fs.existsSync(this.consolidatedDir)) {
                return cleanedItems;
            }
            
            const files = fs.readdirSync(this.consolidatedDir);
            
            for (const file of files) {
                if (file.endsWith('_complete.json')) {
                    try {
                        const pair = file.replace('_complete.json', '');
                        const assetData = this.loadAssetData(pair);
                        let dataChanged = false;
                        
                        // Clean old predictions
                        if (assetData.predictions && assetData.predictions.history) {
                            const originalCount = assetData.predictions.history.length;
                            assetData.predictions.history = assetData.predictions.history.filter(
                                pred => pred.timestamp > cutoffTime
                            );
                            const cleanedPredictions = originalCount - assetData.predictions.history.length;
                            if (cleanedPredictions > 0) {
                                cleanedItems += cleanedPredictions;
                                dataChanged = true;
                                console.log(`🧹 Cleaned ${cleanedPredictions} old predictions for ${pair}`);
                            }
                        }
                        
                        // Clean old training history (but keep at least the last 10)
                        if (assetData.training && assetData.training.history) {
                            const originalCount = assetData.training.history.length;
                            const recentTraining = assetData.training.history
                                .sort((a, b) => b.timestamp - a.timestamp)
                                .slice(0, 10); // Keep last 10
                            const oldTraining = assetData.training.history
                                .filter(session => session.timestamp > cutoffTime);
                            
                            // Combine recent + non-old training
                            const combinedTraining = [...recentTraining];
                            oldTraining.forEach(session => {
                                if (!combinedTraining.find(s => s.sessionId === session.sessionId)) {
                                    combinedTraining.push(session);
                                }
                            });
                            
                            assetData.training.history = combinedTraining;
                            const cleanedTraining = originalCount - assetData.training.history.length;
                            if (cleanedTraining > 0) {
                                cleanedItems += cleanedTraining;
                                dataChanged = true;
                                console.log(`🧹 Cleaned ${cleanedTraining} old training sessions for ${pair}`);
                            }
                        }
                        
                        // Save if data changed
                        if (dataChanged) {
                            await this.saveAssetData(pair, assetData);
                        }
                        
                    } catch (error) {
                        console.error(`Failed to cleanup ${file}:`, error.message);
                    }
                }
            }
            
            // Clear old cache entries
            this.clearOldCacheEntries(cutoffTime);
            
            console.log(`🧹 Consolidated cleanup completed. Cleaned items: ${cleanedItems}, Max age hours: ${maxAgeHours}`);
            
        } catch (error) {
            console.error('Consolidated cleanup failed:', error.message);
        }
        
        return cleanedItems;
    }
    
    clearOldCacheEntries(cutoffTime) {
        for (const [key, value] of this.assetDataCache.entries()) {
            if (value.cacheTime < cutoffTime) {
                this.assetDataCache.delete(key);
            }
        }
    }
    
    // Force save all cached data
    async forceSave() {
        let savedCount = 0;
        
        try {
            // Save all cached asset data
            for (const [pair, cachedData] of this.assetDataCache.entries()) {
                await this.saveAssetData(pair, cachedData.data);
                savedCount++;
            }
            
            console.log(`💾 Consolidated force save completed. Saved assets: ${savedCount}`);
            return savedCount;
            
        } catch (error) {
            console.error('❌ Consolidated force save failed:', error.message);
            throw error;
        }
    }
    
    // Start periodic saving
    startPeriodicSave() {
        if (this.saveIntervalId) {
            clearInterval(this.saveIntervalId);
        }
        
        this.saveIntervalId = setInterval(async () => {
            try {
                await this.forceSave();
                console.log('📅 Periodic consolidated save completed');
            } catch (error) {
                console.error('❌ Periodic consolidated save failed:', error.message);
            }
        }, this.saveInterval);
        
        console.log('⏰ Periodic consolidated save started. Interval:', this.saveInterval, 'ms');
    }
    
    // Stop periodic saving
    stopPeriodicSave() {
        if (this.saveIntervalId) {
            clearInterval(this.saveIntervalId);
            this.saveIntervalId = null;
            console.log('⏹️ Periodic consolidated save stopped');
        }
    }
    
    // Graceful shutdown
    async shutdown() {
        console.log('🛑 Consolidated ML Storage shutting down...');
        
        this.stopPeriodicSave();
        
        try {
            await this.forceSave();
            console.log('✅ Final consolidated save completed during shutdown');
        } catch (error) {
            console.error('❌ Final consolidated save failed during shutdown:', error.message);
        }
        
        // Clear caches
        this.assetDataCache.clear();
        
        console.log('✅ Consolidated ML Storage shutdown completed');
    }

    // Legacy compatibility methods (for backwards compatibility with existing code)
    
    // Model persistence (legacy compatibility)
    async saveModelMetadata(pair, modelInfo) {
        console.log(`🔄 Legacy saveModelMetadata called for ${pair} - using consolidated storage`);
        
        try {
            const assetData = this.loadAssetData(pair);
            
            if (!assetData.metadata) {
                assetData.metadata = {};
            }
            
            assetData.metadata.legacyModelInfo = {
                ...modelInfo,
                savedAt: Date.now(),
                type: 'legacy_model_metadata'
            };
            
            await this.saveAssetData(pair, assetData);
            console.log(`✅ Legacy model metadata saved in consolidated format for ${pair}`);
            
        } catch (error) {
            console.error(`❌ Failed to save legacy model metadata for ${pair}:`, error.message);
        }
    }
    
    loadModelMetadata(pair) {
        console.log(`🔄 Legacy loadModelMetadata called for ${pair} - using consolidated storage`);
        
        try {
            const assetData = this.loadAssetData(pair);
            return assetData.metadata?.legacyModelInfo || null;
        } catch (error) {
            console.error(`❌ Failed to load legacy model metadata for ${pair}:`, error.message);
            return null;
        }
    }
    
    loadTrainingHistory(pair) {
        console.log(`🔄 Legacy loadTrainingHistory called for ${pair} - using consolidated storage`);
        
        try {
            const assetData = this.loadAssetData(pair);
            return {
                pair: pair.toUpperCase(),
                trainingResults: assetData.training?.history || [],
                timestamp: assetData.timestamp,
                version: '2.0.0',
                type: 'consolidated_training_history'
            };
        } catch (error) {
            console.error(`❌ Failed to load training history for ${pair}:`, error.message);
            return null;
        }
    }
    
    loadPredictionHistory(pair) {
        console.log(`🔄 Legacy loadPredictionHistory called for ${pair} - using consolidated storage`);
        
        try {
            const assetData = this.loadAssetData(pair);
            return {
                pair: pair.toUpperCase(),
                predictions: assetData.predictions?.history || [],
                count: assetData.predictions?.totalPredictions || 0,
                timestamp: assetData.timestamp,
                version: '2.0.0',
                type: 'consolidated_prediction_history'
            };
        } catch (error) {
            console.error(`❌ Failed to load prediction history for ${pair}:`, error.message);
            return null;
        }
    }
    
    loadFeatureCache(pair) {
        console.log(`🔄 Legacy loadFeatureCache called for ${pair} - using consolidated storage`);
        
        try {
            const assetData = this.loadAssetData(pair);
            
            if (!assetData.features?.cache) {
                return null;
            }
            
            // Check if cache is still fresh (5 minutes)
            const cacheAge = Date.now() - (assetData.features.lastExtraction || 0);
            if (cacheAge > 300000) {
                return null; // Cache too old
            }
            
            return {
                pair: pair.toUpperCase(),
                features: assetData.features.cache,
                timestamp: assetData.features.lastExtraction,
                version: '2.0.0',
                type: 'consolidated_feature_cache'
            };
        } catch (error) {
            console.error(`❌ Failed to load feature cache for ${pair}:`, error.message);
            return null;
        }
    }
    
    // Utility method to migrate old storage format to new consolidated format
    async migrateFromOldFormat() {
        console.log('🔄 Starting migration from old storage format to consolidated format...');
        
        const migrationResults = {
            migratedAssets: 0,
            migratedModels: 0,
            migratedTraining: 0,
            migratedPredictions: 0,
            errors: []
        };
        
        try {
            // Look for old format directories
            const oldModelsDir = path.join(this.baseDir, 'models');
            const oldWeightsDir = path.join(this.baseDir, 'weights');
            const oldTrainingDir = path.join(this.baseDir, 'training');
            const oldPredictionsDir = path.join(this.baseDir, 'predictions');
            const oldFeaturesDir = path.join(this.baseDir, 'features');
            
            const discoveredAssets = new Set();
            
            // Discover assets from old weight directories
            if (fs.existsSync(oldWeightsDir)) {
                const weightDirs = fs.readdirSync(oldWeightsDir);
                weightDirs.forEach(dir => {
                    const match = dir.match(/^([^_]+)_([^_]+)$/);
                    if (match) {
                        discoveredAssets.add(match[1].toUpperCase());
                    }
                });
            }
            
            // Discover assets from old training files
            if (fs.existsSync(oldTrainingDir)) {
                const trainingFiles = fs.readdirSync(oldTrainingDir);
                trainingFiles.forEach(file => {
                    const match = file.match(/^([^_]+)_training\.json$/);
                    if (match) {
                        discoveredAssets.add(match[1].toUpperCase());
                    }
                });
            }
            
            console.log(`🔍 Discovered ${discoveredAssets.size} assets to migrate:`, Array.from(discoveredAssets));
            
            // Migrate each asset
            for (const pair of discoveredAssets) {
                try {
                    console.log(`🔄 Migrating ${pair}...`);
                    
                    // Create consolidated asset data
                    const assetData = this.createEmptyAssetData(pair);
                    
                    // Migrate models/weights (this would be complex - for now just note the existence)
                    // Note: This is a placeholder - full migration would require recreating the weight extraction
                    
                    // Migrate training history
                    const oldTrainingFile = path.join(oldTrainingDir, `${pair.toLowerCase()}_training.json`);
                    if (fs.existsSync(oldTrainingFile)) {
                        try {
                            const oldTrainingData = JSON.parse(fs.readFileSync(oldTrainingFile, 'utf8'));
                            if (oldTrainingData.trainingResults) {
                                assetData.training.history.push({
                                    ...oldTrainingData.trainingResults,
                                    migrated: true,
                                    originalTimestamp: oldTrainingData.timestamp
                                });
                                migrationResults.migratedTraining++;
                            }
                        } catch (error) {
                            console.warn(`Failed to migrate training for ${pair}:`, error.message);
                            migrationResults.errors.push(`Training migration failed for ${pair}: ${error.message}`);
                        }
                    }
                    
                    // Migrate predictions
                    const oldPredictionsFile = path.join(oldPredictionsDir, `${pair.toLowerCase()}_predictions.json`);
                    if (fs.existsSync(oldPredictionsFile)) {
                        try {
                            const oldPredictionsData = JSON.parse(fs.readFileSync(oldPredictionsFile, 'utf8'));
                            if (oldPredictionsData.predictions) {
                                assetData.predictions.history = oldPredictionsData.predictions.map(pred => ({
                                    ...pred,
                                    migrated: true
                                }));
                                assetData.predictions.totalPredictions = oldPredictionsData.count || assetData.predictions.history.length;
                                migrationResults.migratedPredictions++;
                            }
                        } catch (error) {
                            console.warn(`Failed to migrate predictions for ${pair}:`, error.message);
                            migrationResults.errors.push(`Predictions migration failed for ${pair}: ${error.message}`);
                        }
                    }
                    
                    // Save migrated data
                    await this.saveAssetData(pair, assetData);
                    migrationResults.migratedAssets++;
                    
                    console.log(`✅ Migration completed for ${pair}`);
                    
                } catch (error) {
                    console.error(`❌ Migration failed for ${pair}:`, error.message);
                    migrationResults.errors.push(`Asset migration failed for ${pair}: ${error.message}`);
                }
            }
            
            console.log('✅ Migration completed:', migrationResults);
            return migrationResults;
            
        } catch (error) {
            console.error('❌ Migration process failed:', error.message);
            migrationResults.errors.push(`Migration process failed: ${error.message}`);
            return migrationResults;
        }
    }
    
    // Get detailed information about a specific asset
    getAssetInfo(pair) {
        try {
            const assetData = this.loadAssetData(pair);
            
            return {
                pair: pair.toUpperCase(),
                storageFormat: 'CONSOLIDATED_SINGLE_FILE',
                fileExists: fs.existsSync(this.getAssetFilePath(pair)),
                filePath: this.getAssetFilePath(pair),
                fileSize: fs.existsSync(this.getAssetFilePath(pair)) ? 
                    fs.statSync(this.getAssetFilePath(pair)).size : 0,
                lastUpdated: assetData.lastUpdated,
                models: {
                    count: Object.keys(assetData.models || {}).length,
                    types: Object.keys(assetData.models || {}),
                    details: Object.entries(assetData.models || {}).map(([type, data]) => ({
                        type,
                        hasWeights: !!(data.weights && data.weights.data),
                        weightsCount: data.weights?.count || 0,
                        savedAt: data.weights?.savedAt,
                        modelParams: data.metadata?.totalParams || 0
                    }))
                },
                training: {
                    totalSessions: assetData.training?.totalTrainingSessions || 0,
                    historyCount: (assetData.training?.history || []).length,
                    lastTraining: assetData.training?.lastTraining?.timestamp,
                    totalHours: assetData.metadata?.totalTrainingHours || 0
                },
                predictions: {
                    total: assetData.predictions?.totalPredictions || 0,
                    historyCount: (assetData.predictions?.history || []).length,
                    lastPrediction: assetData.predictions?.lastPrediction?.timestamp
                },
                features: {
                    hasCachedFeatures: !!(assetData.features?.cache),
                    featureCount: assetData.features?.featureCount || 0,
                    lastExtraction: assetData.features?.lastExtraction
                },
                metadata: assetData.metadata || {}
            };
        } catch (error) {
            console.error(`❌ Failed to get asset info for ${pair}:`, error.message);
            return {
                pair: pair.toUpperCase(),
                error: error.message,
                fileExists: false
            };
        }
    }
}

module.exports = MLStorage;