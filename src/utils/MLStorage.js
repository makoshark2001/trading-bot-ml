const fs = require('fs');
const path = require('path');

class MLStorage {
    constructor(config = {}) {
        this.baseDir = config.baseDir || path.join(process.cwd(), 'data', 'ml');
        this.consolidatedDir = path.join(this.baseDir, 'consolidated');
        
        this.saveInterval = config.saveInterval || 300000;
        this.maxAgeHours = config.maxAgeHours || 168;
        this.enableCache = config.enableCache !== false;
        
        this.assetDataCache = new Map();
        
        // DISABLE all legacy directories - only use consolidated
        this.legacyMode = false;
        
        this.initializeDirectories();
        this.startPeriodicSave();
        
        console.log('🔧 PURE CONSOLIDATED MLStorage initialized', {
            baseDir: this.baseDir,
            consolidatedDir: this.consolidatedDir,
            enableCache: this.enableCache,
            saveInterval: this.saveInterval,
            legacyMode: this.legacyMode
        });
    }
    
    initializeDirectories() {
        // ONLY create consolidated directory - no legacy dirs
        const dirs = [this.baseDir, this.consolidatedDir];
        
        dirs.forEach(dir => {
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
                console.log('Created consolidated storage directory:', dir);
            }
        });
        
        // Prevent accidental legacy directory creation
        this.preventLegacyDirectories();
    }
    
    preventLegacyDirectories() {
        const legacyDirs = ['models', 'weights', 'training', 'predictions', 'features'];
        
        legacyDirs.forEach(dirName => {
            const legacyPath = path.join(this.baseDir, dirName);
            if (fs.existsSync(legacyPath)) {
                console.warn(`⚠️ Legacy directory detected: ${legacyPath} - should be migrated`);
            }
        });
    }
    
    getAssetFilePath(pair) {
        return path.join(this.consolidatedDir, `${pair.toLowerCase()}_complete.json`);
    }
    
    async writeFileAtomic(filePath, data) {
        const tempPath = `${filePath}.tmp`;
        const backupPath = `${filePath}.backup`;
        
        try {
            if (fs.existsSync(filePath)) {
                fs.copyFileSync(filePath, backupPath);
            }
            
            const jsonData = JSON.stringify(data, null, 2);
            fs.writeFileSync(tempPath, jsonData, 'utf8');
            
            const verification = JSON.parse(fs.readFileSync(tempPath, 'utf8'));
            if (!verification || !verification.timestamp) {
                throw new Error('Data verification failed');
            }
            
            fs.renameSync(tempPath, filePath);
            
            if (fs.existsSync(backupPath)) {
                fs.unlinkSync(backupPath);
            }
            
            console.log('✅ Atomic write completed:', filePath);
            return true;
            
        } catch (error) {
            console.error('❌ Atomic write failed:', filePath, error.message);
            
            if (fs.existsSync(tempPath)) {
                fs.unlinkSync(tempPath);
            }
            
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
    
    loadAssetData(pair) {
        const cacheKey = pair.toUpperCase();
        
        if (this.enableCache && this.assetDataCache.has(cacheKey)) {
            const cached = this.assetDataCache.get(cacheKey);
            if (Date.now() - cached.cacheTime < 60000) {
                return cached.data;
            }
        }
        
        const filePath = this.getAssetFilePath(pair);
        
        try {
            if (!fs.existsSync(filePath)) {
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
            
            if (!parsed || !parsed.timestamp || !parsed.pair) {
                console.warn('❌ Invalid file structure:', filePath);
                return this.createEmptyAssetData(pair);
            }
            
            if (this.enableCache) {
                this.assetDataCache.set(cacheKey, {
                    data: parsed,
                    cacheTime: Date.now()
                });
            }
            
            return parsed;
            
        } catch (error) {
            console.error(`❌ Failed to load data for ${pair}:`, error.message);
            
            const backupPath = `${filePath}.backup`;
            if (fs.existsSync(backupPath)) {
                try {
                    const backupData = fs.readFileSync(backupPath, 'utf8');
                    const parsed = JSON.parse(backupData);
                    
                    if (parsed && parsed.timestamp) {
                        console.log(`🔄 Recovered from backup for ${pair}`);
                        return parsed;
                    }
                } catch (backupError) {
                    console.error(`❌ Backup recovery failed for ${pair}:`, backupError.message);
                }
            }
            
            return this.createEmptyAssetData(pair);
        }
    }
    
    createEmptyAssetData(pair) {
        return {
            pair: pair.toUpperCase(),
            version: '2.0.0',
            storageFormat: 'PURE_CONSOLIDATED_SINGLE_FILE',
            timestamp: Date.now(),
            lastUpdated: Date.now(),
            models: {},
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
                totalTrainingHours: 0,
                consolidatedStorage: true,
                legacyMode: false
            }
        };
    }
    
    async saveAssetData(pair, assetData) {
        const filePath = this.getAssetFilePath(pair);
        
        assetData.timestamp = Date.now();
        assetData.lastUpdated = Date.now();
        assetData.storageFormat = 'PURE_CONSOLIDATED_SINGLE_FILE';
        assetData.metadata.consolidatedStorage = true;
        assetData.metadata.legacyMode = false;
        
        await this.writeFileAtomic(filePath, assetData);
        
        if (this.enableCache) {
            this.assetDataCache.set(pair.toUpperCase(), {
                data: assetData,
                cacheTime: Date.now()
            });
        }
        
        console.log(`💾 Saved PURE consolidated data for ${pair}`);
    }
    
    // CONSOLIDATED weight management - NO separate files
    async saveModelWeights(pair, modelType, modelWrapper) {
        try {
            console.log(`🔧 STARTING consolidated weight save for ${pair}:${modelType}`);
            
            const assetData = this.loadAssetData(pair);
            
            const model = modelWrapper.model;
            if (!model || !model.getWeights) {
                throw new Error(`Invalid model object for ${pair}:${modelType}`);
            }
            
            console.log(`🔧 Extracting weights from ${pair}:${modelType}...`);
            const weights = model.getWeights();
            console.log(`📊 Found ${weights.length} weight tensors`);
            
            const weightsData = [];
            
            for (let i = 0; i < weights.length; i++) {
                const weightTensor = weights[i];
                console.log(`🔧 Processing tensor ${i + 1}/${weights.length}`);
                
                try {
                    const weightArray = await weightTensor.data();
                    const shape = weightTensor.shape;
                    
                    weightsData.push({
                        data: Array.from(weightArray),
                        shape: shape,
                        dtype: weightTensor.dtype,
                        index: i
                    });
                    
                    console.log(`✅ Tensor ${i + 1} extracted: ${weightArray.length} values`);
                } catch (tensorError) {
                    console.error(`❌ Failed to extract tensor ${i + 1}:`, tensorError.message);
                    throw new Error(`Failed to extract tensor ${i}: ${tensorError.message}`);
                }
            }
            
            if (!assetData.models) {
                assetData.models = {};
            }
            
            assetData.models[modelType] = {
                weights: {
                    data: weightsData,
                    count: weightsData.length,
                    savedAt: Date.now(),
                    version: '2.0.0',
                    storageType: 'PURE_CONSOLIDATED'
                },
                config: modelWrapper.getModelSummary ? modelWrapper.getModelSummary().config : {},
                architecture: modelWrapper.getModelSummary ? modelWrapper.getModelSummary().architecture : {},
                metadata: {
                    modelType: modelType,
                    pair: pair.toUpperCase(),
                    totalParams: modelWrapper.model?.countParams?.() || 0,
                    isCompiled: modelWrapper.isCompiled || false,
                    savedAt: Date.now(),
                    storageFormat: 'PURE_CONSOLIDATED_SINGLE_FILE',
                    noSeparateFiles: true
                }
            };
            
            assetData.metadata.totalModelsSaved++;
            assetData.metadata.lastModelSaved = {
                modelType: modelType,
                savedAt: Date.now(),
                weightsCount: weightsData.length
            };
            
            // SAVE ONLY TO CONSOLIDATED FILE - NO SEPARATE FILES
            await this.saveAssetData(pair, assetData);
            
            console.log(`✅ PURE consolidated weight save completed for ${pair}:${modelType}`, {
                weightsCount: weightsData.length,
                storageType: 'PURE_CONSOLIDATED'
            });
            
            return true;
            
        } catch (error) {
            console.error(`❌ FAILED consolidated weight save for ${pair}:${modelType}`, { 
                error: error.message
            });
            throw error;
        }
    }
    
    async loadModelWeights(pair, modelType, ModelClass, config) {
        try {
            console.log(`🔧 STARTING consolidated weight load for ${pair}:${modelType}`);
            
            const assetData = this.loadAssetData(pair);
            
            if (!assetData.models || !assetData.models[modelType] || !assetData.models[modelType].weights) {
                console.log(`❌ No consolidated weights found for ${pair}:${modelType}`);
                return null;
            }
            
            const modelData = assetData.models[modelType];
            const weightsInfo = modelData.weights;
            
            console.log(`📊 Found consolidated weights for ${pair}:${modelType}:`, {
                weightsCount: weightsInfo.count,
                savedAt: new Date(weightsInfo.savedAt).toLocaleString(),
                storageType: weightsInfo.storageType || 'PURE_CONSOLIDATED'
            });
            
            if (modelData.config && config) {
                const savedFeatureCount = modelData.config.features;
                const currentFeatureCount = config.features;
                
                if (savedFeatureCount !== currentFeatureCount) {
                    console.warn(`❌ Feature count mismatch for ${pair}:${modelType}. Saved: ${savedFeatureCount}, Current: ${currentFeatureCount}`);
                    return null;
                }
            }
            
            console.log(`🔧 Creating new ${modelType} model for ${pair}...`);
            const modelWrapper = new ModelClass(config);
            modelWrapper.buildModel();
            modelWrapper.compileModel();
            
            console.log(`✅ New model created and compiled`);
            
            const tf = require('@tensorflow/tfjs');
            const weightTensors = [];
            
            console.log(`🔧 Reconstructing ${weightsInfo.data.length} tensors...`);
            
            for (let i = 0; i < weightsInfo.data.length; i++) {
                const weightInfo = weightsInfo.data[i];
                console.log(`🔧 Reconstructing tensor ${i + 1}/${weightsInfo.data.length}`);
                
                try {
                    const tensor = tf.tensor(weightInfo.data, weightInfo.shape, weightInfo.dtype);
                    weightTensors.push(tensor);
                    console.log(`✅ Tensor ${i + 1} reconstructed`);
                } catch (tensorError) {
                    console.error(`❌ Failed to reconstruct tensor ${i + 1}:`, tensorError.message);
                    
                    weightTensors.forEach(t => t.dispose());
                    throw new Error(`Failed to reconstruct tensor ${i}: ${tensorError.message}`);
                }
            }
            
            console.log(`🔧 Setting ${weightTensors.length} weights on model...`);
            
            try {
                modelWrapper.model.setWeights(weightTensors);
                console.log(`✅ Weights set successfully`);
            } catch (setWeightsError) {
                console.error(`❌ Failed to set weights:`, setWeightsError.message);
                weightTensors.forEach(tensor => tensor.dispose());
                throw new Error(`Failed to set weights: ${setWeightsError.message}`);
            }
            
            weightTensors.forEach(tensor => tensor.dispose());
            console.log(`🧹 Cleaned up temporary tensors`);
            
            console.log(`✅ PURE consolidated weight load completed for ${pair}:${modelType}`);
            
            return modelWrapper;
            
        } catch (error) {
            console.error(`❌ FAILED consolidated weight load for ${pair}:${modelType}`, { 
                error: error.message
            });
            return null;
        }
    }
    
    hasTrainedWeights(pair, modelType) {
        try {
            const assetData = this.loadAssetData(pair);
            const hasWeights = !!(assetData.models && 
                                 assetData.models[modelType] && 
                                 assetData.models[modelType].weights &&
                                 assetData.models[modelType].weights.data &&
                                 assetData.models[modelType].weights.data.length > 0);
            
            console.log(`🔍 Consolidated weight check for ${pair}:${modelType}: ${hasWeights ? 'EXISTS' : 'NOT FOUND'}`);
            return hasWeights;
        } catch (error) {
            console.error(`❌ Weight check failed for ${pair}:${modelType}:`, error.message);
            return false;
        }
    }
    
    // CONSOLIDATED history management - NO separate files
    async saveTrainingHistory(pair, trainingResults) {
        try {
            console.log(`💾 Saving consolidated training history for ${pair}`);
            
            const assetData = this.loadAssetData(pair);
            
            if (!assetData.training) {
                assetData.training = {
                    history: [],
                    lastTraining: null,
                    totalTrainingSessions: 0
                };
            }
            
            const trainingSession = {
                ...trainingResults,
                timestamp: Date.now(),
                sessionId: `${pair}_${trainingResults.modelType}_${Date.now()}`,
                storageType: 'PURE_CONSOLIDATED'
            };
            
            assetData.training.history.push(trainingSession);
            assetData.training.lastTraining = trainingSession;
            assetData.training.totalTrainingSessions++;
            
            if (assetData.training.history.length > 100) {
                assetData.training.history = assetData.training.history.slice(-100);
            }
            
            assetData.metadata.totalTrainingHours += (trainingSession.duration || 0) / 1000 / 60 / 60;
            
            // SAVE ONLY TO CONSOLIDATED FILE
            await this.saveAssetData(pair, assetData);
            
            console.log(`✅ Consolidated training history saved for ${pair}:${trainingResults.modelType}`);
            
        } catch (error) {
            console.error(`❌ Failed to save consolidated training history for ${pair}:`, error.message);
            throw error;
        }
    }
    
    async savePredictionHistory(pair, predictions) {
        try {
            const assetData = this.loadAssetData(pair);
            
            if (!assetData.predictions) {
                assetData.predictions = {
                    history: [],
                    lastPrediction: null,
                    totalPredictions: 0
                };
            }
            
            const predictionData = Array.isArray(predictions) ? predictions : [predictions];
            
            predictionData.forEach(pred => {
                const predictionEntry = {
                    ...pred,
                    timestamp: pred.timestamp || Date.now(),
                    predictionId: `${pair}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                    storageType: 'PURE_CONSOLIDATED'
                };
                
                assetData.predictions.history.push(predictionEntry);
                assetData.predictions.lastPrediction = predictionEntry;
                assetData.predictions.totalPredictions++;
            });
            
            if (assetData.predictions.history.length > 1000) {
                assetData.predictions.history = assetData.predictions.history.slice(-1000);
            }
            
            assetData.metadata.totalPredictionsMade = assetData.predictions.totalPredictions;
            
            if (assetData.predictions.totalPredictions % 10 === 0) {
                await this.saveAssetData(pair, assetData);
                console.log(`💾 Consolidated prediction history saved for ${pair} (${assetData.predictions.totalPredictions} total)`);
            }
            
        } catch (error) {
            console.error(`❌ Failed to save consolidated prediction history for ${pair}:`, error.message);
        }
    }
    
    async saveFeatureCache(pair, features) {
        try {
            const assetData = this.loadAssetData(pair);
            
            assetData.features = {
                cache: features,
                lastExtraction: Date.now(),
                featureCount: features.count || 0,
                extractedAt: Date.now(),
                storageType: 'PURE_CONSOLIDATED'
            };
            
            await this.saveAssetData(pair, assetData);
            
            console.log(`💾 Consolidated feature cache saved for ${pair}`);
            
        } catch (error) {
            console.error(`❌ Failed to save consolidated feature cache for ${pair}:`, error.message);
        }
    }
    
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
                                        storageFormat: 'PURE_CONSOLIDATED_SINGLE_FILE'
                                    });
                                }
                            });
                        }
                    } catch (error) {
                        console.warn(`Failed to read file ${file}:`, error.message);
                    }
                }
            }
            
            trainedModels.sort((a, b) => b.savedAt - a.savedAt);
            
            return trainedModels;
            
        } catch (error) {
            console.error('Failed to get trained models list:', error.message);
            return [];
        }
    }
    
    getStorageStats() {
        const stats = {
            storageFormat: 'PURE_CONSOLIDATED_SINGLE_FILE',
            legacyMode: false,
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
            console.error('Failed to get storage stats:', error.message);
        }
        
        return stats;
    }
    
    // Rest of the methods remain the same but ensure no legacy file creation
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
                        
                        if (assetData.training && assetData.training.history) {
                            const originalCount = assetData.training.history.length;
                            const recentTraining = assetData.training.history
                                .sort((a, b) => b.timestamp - a.timestamp)
                                .slice(0, 10);
                            const oldTraining = assetData.training.history
                                .filter(session => session.timestamp > cutoffTime);
                            
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
                        
                        if (dataChanged) {
                            await this.saveAssetData(pair, assetData);
                        }
                        
                    } catch (error) {
                        console.error(`Failed to cleanup ${file}:`, error.message);
                    }
                }
            }
            
            this.clearOldCacheEntries(cutoffTime);
            
            console.log(`🧹 Consolidated cleanup completed. Cleaned items: ${cleanedItems}`);
            
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
    
    async forceSave() {
        let savedCount = 0;
        
        try {
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
    
    stopPeriodicSave() {
        if (this.saveIntervalId) {
            clearInterval(this.saveIntervalId);
            this.saveIntervalId = null;
            console.log('⏹️ Periodic consolidated save stopped');
        }
    }
    
    async shutdown() {
        console.log('🛑 Pure Consolidated ML Storage shutting down...');
        
        this.stopPeriodicSave();
        
        try {
            await this.forceSave();
            console.log('✅ Final consolidated save completed during shutdown');
        } catch (error) {
            console.error('❌ Final consolidated save failed during shutdown:', error.message);
        }
        
        this.assetDataCache.clear();
        
        console.log('✅ Pure Consolidated ML Storage shutdown completed');
    }
    
    // Legacy compatibility methods - all redirect to consolidated storage
    async saveModelMetadata(pair, modelInfo) {
        console.log(`🔄 Legacy saveModelMetadata called for ${pair} - using PURE consolidated storage`);
        
        try {
            const assetData = this.loadAssetData(pair);
            
            if (!assetData.metadata) {
                assetData.metadata = {};
            }
            
            assetData.metadata.legacyModelInfo = {
                ...modelInfo,
                savedAt: Date.now(),
                type: 'legacy_model_metadata',
                storageType: 'PURE_CONSOLIDATED'
            };
            
            await this.saveAssetData(pair, assetData);
            console.log(`✅ Legacy model metadata saved in PURE consolidated format for ${pair}`);
            
        } catch (error) {
            console.error(`❌ Failed to save legacy model metadata for ${pair}:`, error.message);
        }
    }
    
    loadModelMetadata(pair) {
        console.log(`🔄 Legacy loadModelMetadata called for ${pair} - using PURE consolidated storage`);
        
        try {
            const assetData = this.loadAssetData(pair);
            return assetData.metadata?.legacyModelInfo || null;
        } catch (error) {
            console.error(`❌ Failed to load legacy model metadata for ${pair}:`, error.message);
            return null;
        }
    }
    
    loadTrainingHistory(pair) {
        console.log(`🔄 Legacy loadTrainingHistory called for ${pair} - using PURE consolidated storage`);
        
        try {
            const assetData = this.loadAssetData(pair);
            return {
                pair: pair.toUpperCase(),
                trainingResults: assetData.training?.history || [],
                timestamp: assetData.timestamp,
                version: '2.0.0',
                type: 'pure_consolidated_training_history'
            };
        } catch (error) {
            console.error(`❌ Failed to load training history for ${pair}:`, error.message);
            return null;
        }
    }
    
    loadPredictionHistory(pair) {
        console.log(`🔄 Legacy loadPredictionHistory called for ${pair} - using PURE consolidated storage`);
        
        try {
            const assetData = this.loadAssetData(pair);
            return {
                pair: pair.toUpperCase(),
                predictions: assetData.predictions?.history || [],
                count: assetData.predictions?.totalPredictions || 0,
                timestamp: assetData.timestamp,
                version: '2.0.0',
                type: 'pure_consolidated_prediction_history'
            };
        } catch (error) {
            console.error(`❌ Failed to load prediction history for ${pair}:`, error.message);
            return null;
        }
    }
    
    loadFeatureCache(pair) {
        console.log(`🔄 Legacy loadFeatureCache called for ${pair} - using PURE consolidated storage`);
        
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
                type: 'pure_consolidated_feature_cache'
            };
        } catch (error) {
            console.error(`❌ Failed to load feature cache for ${pair}:`, error.message);
            return null;
        }
    }
    
    getAssetInfo(pair) {
        try {
            const assetData = this.loadAssetData(pair);
            
            return {
                pair: pair.toUpperCase(),
                storageFormat: 'PURE_CONSOLIDATED_SINGLE_FILE',
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
                        modelParams: data.metadata?.totalParams || 0,
                        storageType: 'PURE_CONSOLIDATED'
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
                metadata: {
                    ...assetData.metadata || {},
                    consolidatedStorage: true,
                    legacyMode: false
                }
            };
        } catch (error) {
            console.error(`❌ Failed to get asset info for ${pair}:`, error.message);
            return {
                pair: pair.toUpperCase(),
                error: error.message,
                fileExists: false,
                storageFormat: 'PURE_CONSOLIDATED_SINGLE_FILE'
            };
        }
    }
    
    // PREVENT legacy directory creation methods
    async preventLegacyFileSaving() {
        const legacyMethods = [
            'saveModelToFile',
            'saveWeightsToFile', 
            'saveTrainingToFile',
            'savePredictionsToFile',
            'saveFeaturesToFile'
        ];
        
        legacyMethods.forEach(methodName => {
            if (this[methodName]) {
                this[methodName] = () => {
                    console.warn(`❌ Legacy method ${methodName} called - redirecting to consolidated storage`);
                    throw new Error(`Legacy method ${methodName} disabled - use consolidated storage only`);
                };
            }
        });
    }
    
    // Check for and warn about legacy files
    checkForLegacyFiles() {
        const legacyDirs = ['models', 'weights', 'training', 'predictions', 'features'];
        const foundLegacy = [];
        
        legacyDirs.forEach(dirName => {
            const legacyPath = path.join(this.baseDir, dirName);
            if (fs.existsSync(legacyPath)) {
                const files = fs.readdirSync(legacyPath);
                if (files.length > 0) {
                    foundLegacy.push({
                        directory: dirName,
                        path: legacyPath,
                        fileCount: files.length
                    });
                }
            }
        });
        
        if (foundLegacy.length > 0) {
            console.warn('⚠️ Legacy storage files detected:', foundLegacy);
            console.warn('💡 Consider running migration to consolidate these files');
            return foundLegacy;
        }
        
        console.log('✅ No legacy files detected - pure consolidated storage');
        return [];
    }
    
    // Migration method for legacy data
    async migrateLegacyData() {
        console.log('🔄 Starting migration from legacy storage to PURE consolidated...');
        
        const migrationResults = {
            migratedAssets: 0,
            migratedModels: 0,
            migratedWeights: 0,
            migratedTraining: 0,
            migratedPredictions: 0,
            migratedFeatures: 0,
            errors: [],
            details: []
        };
        
        try {
            const oldDirs = {
                models: path.join(this.baseDir, 'models'),
                weights: path.join(this.baseDir, 'weights'), 
                training: path.join(this.baseDir, 'training'),
                predictions: path.join(this.baseDir, 'predictions'),
                features: path.join(this.baseDir, 'features')
            };
            
            const discoveredAssets = new Set();
            
            console.log('🔍 Discovering assets from legacy storage...');
            
            // Discover assets from various legacy directories
            Object.entries(oldDirs).forEach(([type, dirPath]) => {
                if (fs.existsSync(dirPath)) {
                    const files = fs.readdirSync(dirPath);
                    files.forEach(file => {
                        if (type === 'weights') {
                            // For weights directory, look for subdirectories
                            const itemPath = path.join(dirPath, file);
                            if (fs.statSync(itemPath).isDirectory()) {
                                const parts = file.split('_');
                                if (parts.length >= 1) {
                                    discoveredAssets.add(parts[0].toUpperCase());
                                }
                            }
                        } else {
                            // For other directories, extract pair from filename
                            const match = file.match(/^([^_]+)_/);
                            if (match) {
                                discoveredAssets.add(match[1].toUpperCase());
                            }
                        }
                    });
                }
            });
            
            console.log(`🔍 Discovered ${discoveredAssets.size} assets:`, Array.from(discoveredAssets));
            
            if (discoveredAssets.size === 0) {
                console.log('ℹ️ No legacy assets found to migrate');
                return migrationResults;
            }
            
            // Migrate each discovered asset
            for (const pair of discoveredAssets) {
                try {
                    console.log(`🔄 Migrating ${pair} to pure consolidated storage...`);
                    
                    const assetDetail = {
                        pair: pair,
                        migratedComponents: [],
                        errors: []
                    };
                    
                    let assetData;
                    if (fs.existsSync(this.getAssetFilePath(pair))) {
                        assetData = this.loadAssetData(pair);
                        console.log(`📂 Found existing consolidated file for ${pair}, merging...`);
                    } else {
                        assetData = this.createEmptyAssetData(pair);
                        console.log(`📝 Creating new consolidated file for ${pair}...`);
                    }
                    
                    // Migrate training history
                    await this.migratePairTrainingHistory(pair, oldDirs.training, assetData, assetDetail, migrationResults);
                    
                    // Migrate prediction history
                    await this.migratePairPredictionHistory(pair, oldDirs.predictions, assetData, assetDetail, migrationResults);
                    
                    // Migrate feature cache
                    await this.migratePairFeatureCache(pair, oldDirs.features, assetData, assetDetail, migrationResults);
                    
                    // Migrate model metadata
                    await this.migratePairModelMetadata(pair, oldDirs.models, assetData, assetDetail, migrationResults);
                    
                    // Check for weight files (create placeholders)
                    await this.migratePairWeights(pair, oldDirs.weights, assetData, assetDetail, migrationResults);
                    
                    // Update asset metadata
                    assetData.metadata.migrated = true;
                    assetData.metadata.migratedAt = Date.now();
                    assetData.metadata.migratedComponents = assetDetail.migratedComponents;
                    assetData.metadata.migrationErrors = assetDetail.errors;
                    assetData.metadata.consolidatedStorage = true;
                    assetData.metadata.legacyMode = false;
                    
                    await this.saveAssetData(pair, assetData);
                    migrationResults.migratedAssets++;
                    
                    assetDetail.status = 'completed';
                    assetDetail.consolidatedFile = this.getAssetFilePath(pair);
                    migrationResults.details.push(assetDetail);
                    
                    console.log(`✅ Migration completed for ${pair}`, {
                        components: assetDetail.migratedComponents,
                        errors: assetDetail.errors.length
                    });
                    
                } catch (error) {
                    const errorMsg = `Asset migration failed for ${pair}: ${error.message}`;
                    console.error(`❌ ${errorMsg}`);
                    migrationResults.errors.push(errorMsg);
                    
                    migrationResults.details.push({
                        pair: pair,
                        status: 'failed',
                        error: error.message,
                        migratedComponents: [],
                        errors: [errorMsg]
                    });
                }
            }
            
            const summary = {
                totalAssetsDiscovered: discoveredAssets.size,
                successful: migrationResults.migratedAssets,
                failed: discoveredAssets.size - migrationResults.migratedAssets,
                componentsTotal: migrationResults.migratedModels + migrationResults.migratedWeights + 
                               migrationResults.migratedTraining + migrationResults.migratedPredictions + 
                               migrationResults.migratedFeatures,
                errorCount: migrationResults.errors.length
            };
            
            console.log('✅ Migration to PURE consolidated storage finished:', summary);
            
            return {
                ...migrationResults,
                summary: summary,
                completed: true,
                completedAt: Date.now()
            };
            
        } catch (error) {
            console.error('❌ Migration process failed:', error.message);
            migrationResults.errors.push(`Migration process failed: ${error.message}`);
            migrationResults.completed = false;
            migrationResults.failedAt = Date.now();
            return migrationResults;
        }
    }
    
    // Helper migration methods
    async migratePairTrainingHistory(pair, trainingDir, assetData, assetDetail, migrationResults) {
        const oldTrainingFile = path.join(trainingDir, `${pair.toLowerCase()}_training.json`);
        if (fs.existsSync(oldTrainingFile)) {
            try {
                console.log(`📚 Migrating training history for ${pair}...`);
                const oldTrainingData = JSON.parse(fs.readFileSync(oldTrainingFile, 'utf8'));
                
                if (oldTrainingData.trainingResults) {
                    const trainingEntry = {
                        ...oldTrainingData.trainingResults,
                        migrated: true,
                        originalTimestamp: oldTrainingData.timestamp,
                        migratedAt: Date.now(),
                        sessionId: `${pair}_migrated_${Date.now()}`,
                        storageType: 'PURE_CONSOLIDATED'
                    };
                    
                    assetData.training.history.push(trainingEntry);
                    assetData.training.totalTrainingSessions++;
                    assetData.training.lastTraining = trainingEntry;
                    
                    migrationResults.migratedTraining++;
                    assetDetail.migratedComponents.push('training');
                    console.log(`✅ Training history migrated for ${pair}`);
                }
            } catch (error) {
                const errorMsg = `Training migration failed for ${pair}: ${error.message}`;
                console.warn(`❌ ${errorMsg}`);
                migrationResults.errors.push(errorMsg);
                assetDetail.errors.push(errorMsg);
            }
        }
    }
    
    async migratePairPredictionHistory(pair, predictionsDir, assetData, assetDetail, migrationResults) {
        const oldPredictionsFile = path.join(predictionsDir, `${pair.toLowerCase()}_predictions.json`);
        if (fs.existsSync(oldPredictionsFile)) {
            try {
                console.log(`🎯 Migrating prediction history for ${pair}...`);
                const oldPredictionsData = JSON.parse(fs.readFileSync(oldPredictionsFile, 'utf8'));
                
                if (oldPredictionsData.predictions && Array.isArray(oldPredictionsData.predictions)) {
                    const migratedPredictions = oldPredictionsData.predictions.map(pred => ({
                        ...pred,
                        migrated: true,
                        migratedAt: Date.now(),
                        predictionId: `${pair}_migrated_${pred.timestamp || Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                        storageType: 'PURE_CONSOLIDATED'
                    }));
                    
                    assetData.predictions.history.push(...migratedPredictions);
                    assetData.predictions.totalPredictions += migratedPredictions.length;
                    assetData.predictions.lastPrediction = migratedPredictions[migratedPredictions.length - 1];
                    
                    migrationResults.migratedPredictions++;
                    assetDetail.migratedComponents.push('predictions');
                    console.log(`✅ ${migratedPredictions.length} predictions migrated for ${pair}`);
                }
            } catch (error) {
                const errorMsg = `Predictions migration failed for ${pair}: ${error.message}`;
                console.warn(`❌ ${errorMsg}`);
                migrationResults.errors.push(errorMsg);
                assetDetail.errors.push(errorMsg);
            }
        }
    }
    
    async migratePairFeatureCache(pair, featuresDir, assetData, assetDetail, migrationResults) {
        const oldFeaturesFile = path.join(featuresDir, `${pair.toLowerCase()}_features.json`);
        if (fs.existsSync(oldFeaturesFile)) {
            try {
                console.log(`🔧 Migrating feature cache for ${pair}...`);
                const oldFeaturesData = JSON.parse(fs.readFileSync(oldFeaturesFile, 'utf8'));
                
                if (oldFeaturesData.features) {
                    assetData.features = {
                        cache: {
                            ...oldFeaturesData.features,
                            migrated: true,
                            migratedAt: Date.now(),
                            storageType: 'PURE_CONSOLIDATED'
                        },
                        lastExtraction: oldFeaturesData.timestamp || Date.now(),
                        featureCount: oldFeaturesData.features.count || 0,
                        extractedAt: oldFeaturesData.timestamp || Date.now()
                    };
                    
                    migrationResults.migratedFeatures++;
                    assetDetail.migratedComponents.push('features');
                    console.log(`✅ Feature cache migrated for ${pair}`);
                }
            } catch (error) {
                const errorMsg = `Features migration failed for ${pair}: ${error.message}`;
                console.warn(`❌ ${errorMsg}`);
                migrationResults.errors.push(errorMsg);
                assetDetail.errors.push(errorMsg);
            }
        }
    }
    
    async migratePairModelMetadata(pair, modelsDir, assetData, assetDetail, migrationResults) {
        const oldModelsFile = path.join(modelsDir, `${pair.toLowerCase()}_models.json`);
        if (fs.existsSync(oldModelsFile)) {
            try {
                console.log(`🤖 Migrating model metadata for ${pair}...`);
                const oldModelsData = JSON.parse(fs.readFileSync(oldModelsFile, 'utf8'));
                
                if (oldModelsData.models) {
                    Object.entries(oldModelsData.models).forEach(([modelType, modelInfo]) => {
                        if (!assetData.models[modelType]) {
                            assetData.models[modelType] = {};
                        }
                        
                        assetData.models[modelType].metadata = {
                            ...modelInfo,
                            migrated: true,
                            migratedAt: Date.now(),
                            originalData: modelInfo,
                            storageType: 'PURE_CONSOLIDATED'
                        };
                    });
                    
                    migrationResults.migratedModels++;
                    assetDetail.migratedComponents.push('models');
                    console.log(`✅ Model metadata migrated for ${pair}`);
                }
            } catch (error) {
                const errorMsg = `Models migration failed for ${pair}: ${error.message}`;
                console.warn(`❌ ${errorMsg}`);
                migrationResults.errors.push(errorMsg);
                assetDetail.errors.push(errorMsg);
            }
        }
    }
    
    async migratePairWeights(pair, weightsDir, assetData, assetDetail, migrationResults) {
        if (fs.existsSync(weightsDir)) {
            const weightDirs = fs.readdirSync(weightsDir);
            const pairWeightDirs = weightDirs.filter(dir => dir.toLowerCase().startsWith(pair.toLowerCase()));
            
            for (const weightDir of pairWeightDirs) {
                try {
                    console.log(`⚖️ Checking weights in ${weightDir} for ${pair}...`);
                    const weightDirPath = path.join(weightsDir, weightDir);
                    
                    if (fs.statSync(weightDirPath).isDirectory()) {
                        const weightFiles = fs.readdirSync(weightDirPath);
                        const modelJsonFiles = weightFiles.filter(f => f === 'model.json');
                        const binFiles = weightFiles.filter(f => f.endsWith('.bin'));
                        
                        if (modelJsonFiles.length > 0 && binFiles.length > 0) {
                            const parts = weightDir.split('_');
                            const modelType = parts.length > 1 ? parts[1].toLowerCase() : 'lstm';
                            
                            if (!assetData.models[modelType]) {
                                assetData.models[modelType] = {};
                            }
                            
                            assetData.models[modelType].weights = {
                                migrated: true,
                                migratedAt: Date.now(),
                                originalPath: weightDirPath,
                                modelJsonFile: modelJsonFiles[0],
                                binFiles: binFiles,
                                note: 'Weight files detected but not migrated (requires model reconstruction)',
                                status: 'placeholder',
                                storageType: 'PURE_CONSOLIDATED'
                            };
                            
                            migrationResults.migratedWeights++;
                            assetDetail.migratedComponents.push(`weights_${modelType}`);
                            console.log(`⚖️ Weight placeholder created for ${pair}:${modelType}`);
                        }
                    }
                } catch (error) {
                    const errorMsg = `Weight migration failed for ${pair} in ${weightDir}: ${error.message}`;
                    console.warn(`❌ ${errorMsg}`);
                    migrationResults.errors.push(errorMsg);
                    assetDetail.errors.push(errorMsg);
                }
            }
        }
    }
}

module.exports = MLStorage;