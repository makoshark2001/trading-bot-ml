const fs = require('fs');
const path = require('path');

class MLStorage {
    constructor(config = {}) {
        this.baseDir = config.baseDir || path.join(process.cwd(), 'data', 'ml');
        this.modelsDir = path.join(this.baseDir, 'models');
        this.weightsDir = path.join(this.baseDir, 'weights');
        this.trainingDir = path.join(this.baseDir, 'training');
        this.predictionsDir = path.join(this.baseDir, 'predictions');
        this.featuresDir = path.join(this.baseDir, 'features');
        
        this.saveInterval = config.saveInterval || 300000; // 5 minutes
        this.maxAgeHours = config.maxAgeHours || 168; // 7 days
        this.enableCache = config.enableCache !== false;
        
        // In-memory caches
        this.modelCache = new Map();
        this.predictionCache = new Map();
        this.featureCache = new Map();
        this.trainingHistory = new Map();
        this.weightsCache = new Map();
        
        this.initializeDirectories();
        this.startPeriodicSave();
        
        console.log('Enhanced MLStorage initialized with weight persistence', {
            baseDir: this.baseDir,
            enableCache: this.enableCache,
            saveInterval: this.saveInterval,
            weightPersistence: true
        });
    }
    
    initializeDirectories() {
        const dirs = [
            this.baseDir, 
            this.modelsDir, 
            this.weightsDir,
            this.trainingDir, 
            this.predictionsDir, 
            this.featuresDir
        ];
        
        dirs.forEach(dir => {
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
                console.log('Created storage directory:', dir);
            }
        });
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
            
            console.log('Atomic file write completed:', filePath);
            return true;
            
        } catch (error) {
            console.error('Atomic file write failed:', filePath, error.message);
            
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
                console.log('Restored from backup:', filePath);
            }
            
            throw error;
        }
    }
    
    // Safe file reading with error recovery
    readFileSecure(filePath) {
        try {
            if (!fs.existsSync(filePath)) {
                return null;
            }
            
            const data = fs.readFileSync(filePath, 'utf8');
            const parsed = JSON.parse(data);
            
            // Validate basic structure
            if (!parsed || !parsed.timestamp) {
                console.warn('Invalid file structure detected:', filePath);
                return null;
            }
            
            return parsed;
            
        } catch (error) {
            console.error('File read failed:', filePath, error.message);
            
            // Try to recover from backup
            const backupPath = `${filePath}.backup`;
            if (fs.existsSync(backupPath)) {
                try {
                    const backupData = fs.readFileSync(backupPath, 'utf8');
                    const parsed = JSON.parse(backupData);
                    
                    if (parsed && parsed.timestamp) {
                        console.log('Recovered from backup file:', filePath);
                        return parsed;
                    }
                } catch (backupError) {
                    console.error('Backup recovery failed:', filePath, backupError.message);
                }
            }
            
            return null;
        }
    }
    
    // Model persistence
    async saveModelMetadata(pair, modelInfo) {
        const filePath = path.join(this.modelsDir, `${pair.toLowerCase()}_model.json`);
        
        const data = {
            pair: pair.toUpperCase(),
            modelInfo,
            timestamp: Date.now(),
            version: '1.0.0',
            type: 'model_metadata'
        };
        
        await this.writeFileAtomic(filePath, data);
        
        if (this.enableCache) {
            this.modelCache.set(pair.toUpperCase(), data);
        }
        
        console.log('Model metadata saved:', pair, filePath);
    }
    
    loadModelMetadata(pair) {
        const cacheKey = pair.toUpperCase();
        
        // Check cache first
        if (this.enableCache && this.modelCache.has(cacheKey)) {
            return this.modelCache.get(cacheKey);
        }
        
        const filePath = path.join(this.modelsDir, `${pair.toLowerCase()}_model.json`);
        const data = this.readFileSecure(filePath);
        
        if (data && this.enableCache) {
            this.modelCache.set(cacheKey, data);
        }
        
        return data;
    }
    
    // 🆕 FIXED: Weight persistence methods without file:// URLs
    async saveModelWeights(pair, modelType, model) {
        try {
            const weightsDir = path.join(this.weightsDir, `${pair.toLowerCase()}_${modelType}`);
            
            // Create weights directory if it doesn't exist
            if (!fs.existsSync(weightsDir)) {
                fs.mkdirSync(weightsDir, { recursive: true });
            }
            
            // 🆕 FIXED: Extract weights as JSON data instead of using model.save()
            const weights = model.model.getWeights();
            const weightsData = [];
            
            for (let i = 0; i < weights.length; i++) {
                const weightTensor = weights[i];
                const weightArray = await weightTensor.data();
                const shape = weightTensor.shape;
                
                weightsData.push({
                    data: Array.from(weightArray),
                    shape: shape,
                    dtype: weightTensor.dtype
                });
                
                
            }
            
            // Save weights as JSON
            const weightsPath = path.join(weightsDir, 'weights.json');
            await this.writeFileAtomic(weightsPath, {
                weights: weightsData,
                timestamp: Date.now(),
                version: '1.0.0'
            });
            
            // Save model configuration
            const configPath = path.join(weightsDir, 'config.json');
            await this.writeFileAtomic(configPath, {
                modelType: modelType,
                config: model.getModelSummary().config,
                architecture: model.getModelSummary().architecture,
                timestamp: Date.now(),
                version: '1.0.0'
            });
            
            // 🆕 FIXED: Save metadata with timestamp at root level for verification
            const metadata = {
                pair: pair.toUpperCase(),
                modelType: modelType,
                savedAt: Date.now(),
                timestamp: Date.now(), // 🆕 ADDED: Required for verification
                modelSummary: model.getModelSummary(),
                weightsCount: weightsData.length,
                version: '1.0.0',
                type: 'model_weights'
            };
            
            const metadataPath = path.join(weightsDir, 'metadata.json');
            await this.writeFileAtomic(metadataPath, metadata);
            
            // Cache weights info
            const cacheKey = `${pair.toUpperCase()}_${modelType}`;
            this.weightsCache.set(cacheKey, {
                hasWeights: true,
                savedAt: Date.now(),
                weightsDir: weightsDir,
                weightsCount: weightsData.length
            });
            
            console.log(`✅ Model weights saved for ${pair}:${modelType}`, {
                weightsDir: weightsDir,
                weightsCount: weightsData.length,
                modelParams: metadata.modelSummary.totalParams
            });
            
            return weightsDir;
            
        } catch (error) {
            console.error(`❌ Failed to save model weights for ${pair}:${modelType}`, { 
                error: error.message 
            });
            throw error;
        }
    }
    
    // 🆕 FIXED: Load weights from JSON data and rebuild model
    async loadModelWeights(pair, modelType, ModelClass, config) {
        try {
            const weightsDir = path.join(this.weightsDir, `${pair.toLowerCase()}_${modelType}`);
            const weightsPath = path.join(weightsDir, 'weights.json');
            const configPath = path.join(weightsDir, 'config.json');
            const metadataPath = path.join(weightsDir, 'metadata.json');
            
            // Check if weights exist
            if (!fs.existsSync(weightsPath) || !fs.existsSync(metadataPath)) {
                console.log(`No trained weights found for ${pair}:${modelType}`);
                return null;
            }
            
            // Load metadata to verify compatibility
            const metadata = this.readFileSecure(metadataPath);
            if (!metadata) {
                console.warn(`Invalid metadata for ${pair}:${modelType} weights`);
                return null;
            }
            
            // Check feature count compatibility
            if (metadata.modelSummary && metadata.modelSummary.config) {
                const savedFeatureCount = metadata.modelSummary.config.features;
                const currentFeatureCount = config.features;
                
                if (savedFeatureCount !== currentFeatureCount) {
                    console.warn(`Feature count mismatch for ${pair}:${modelType}. Saved: ${savedFeatureCount}, Current: ${currentFeatureCount}. Cannot load weights.`);
                    return null;
                }
            }
            
            // Load weights data
            const weightsData = this.readFileSecure(weightsPath);
            if (!weightsData || !weightsData.weights) {
                console.warn(`Invalid weights data for ${pair}:${modelType}`);
                return null;
            }
            
            // Create new model with same config
            const modelWrapper = new ModelClass(config);
            modelWrapper.buildModel();
            modelWrapper.compileModel();
            
            // 🆕 FIXED: Recreate tensors from saved data and set weights
            const tf = require('@tensorflow/tfjs');
            const weightTensors = [];
            
            for (const weightInfo of weightsData.weights) {
                const tensor = tf.tensor(weightInfo.data, weightInfo.shape, weightInfo.dtype);
                weightTensors.push(tensor);
            }
            
            // Set the weights on the model
            modelWrapper.model.setWeights(weightTensors);
            
            // Clean up weight tensors (model has its own copies now)
            weightTensors.forEach(tensor => tensor.dispose());
            
            // Cache weights info
            const cacheKey = `${pair.toUpperCase()}_${modelType}`;
            this.weightsCache.set(cacheKey, {
                hasWeights: true,
                loadedAt: Date.now(),
                weightsDir: weightsDir,
                metadata: metadata,
                weightsCount: weightsData.weights.length
            });
            
            console.log(`✅ Model weights loaded for ${pair}:${modelType}`, {
                weightsDir: weightsDir,
                savedAt: new Date(metadata.savedAt).toLocaleString(),
                weightsCount: weightsData.weights.length,
                modelParams: metadata.modelSummary.totalParams
            });
            
            return modelWrapper;
            
        } catch (error) {
            console.error(`❌ Failed to load model weights for ${pair}:${modelType}`, { 
                error: error.message 
            });
            return null;
        }
    }
    
    hasTrainedWeights(pair, modelType) {
        const cacheKey = `${pair.toUpperCase()}_${modelType}`;
        
        // Check cache first
        if (this.weightsCache.has(cacheKey)) {
            return this.weightsCache.get(cacheKey).hasWeights;
        }
        
        // Check filesystem
        const weightsDir = path.join(this.weightsDir, `${pair.toLowerCase()}_${modelType}`);
        const weightsPath = path.join(weightsDir, 'weights.json');
        const metadataPath = path.join(weightsDir, 'metadata.json');
        
        const hasWeights = fs.existsSync(weightsPath) && fs.existsSync(metadataPath);
        
        // Cache result
        this.weightsCache.set(cacheKey, {
            hasWeights: hasWeights,
            checkedAt: Date.now(),
            weightsDir: weightsDir
        });
        
        return hasWeights;
    }
    
    // Get list of all trained models
    getTrainedModelsList() {
        try {
            const trainedModels = [];
            
            if (!fs.existsSync(this.weightsDir)) {
                return trainedModels;
            }
            
            const weightDirs = fs.readdirSync(this.weightsDir);
            
            for (const dir of weightDirs) {
                const dirPath = path.join(this.weightsDir, dir);
                const metadataPath = path.join(dirPath, 'metadata.json');
                
                if (fs.existsSync(metadataPath)) {
                    try {
                        const metadata = this.readFileSecure(metadataPath);
                        if (metadata) {
                            trainedModels.push({
                                pair: metadata.pair,
                                modelType: metadata.modelType,
                                savedAt: metadata.savedAt,
                                modelParams: metadata.modelSummary?.totalParams || 0,
                                featureCount: metadata.modelSummary?.config?.features || 0,
                                weightsDir: dir,
                                weightsCount: metadata.weightsCount || 0
                            });
                        }
                    } catch (error) {
                        console.warn(`Failed to read metadata for ${dir}:`, error.message);
                    }
                }
            }
            
            // Sort by saved date (newest first)
            trainedModels.sort((a, b) => b.savedAt - a.savedAt);
            
            return trainedModels;
            
        } catch (error) {
            console.error('Failed to get trained models list', { error: error.message });
            return [];
        }
    }
    
    // Training history persistence
    async saveTrainingHistory(pair, trainingResults) {
        const filePath = path.join(this.trainingDir, `${pair.toLowerCase()}_training.json`);
        
        const data = {
            pair: pair.toUpperCase(),
            trainingResults,
            timestamp: Date.now(),
            version: '1.0.0',
            type: 'training_history'
        };
        
        await this.writeFileAtomic(filePath, data);
        
        if (this.enableCache) {
            this.trainingHistory.set(pair.toUpperCase(), data);
        }
        
        console.log('Training history saved:', pair, filePath);
    }
    
    loadTrainingHistory(pair) {
        const cacheKey = pair.toUpperCase();
        
        // Check cache first
        if (this.enableCache && this.trainingHistory.has(cacheKey)) {
            return this.trainingHistory.get(cacheKey);
        }
        
        const filePath = path.join(this.trainingDir, `${pair.toLowerCase()}_training.json`);
        const data = this.readFileSecure(filePath);
        
        if (data && this.enableCache) {
            this.trainingHistory.set(cacheKey, data);
        }
        
        return data;
    }
    
    // Prediction history persistence
    async savePredictionHistory(pair, predictions) {
        const filePath = path.join(this.predictionsDir, `${pair.toLowerCase()}_predictions.json`);
        
        // Load existing predictions and append new ones
        let existingData = this.readFileSecure(filePath);
        let allPredictions = [];
        
        if (existingData && existingData.predictions) {
            allPredictions = existingData.predictions;
        }
        
        // Add new predictions
        if (Array.isArray(predictions)) {
            allPredictions.push(...predictions);
        } else {
            allPredictions.push(predictions);
        }
        
        // Keep only recent predictions (last 1000)
        if (allPredictions.length > 1000) {
            allPredictions = allPredictions.slice(-1000);
        }
        
        const data = {
            pair: pair.toUpperCase(),
            predictions: allPredictions,
            count: allPredictions.length,
            timestamp: Date.now(),
            version: '1.0.0',
            type: 'prediction_history'
        };
        
        await this.writeFileAtomic(filePath, data);
        
        if (this.enableCache) {
            this.predictionCache.set(pair.toUpperCase(), data);
        }
        
        console.log('Prediction history saved:', pair, 'count:', allPredictions.length);
    }
    
    loadPredictionHistory(pair) {
        const cacheKey = pair.toUpperCase();
        
        // Check cache first
        if (this.enableCache && this.predictionCache.has(cacheKey)) {
            return this.predictionCache.get(cacheKey);
        }
        
        const filePath = path.join(this.predictionsDir, `${pair.toLowerCase()}_predictions.json`);
        const data = this.readFileSecure(filePath);
        
        if (data && this.enableCache) {
            this.predictionCache.set(cacheKey, data);
        }
        
        return data;
    }
    
    // Feature cache persistence
    async saveFeatureCache(pair, features) {
        const filePath = path.join(this.featuresDir, `${pair.toLowerCase()}_features.json`);
        
        const data = {
            pair: pair.toUpperCase(),
            features,
            timestamp: Date.now(),
            version: '1.0.0',
            type: 'feature_cache'
        };
        
        await this.writeFileAtomic(filePath, data);
        
        if (this.enableCache) {
            this.featureCache.set(pair.toUpperCase(), data);
        }
        
        console.log('Feature cache saved:', pair);
    }
    
    loadFeatureCache(pair) {
        const cacheKey = pair.toUpperCase();
        
        // Check cache first
        if (this.enableCache && this.featureCache.has(cacheKey)) {
            const cached = this.featureCache.get(cacheKey);
            // Check if cache is still fresh (5 minutes)
            if (Date.now() - cached.timestamp < 300000) {
                return cached;
            }
        }
        
        const filePath = path.join(this.featuresDir, `${pair.toLowerCase()}_features.json`);
        const data = this.readFileSecure(filePath);
        
        if (data && this.enableCache) {
            this.featureCache.set(cacheKey, data);
        }
        
        return data;
    }
    
    // Storage statistics
    getStorageStats() {
        const stats = {
            models: this.getDirectoryStats(this.modelsDir),
            weights: this.getDirectoryStats(this.weightsDir),
            training: this.getDirectoryStats(this.trainingDir),
            predictions: this.getDirectoryStats(this.predictionsDir),
            features: this.getDirectoryStats(this.featuresDir),
            cache: {
                models: this.modelCache.size,
                weights: this.weightsCache.size,
                training: this.trainingHistory.size,
                predictions: this.predictionCache.size,
                features: this.featureCache.size
            },
            totalSizeBytes: 0,
            trainedModels: this.getTrainedModelsList(),
            timestamp: Date.now()
        };
        
        stats.totalSizeBytes = stats.models.sizeBytes + 
                              stats.weights.sizeBytes +
                              stats.training.sizeBytes + 
                              stats.predictions.sizeBytes + 
                              stats.features.sizeBytes;
        
        return stats;
    }
    
    getDirectoryStats(directory) {
        const stats = {
            count: 0,
            sizeBytes: 0,
            files: []
        };
        
        try {
            if (!fs.existsSync(directory)) {
                return stats;
            }
            
            const items = fs.readdirSync(directory);
            
            items.forEach(item => {
                const itemPath = path.join(directory, item);
                const itemStat = fs.statSync(itemPath);
                
                if (itemStat.isFile() && (item.endsWith('.json') || item.endsWith('.bin'))) {
                    stats.count++;
                    stats.sizeBytes += itemStat.size;
                    stats.files.push({
                        name: item,
                        sizeBytes: itemStat.size,
                        lastModified: itemStat.mtime.toISOString()
                    });
                } else if (itemStat.isDirectory()) {
                    // For directories (like weights), count contents
                    const dirStats = this.getDirectoryStats(itemPath);
                    stats.count += dirStats.count;
                    stats.sizeBytes += dirStats.sizeBytes;
                    // Add directory entry
                    stats.files.push({
                        name: item,
                        type: 'directory',
                        sizeBytes: dirStats.sizeBytes,
                        lastModified: itemStat.mtime.toISOString(),
                        contents: dirStats.count
                    });
                }
            });
        } catch (error) {
            console.error('Failed to get directory stats:', directory, error.message);
        }
        
        return stats;
    }
    
    // Cleanup old files
    async cleanup(maxAgeHours = this.maxAgeHours) {
        const cutoffTime = Date.now() - (maxAgeHours * 60 * 60 * 1000);
        let cleanedCount = 0;
        
        const directories = [this.predictionsDir, this.featuresDir];
        
        for (const directory of directories) {
            try {
                if (!fs.existsSync(directory)) continue;
                
                const files = fs.readdirSync(directory);
                
                for (const file of files) {
                    if (file.endsWith('.json')) {
                        const filePath = path.join(directory, file);
                        const stats = fs.statSync(filePath);
                        
                        if (stats.mtime.getTime() < cutoffTime) {
                            fs.unlinkSync(filePath);
                            cleanedCount++;
                            console.log('Cleaned up old file:', filePath);
                        }
                    }
                }
            } catch (error) {
                console.error('Cleanup failed for directory:', directory, error.message);
            }
        }
        
        // Clear old cache entries
        this.clearOldCacheEntries(cutoffTime);
        
        console.log('Storage cleanup completed. Cleaned files:', cleanedCount, 'Max age hours:', maxAgeHours);
        
        return cleanedCount;
    }
    
    clearOldCacheEntries(cutoffTime) {
        const caches = [this.predictionCache, this.featureCache];
        
        caches.forEach(cache => {
            for (const [key, value] of cache.entries()) {
                if (value.timestamp < cutoffTime) {
                    cache.delete(key);
                }
            }
        });
    }
    
    // Force save all cached data
    async forceSave() {
        let savedCount = 0;
        
        try {
            // Save all cached models
            for (const [pair, data] of this.modelCache.entries()) {
                await this.saveModelMetadata(pair, data.modelInfo);
                savedCount++;
            }
            
            // Save all cached training history
            for (const [pair, data] of this.trainingHistory.entries()) {
                await this.saveTrainingHistory(pair, data.trainingResults);
                savedCount++;
            }
            
            // Save all cached predictions
            for (const [pair, data] of this.predictionCache.entries()) {
                const filePath = path.join(this.predictionsDir, `${pair.toLowerCase()}_predictions.json`);
                await this.writeFileAtomic(filePath, data);
                savedCount++;
            }
            
            // Save all cached features
            for (const [pair, data] of this.featureCache.entries()) {
                const filePath = path.join(this.featuresDir, `${pair.toLowerCase()}_features.json`);
                await this.writeFileAtomic(filePath, data);
                savedCount++;
            }
            
            console.log('Force save completed. Saved items:', savedCount);
            return savedCount;
            
        } catch (error) {
            console.error('Force save failed:', error.message);
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
                console.log('Periodic save completed');
            } catch (error) {
                console.error('Periodic save failed:', error.message);
            }
        }, this.saveInterval);
        
        console.log('Periodic save started. Interval:', this.saveInterval, 'ms');
    }
    
    // Stop periodic saving
    stopPeriodicSave() {
        if (this.saveIntervalId) {
            clearInterval(this.saveIntervalId);
            this.saveIntervalId = null;
            console.log('Periodic save stopped');
        }
    }
    
    // Graceful shutdown
    async shutdown() {
        console.log('ML Storage shutting down...');
        
        this.stopPeriodicSave();
        
        try {
            await this.forceSave();
            console.log('Final save completed during shutdown');
        } catch (error) {
            console.error('Final save failed during shutdown:', error.message);
        }
        
        // Clear caches
        this.modelCache.clear();
        this.predictionCache.clear();
        this.featureCache.clear();
        this.trainingHistory.clear();
        this.weightsCache.clear();
        
        console.log('ML Storage shutdown completed');
    }
}

module.exports = MLStorage;