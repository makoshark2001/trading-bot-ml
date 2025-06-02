const fs = require('fs');
const path = require('path');
const { Logger } = require('./Logger'); // Direct import from Logger file

class MLStorage {
    constructor(config = {}) {
        this.baseDir = config.baseDir || path.join(process.cwd(), 'data', 'ml');
        this.modelsDir = path.join(this.baseDir, 'models');
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
        
        this.initializeDirectories();
        this.startPeriodicSave();
        
        console.log('MLStorage initialized with config:', {
            baseDir: this.baseDir,
            enableCache: this.enableCache,
            saveInterval: this.saveInterval
        });
    }
    
    initializeDirectories() {
        const dirs = [this.baseDir, this.modelsDir, this.trainingDir, this.predictionsDir, this.featuresDir];
        
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
            training: this.getDirectoryStats(this.trainingDir),
            predictions: this.getDirectoryStats(this.predictionsDir),
            features: this.getDirectoryStats(this.featuresDir),
            cache: {
                models: this.modelCache.size,
                training: this.trainingHistory.size,
                predictions: this.predictionCache.size,
                features: this.featureCache.size
            },
            totalSizeBytes: 0,
            timestamp: Date.now()
        };
        
        stats.totalSizeBytes = stats.models.sizeBytes + 
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
            
            const files = fs.readdirSync(directory);
            
            files.forEach(file => {
                if (file.endsWith('.json') && !file.endsWith('.tmp') && !file.endsWith('.backup')) {
                    const filePath = path.join(directory, file);
                    const fileStat = fs.statSync(filePath);
                    
                    stats.count++;
                    stats.sizeBytes += fileStat.size;
                    stats.files.push({
                        name: file,
                        sizeBytes: fileStat.size,
                        lastModified: fileStat.mtime.toISOString()
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
        
        console.log('ML Storage shutdown completed');
    }
}

module.exports = MLStorage;