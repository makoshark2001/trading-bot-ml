const express = require('express');
const config = require('config');
const DataClient = require('../data/DataClient');
const FeatureExtractor = require('../data/FeatureExtractor');
const { Logger, MLStorage } = require('../utils');

class MLServer {
    constructor() {
        this.app = express();
        this.port = config.get('server.port') || 3001;
        this.startTime = Date.now();
        
        this.dataClient = null;
        this.featureExtractor = null;
        this.mlStorage = null;
        this.predictions = {}; // Cache recent predictions
        this.featureCounts = {}; // Track feature count per pair
        
        // Model configuration
        this.modelTypes = ['lstm', 'gru', 'cnn', 'transformer'];
        this.enabledModels = config.get('ml.ensemble.enabledModels') || this.modelTypes;
        this.ensembleStrategy = config.get('ml.ensemble.strategy') || 'weighted';
        
        // Training configuration - simplified
        this.autoTrainingEnabled = config.get('ml.training.autoTraining') !== false;
        this.periodicTrainingEnabled = config.get('ml.training.periodicTraining') !== false;
        this.trainingInterval = config.get('ml.training.interval') || 24 * 60 * 60 * 1000;
        this.minDataAgeForRetraining = config.get('ml.training.minDataAge') || 12 * 60 * 60 * 1000;
        this.trainingQueue = new Set();
        this.lastPeriodicTraining = {};
        
        this.initializeServices();
        this.setupMiddleware();
        this.setupRoutes();
    }
    
    initializeServices() {
        Logger.info('Initializing simplified ML services...');
        
        // Initialize data client
        this.dataClient = new DataClient();
        
        // Initialize feature extractor
        this.featureExtractor = new FeatureExtractor(config.get('ml.features'));
        
        // Initialize ML storage
        this.mlStorage = new MLStorage({
            baseDir: config.get('ml.storage.baseDir'),
            saveInterval: config.get('ml.storage.saveInterval'),
            maxAgeHours: config.get('ml.storage.maxAgeHours'),
            enableCache: config.get('ml.storage.enableCache')
        });
        
        Logger.info('Simplified ML services initialized successfully');
    }
    
    setupMiddleware() {
        this.app.use(express.json({ limit: '10mb' }));
        
        // CORS middleware
        this.app.use((req, res, next) => {
            res.header('Access-Control-Allow-Origin', '*');
            res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
            res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
            if (req.method === 'OPTIONS') {
                res.sendStatus(200);
            } else {
                next();
            }
        });
        
        // Request logging
        this.app.use((req, res, next) => {
            Logger.debug('API request', {
                method: req.method,
                url: req.url
            });
            next();
        });
        
        // Set response timeouts
        this.app.use((req, res, next) => {
            res.setTimeout(30000, () => {
                res.status(408).json({
                    error: 'Request timeout',
                    message: 'Request took too long to process'
                });
            });
            next();
        });
    }
    
    setupRoutes() {
        // Simple health check - no model operations
        this.app.get('/api/health', async (req, res) => {
            try {
                // Quick core health check with timeout
                let coreHealth = { status: 'unknown' };
                try {
                    coreHealth = await Promise.race([
                        this.dataClient.checkCoreHealth(),
                        new Promise((_, reject) => 
                            setTimeout(() => reject(new Error('Core health timeout')), 3000)
                        )
                    ]);
                } catch (error) {
                    Logger.warn('Core health check failed', { error: error.message });
                    coreHealth = { status: 'error', error: error.message };
                }
                
                const storageStats = this.mlStorage.getStorageStats();
                
                res.json({
                    status: 'healthy',
                    service: 'trading-bot-ml-auto-training',
                    timestamp: Date.now(),
                    uptime: this.getUptime(),
                    core: coreHealth,
                    training: {
                        autoTraining: {
                            enabled: this.autoTrainingEnabled,
                            description: 'Train models automatically on first use'
                        },
                        periodicTraining: {
                            enabled: this.periodicTrainingEnabled,
                            interval: this.trainingInterval,
                            intervalHours: this.trainingInterval / (60 * 60 * 1000)
                        },
                        currentlyTraining: {
                            count: this.trainingQueue.size,
                            models: Array.from(this.trainingQueue)
                        }
                    },
                    predictions: {
                        cached: Object.keys(this.predictions).length,
                        lastUpdate: this.getLastPredictionTime()
                    },
                    storage: {
                        enabled: true,
                        stats: {
                            totalSizeBytes: storageStats.totalSizeBytes,
                            models: storageStats.models.count,
                            weights: storageStats.weights.count,
                            trainedModels: storageStats.trainedModels.length
                        }
                    },
                    note: 'Simplified ML service - models load on demand'
                });
            } catch (error) {
                Logger.error('Health check failed', { error: error.message });
                res.status(500).json({
                    status: 'unhealthy',
                    service: 'trading-bot-ml-auto-training',
                    error: error.message,
                    timestamp: Date.now()
                });
            }
        });
        
        // Training status - simple response
        this.app.get('/api/training/status', (req, res) => {
            try {
                res.json({
                    training: {
                        autoTraining: {
                            enabled: this.autoTrainingEnabled,
                            description: 'Train models automatically on first use'
                        },
                        periodicTraining: {
                            enabled: this.periodicTrainingEnabled,
                            interval: this.trainingInterval,
                            intervalHours: this.trainingInterval / (60 * 60 * 1000)
                        },
                        currentlyTraining: {
                            count: this.trainingQueue.size,
                            models: Array.from(this.trainingQueue)
                        },
                        trainedModels: this.mlStorage.getTrainedModelsList().length
                    },
                    timestamp: Date.now()
                });
            } catch (error) {
                Logger.error('Failed to get training status', { error: error.message });
                res.status(500).json({
                    error: 'Failed to get training status',
                    message: error.message
                });
            }
        });
        
        // Training configuration
        this.app.get('/api/training/config', (req, res) => {
            res.json({
                autoTraining: {
                    enabled: this.autoTrainingEnabled,
                    description: 'Train models automatically on first use if no weights exist'
                },
                periodicTraining: {
                    enabled: this.periodicTrainingEnabled,
                    interval: this.trainingInterval,
                    intervalHours: this.trainingInterval / (60 * 60 * 1000),
                    minDataAge: this.minDataAgeForRetraining,
                    minDataAgeHours: this.minDataAgeForRetraining / (60 * 60 * 1000)
                },
                enabledModels: this.enabledModels,
                modelTypes: this.modelTypes,
                lastPeriodicTraining: this.lastPeriodicTraining,
                timestamp: Date.now()
            });
        });
        
        // Update training configuration
        this.app.post('/api/training/config', (req, res) => {
            try {
                const { autoTraining, periodicTraining, trainingInterval, minDataAge } = req.body;
                
                if (autoTraining !== undefined) {
                    this.autoTrainingEnabled = autoTraining;
                    Logger.info('Auto training configuration updated', { enabled: autoTraining });
                }
                
                if (periodicTraining !== undefined) {
                    this.periodicTrainingEnabled = periodicTraining;
                    Logger.info('Periodic training configuration updated', { enabled: periodicTraining });
                }
                
                if (trainingInterval) {
                    this.trainingInterval = trainingInterval;
                    Logger.info('Training interval updated', { 
                        interval: trainingInterval,
                        hours: trainingInterval / (60 * 60 * 1000)
                    });
                }
                
                if (minDataAge) {
                    this.minDataAgeForRetraining = minDataAge;
                    Logger.info('Min data age for retraining updated', { 
                        minDataAge: minDataAge,
                        hours: minDataAge / (60 * 60 * 1000)
                    });
                }
                
                res.json({
                    success: true,
                    message: 'Training configuration updated',
                    config: {
                        autoTraining: this.autoTrainingEnabled,
                        periodicTraining: this.periodicTrainingEnabled,
                        trainingInterval: this.trainingInterval,
                        minDataAge: this.minDataAgeForRetraining
                    },
                    timestamp: Date.now()
                });
                
            } catch (error) {
                Logger.error('Failed to update training configuration', { error: error.message });
                res.status(500).json({
                    error: 'Failed to update training configuration',
                    message: error.message
                });
            }
        });
        
        // Simple model status - no model loading
        this.app.get('/api/models/:pair/status', async (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                
                // Quick feature count check without loading models
                let featureCount = 'unknown';
                try {
                    const pairData = await Promise.race([
                        this.dataClient.getPairData(pair),
                        new Promise((_, reject) => 
                            setTimeout(() => reject(new Error('Data fetch timeout')), 5000)
                        )
                    ]);
                    const features = this.featureExtractor.extractFeatures(pairData);
                    featureCount = features.features.length;
                    this.featureCounts[pair] = featureCount;
                } catch (error) {
                    Logger.warn('Failed to get feature count', { error: error.message });
                }
                
                const trainedModels = {};
                for (const modelType of this.enabledModels) {
                    const hasWeights = this.mlStorage.hasTrainedWeights(pair, modelType);
                    const isTraining = this.trainingQueue.has(`${pair}_${modelType}`);
                    
                    trainedModels[modelType] = {
                        hasModel: false, // Models not loaded yet
                        hasTrainedWeights: hasWeights,
                        isTraining: isTraining,
                        usingTrainedWeights: hasWeights
                    };
                }
                
                res.json({
                    pair,
                    featureCount: featureCount,
                    individual: trainedModels,
                    training: {
                        autoTraining: this.autoTrainingEnabled,
                        periodicTraining: this.periodicTrainingEnabled,
                        currentlyTraining: this.trainingQueue.has(pair)
                    },
                    ensemble: {
                        hasEnsemble: false, // Not loaded yet
                        enabledModels: this.enabledModels
                    },
                    weightPersistence: {
                        enabled: true,
                        trainedModelsCount: Object.values(trainedModels).filter(m => m.hasTrainedWeights).length,
                        totalModelsCount: this.enabledModels.length
                    },
                    note: 'Models will be loaded on first prediction request',
                    timestamp: Date.now()
                });
            } catch (error) {
                Logger.error(`Model status failed for ${req.params.pair}`, { 
                    error: error.message 
                });
                res.status(500).json({
                    error: 'Model status failed',
                    message: error.message,
                    pair: req.params.pair.toUpperCase()
                });
            }
        });
        
        // Storage statistics - this already works
        this.app.get('/api/storage/stats', (req, res) => {
            try {
                const stats = this.mlStorage.getStorageStats();
                res.json({
                    storage: stats,
                    timestamp: Date.now()
                });
            } catch (error) {
                Logger.error('Failed to get storage stats', { error: error.message });
                res.status(500).json({
                    error: 'Failed to get storage statistics',
                    message: error.message
                });
            }
        });
        
        // Force save storage
        this.app.post('/api/storage/save', async (req, res) => {
            try {
                const savedCount = await this.mlStorage.forceSave();
                res.json({
                    message: 'Storage force save completed',
                    savedCount: savedCount,
                    timestamp: Date.now()
                });
            } catch (error) {
                Logger.error('Storage force save failed', { error: error.message });
                res.status(500).json({
                    error: 'Storage force save failed',
                    message: error.message
                });
            }
        });
        
        // Cleanup storage
        this.app.post('/api/storage/cleanup', async (req, res) => {
            try {
                const maxAgeHours = req.body.maxAgeHours || this.mlStorage.maxAgeHours;
                const cleanedCount = await this.mlStorage.cleanup(maxAgeHours);
                res.json({
                    message: 'Storage cleanup completed',
                    cleanedCount: cleanedCount,
                    maxAgeHours: maxAgeHours,
                    timestamp: Date.now()
                });
            } catch (error) {
                Logger.error('Storage cleanup failed', { error: error.message });
                res.status(500).json({
                    error: 'Storage cleanup failed',
                    message: error.message
                });
            }
        });
        
        // Get trained models list
        this.app.get('/api/models/trained', (req, res) => {
            try {
                const trainedModels = this.mlStorage.getTrainedModelsList();
                
                res.json({
                    trainedModels,
                    count: trainedModels.length,
                    timestamp: Date.now()
                });
                
            } catch (error) {
                Logger.error('Failed to get trained models list', { error: error.message });
                res.status(500).json({
                    error: 'Failed to get trained models list',
                    message: error.message
                });
            }
        });
        
        // Simple prediction endpoint - returns mock data for now
        this.app.get('/api/predictions/:pair', async (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                const strategy = req.query.strategy || this.ensembleStrategy;
                
                // For now, return a mock prediction to test the endpoint
                const mockPrediction = {
                    prediction: 0.5 + (Math.random() - 0.5) * 0.4, // Random between 0.3-0.7
                    confidence: 0.3 + Math.random() * 0.4, // Random between 0.3-0.7
                    direction: Math.random() > 0.5 ? 'up' : 'down',
                    signal: 'HOLD',
                    ensemble: {
                        strategy: strategy,
                        modelCount: 0,
                        individualPredictions: {},
                        note: 'Mock prediction - models not loaded yet'
                    }
                };
                
                // Cache the prediction
                this.predictions[pair] = {
                    ...mockPrediction,
                    timestamp: Date.now(),
                    type: 'mock'
                };
                
                res.json({
                    pair,
                    prediction: mockPrediction,
                    ensemble: true,
                    strategy: strategy,
                    timestamp: Date.now(),
                    cached: false,
                    note: 'This is a mock prediction. Actual ML models will be implemented in the next phase.',
                    autoTraining: {
                        enabled: this.autoTrainingEnabled,
                        currentlyTraining: false
                    }
                });
                
            } catch (error) {
                Logger.error(`Prediction failed for ${req.params.pair}`, { 
                    error: error.message 
                });
                res.status(500).json({
                    error: 'Prediction failed',
                    message: error.message,
                    pair: req.params.pair.toUpperCase()
                });
            }
        });
        
        // Individual model predictions - mock for now
        this.app.get('/api/models/:pair/:modelType/predict', async (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                const modelType = req.params.modelType.toLowerCase();
                
                if (!this.enabledModels.includes(modelType)) {
                    return res.status(400).json({
                        error: 'Invalid model type',
                        message: `Model type ${modelType} not supported. Available: ${this.enabledModels.join(', ')}`
                    });
                }
                
                // Mock individual model prediction
                const prediction = 0.5 + (Math.random() - 0.5) * 0.4;
                const confidence = Math.abs(prediction - 0.5) * 2;
                
                const result = {
                    prediction: prediction,
                    confidence: confidence,
                    direction: prediction > 0.5 ? 'up' : 'down',
                    signal: confidence > 0.6 ? (prediction > 0.5 ? 'BUY' : 'SELL') : 'HOLD',
                    modelType: modelType,
                    note: 'Mock prediction - actual model not loaded yet'
                };
                
                res.json({
                    pair,
                    modelType,
                    prediction: result,
                    usingTrainedWeights: this.mlStorage.hasTrainedWeights(pair, modelType),
                    autoTraining: {
                        enabled: this.autoTrainingEnabled,
                        isTraining: this.trainingQueue.has(`${pair}_${modelType}`)
                    },
                    timestamp: Date.now()
                });
                
            } catch (error) {
                Logger.error(`Individual model prediction failed for ${req.params.pair}:${req.params.modelType}`, { 
                    error: error.message 
                });
                res.status(500).json({
                    error: 'Model prediction failed',
                    message: error.message,
                    pair: req.params.pair.toUpperCase(),
                    modelType: req.params.modelType.toLowerCase()
                });
            }
        });
        
        // Rebuild models endpoint
        this.app.post('/api/models/:pair/rebuild', async (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                
                // For now, just clear any cached data
                delete this.predictions[pair];
                delete this.featureCounts[pair];
                
                res.json({
                    success: true,
                    message: `Models cleared for ${pair} - will be rebuilt on next prediction request`,
                    pair: pair,
                    newFeatureCount: 'will be determined on next use',
                    rebuiltModels: this.enabledModels,
                    timestamp: Date.now()
                });
                
            } catch (error) {
                Logger.error(`Model rebuild failed for ${req.params.pair}`, { 
                    error: error.message 
                });
                res.status(500).json({
                    error: 'Model rebuild failed',
                    message: error.message,
                    pair: req.params.pair.toUpperCase()
                });
            }
        });
    }
    
    // Utility methods
    getUptime() {
        const uptimeMs = Date.now() - this.startTime;
        const hours = Math.floor(uptimeMs / 3600000);
        const minutes = Math.floor((uptimeMs % 3600000) / 60000);
        const seconds = Math.floor((uptimeMs % 60000) / 1000);
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }
    
    getLastPredictionTime() {
        const times = Object.values(this.predictions).map(p => p.timestamp);
        return times.length > 0 ? Math.max(...times) : null;
    }
    
    async start() {
        try {
            Logger.info('Starting Simplified ML Server...');
            
            // Don't wait for core service - just start the server
            Logger.info('Starting server without waiting for core...');
            
            // Start HTTP server
            this.server = this.app.listen(this.port, () => {
                Logger.info(`Simplified ML Server running at http://localhost:${this.port}`);
                console.log(`🤖 Simplified ML API available at: http://localhost:${this.port}/api`);
                console.log(`📊 Health check: http://localhost:${this.port}/api/health`);
                console.log(`🔮 Mock predictions: http://localhost:${this.port}/api/predictions/RVN`);
                console.log(`⏰ Training status: http://localhost:${this.port}/api/training/status`);
                console.log(`⚙️ Training config: http://localhost:${this.port}/api/training/config`);
                console.log(`💾 Storage stats: http://localhost:${this.port}/api/storage/stats`);
                console.log(`🔧 Individual models: http://localhost:${this.port}/api/models/RVN/lstm/predict`);
                console.log('');
                console.log('🚀 Features Available:');
                console.log(`   • Fast response times (no model loading)`);
                console.log(`   • Mock predictions for testing`);
                console.log(`   • Storage system working`);
                console.log(`   • Training configuration`);
                console.log(`   • All API endpoints responsive`);
                console.log('');
                console.log('📝 Note: This is a simplified version with mock predictions.');
                console.log('   Real ML models will be implemented in the next phase.');
            });
            
        } catch (error) {
            Logger.error('Failed to start Simplified ML server', { error: error.message });
            process.exit(1);
        }
    }
    
    async stop() {
        Logger.info('Stopping Simplified ML Server...');
        
        // Clear training queue
        this.trainingQueue.clear();
        
        // Shutdown storage system gracefully
        if (this.mlStorage) {
            await this.mlStorage.shutdown();
        }
        
        if (this.server) {
            this.server.close();
        }
        
        Logger.info('Simplified ML Server stopped');
    }
}

module.exports = MLServer;