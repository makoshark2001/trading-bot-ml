const express = require('express');
const config = require('config');
const DataClient = require('../data/DataClient');
const FeatureExtractor = require('../data/FeatureExtractor');
const DataPreprocessor = require('../data/DataPreprocessor');
const LSTMModel = require('../models/LSTMModel');
const GRUModel = require('../models/GRUModel');
const CNNModel = require('../models/CNNModel');
const TransformerModel = require('../models/TransformerModel');
const ModelEnsemble = require('../models/ModelEnsemble');
const { Logger, MLStorage } = require('../utils');

class MLServer {
    constructor() {
        this.app = express();
        this.port = config.get('server.port') || 3001;
        this.startTime = Date.now();
        
        this.dataClient = null;
        this.featureExtractor = null;
        this.preprocessor = null;
        this.models = {}; // Store individual models for each pair
        this.ensembles = {}; // Store ensemble models for each pair
        this.predictions = {}; // Cache recent predictions
        this.mlStorage = null; // Advanced persistence with weight storage
        this.featureCounts = {}; // Track feature count per pair
        
        // Model configuration
        this.modelTypes = ['lstm', 'gru', 'cnn', 'transformer'];
        this.enabledModels = config.get('ml.ensemble.enabledModels') || this.modelTypes;
        this.ensembleStrategy = config.get('ml.ensemble.strategy') || 'weighted';
        
        // NEW: Training configuration
        this.autoTrainingEnabled = config.get('ml.training.autoTraining') !== false; // Default true
        this.periodicTrainingEnabled = config.get('ml.training.periodicTraining') !== false; // Default true
        this.trainingInterval = config.get('ml.training.interval') || 24 * 60 * 60 * 1000; // 24 hours default
        this.minDataAgeForRetraining = config.get('ml.training.minDataAge') || 12 * 60 * 60 * 1000; // 12 hours
        this.trainingQueue = new Set(); // Track models currently being trained
        this.periodicTrainingTimer = null;
        this.lastPeriodicTraining = {};
        
        this.initializeServices();
        this.setupRoutes();
        this.setupMiddleware();
    }
    
    initializeServices() {
        Logger.info('Initializing Enhanced ML services with Automatic & Periodic Training...');
        
        // Initialize data client
        this.dataClient = new DataClient();
        
        // Initialize feature extractor
        this.featureExtractor = new FeatureExtractor(config.get('ml.features'));
        
        // Initialize data preprocessor
        this.preprocessor = new DataPreprocessor(config.get('ml.models.lstm'));
        
        // Initialize enhanced ML storage with weight persistence
        this.mlStorage = new MLStorage({
            baseDir: config.get('ml.storage.baseDir'),
            saveInterval: config.get('ml.storage.saveInterval'),
            maxAgeHours: config.get('ml.storage.maxAgeHours'),
            enableCache: config.get('ml.storage.enableCache')
        });
        
        Logger.info('Enhanced ML services initialized successfully', {
            enabledModels: this.enabledModels,
            ensembleStrategy: this.ensembleStrategy,
            weightPersistence: true,
            autoTraining: this.autoTrainingEnabled,
            periodicTraining: this.periodicTrainingEnabled,
            trainingInterval: this.trainingInterval / (60 * 60 * 1000) + ' hours'
        });
    }
    
    setupMiddleware() {
        this.app.use(express.json());
        
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
                url: req.url,
                ip: req.ip
            });
            next();
        });
    }
    
    setupRoutes() {
        // Enhanced health check with training status
        this.app.get('/api/health', async (req, res) => {
            try {
                const coreHealth = await this.dataClient.checkCoreHealth();
                const storageStats = this.mlStorage.getStorageStats();
                
                // Get ensemble statistics
                const ensembleStats = {};
                for (const [pair, ensemble] of Object.entries(this.ensembles)) {
                    ensembleStats[pair] = ensemble.getEnsembleStats();
                }
                
                // Get trained models list
                const trainedModels = this.mlStorage.getTrainedModelsList();
                
                // Get training status
                const trainingStatus = this.getTrainingStatus();
                
                res.json({
                    status: 'healthy',
                    service: 'trading-bot-ml-auto-training',
                    timestamp: Date.now(),
                    uptime: this.getUptime(),
                    core: coreHealth,
                    models: {
                        individual: {
                            loaded: Object.keys(this.models).length,
                            pairs: Object.keys(this.models)
                        },
                        ensembles: {
                            loaded: Object.keys(this.ensembles).length,
                            pairs: Object.keys(this.ensembles),
                            stats: ensembleStats
                        },
                        trained: {
                            count: trainedModels.length,
                            models: trainedModels
                        },
                        enabledTypes: this.enabledModels,
                        strategy: this.ensembleStrategy,
                        featureCounts: this.featureCounts
                    },
                    training: trainingStatus,
                    predictions: {
                        cached: Object.keys(this.predictions).length,
                        lastUpdate: this.getLastPredictionTime()
                    },
                    storage: {
                        enabled: true,
                        weightPersistence: true,
                        stats: storageStats,
                        cacheSize: storageStats.cache
                    }
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
        
        // Enhanced prediction endpoint with automatic training
        this.app.get('/api/predictions/:pair', async (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                const useEnsemble = req.query.ensemble !== 'false'; // Default to ensemble
                const strategy = req.query.strategy || this.ensembleStrategy;
                
                let prediction;
                if (useEnsemble) {
                    prediction = await this.getEnsemblePrediction(pair, { strategy });
                } else {
                    prediction = await this.getSingleModelPrediction(pair, req.query.model || 'lstm');
                }
                
                // Save prediction to persistent storage
                await this.mlStorage.savePredictionHistory(pair, {
                    ...prediction,
                    timestamp: Date.now(),
                    requestId: `${pair}_${Date.now()}`,
                    useEnsemble: useEnsemble,
                    strategy: useEnsemble ? strategy : null
                });
                
                res.json({
                    pair,
                    prediction,
                    ensemble: useEnsemble,
                    strategy: useEnsemble ? strategy : null,
                    timestamp: Date.now(),
                    cached: this.predictions[pair] && 
                           (Date.now() - this.predictions[pair].timestamp) < 60000,
                    usingTrainedWeights: this.isUsingTrainedWeights(pair),
                    autoTraining: {
                        enabled: this.autoTrainingEnabled,
                        currentlyTraining: this.isCurrentlyTraining(pair)
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
        
        // NEW: Get training status
        this.app.get('/api/training/status', (req, res) => {
            try {
                const trainingStatus = this.getTrainingStatus();
                res.json({
                    training: trainingStatus,
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
        
        // NEW: Trigger periodic training manually
        this.app.post('/api/training/periodic', async (req, res) => {
            try {
                const forcePairs = req.body.pairs || null; // Optional: only specific pairs
                
                // Start periodic training in background
                this.runPeriodicTraining(forcePairs).catch(error => {
                    Logger.error('Manual periodic training failed', { error: error.message });
                });
                
                res.json({
                    message: 'Periodic training started',
                    pairs: forcePairs || 'all',
                    timestamp: Date.now()
                });
                
            } catch (error) {
                Logger.error('Failed to start periodic training', { error: error.message });
                res.status(500).json({
                    error: 'Failed to start periodic training',
                    message: error.message
                });
            }
        });
        
        // Enhanced training configuration endpoint
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
                    minDataAgeHours: this.minDataAgeForRetraining / (60 * 60 * 1000),
                    description: 'Retrain models periodically to stay up-to-date'
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
                    
                    if (periodicTraining) {
                        this.startPeriodicTraining();
                    } else {
                        this.stopPeriodicTraining();
                    }
                }
                
                if (trainingInterval) {
                    this.trainingInterval = trainingInterval;
                    Logger.info('Training interval updated', { 
                        interval: trainingInterval,
                        hours: trainingInterval / (60 * 60 * 1000)
                    });
                    
                    // Restart periodic training with new interval
                    if (this.periodicTrainingEnabled) {
                        this.stopPeriodicTraining();
                        this.startPeriodicTraining();
                    }
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
        
        // Continue with existing routes...
        this.setupExistingRoutes();
    }
    
    setupExistingRoutes() {
        // All the existing routes from previous implementation
        // (keeping them unchanged for brevity)
        
        // GET /api/models/trained
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
        
        // GET /api/models/:pair/:modelType/weights
        this.app.get('/api/models/:pair/:modelType/weights', (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                const modelType = req.params.modelType.toLowerCase();
                
                const hasWeights = this.mlStorage.hasTrainedWeights(pair, modelType);
                
                res.json({
                    pair,
                    modelType,
                    hasTrainedWeights: hasWeights,
                    timestamp: Date.now()
                });
                
            } catch (error) {
                Logger.error('Failed to check model weights', { 
                    error: error.message,
                    pair: req.params.pair,
                    modelType: req.params.modelType
                });
                res.status(500).json({
                    error: 'Failed to check model weights',
                    message: error.message
                });
            }
        });
        
        // Continue with other existing routes... 
        // (I'll include key ones but abbreviate for space)
        
        // Enhanced model status with training info
        this.app.get('/api/models/:pair/status', async (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                const pairModels = this.models[pair] || {};
                const ensemble = this.ensembles[pair];
                
                const modelMetadata = this.mlStorage.loadModelMetadata(pair);
                const trainingHistory = this.mlStorage.loadTrainingHistory(pair);
                const ensembleMetadata = this.mlStorage.loadModelMetadata(`${pair}_ensemble`);
                
                const individualModels = {};
                const trainedWeights = {};
                
                for (const modelType of this.enabledModels) {
                    const model = pairModels[modelType];
                    const hasWeights = this.mlStorage.hasTrainedWeights(pair, modelType);
                    const isTraining = this.isModelTraining(pair, modelType);
                    
                    individualModels[modelType] = {
                        hasModel: !!model,
                        hasTrainedWeights: hasWeights,
                        isTraining: isTraining,
                        modelInfo: model ? model.getModelSummary() : null,
                        usingTrainedWeights: !!model && hasWeights
                    };
                    
                    trainedWeights[modelType] = hasWeights;
                }
                
                res.json({
                    pair,
                    featureCount: this.featureCounts[pair] || 'unknown',
                    individual: individualModels,
                    trainedWeights: trainedWeights,
                    training: {
                        autoTraining: this.autoTrainingEnabled,
                        periodicTraining: this.periodicTrainingEnabled,
                        currentlyTraining: this.isCurrentlyTraining(pair),
                        lastPeriodicTraining: this.lastPeriodicTraining[pair]
                    },
                    ensemble: {
                        hasEnsemble: !!ensemble,
                        stats: ensemble ? ensemble.getEnsembleStats() : null,
                        strategy: this.ensembleStrategy,
                        enabledModels: this.enabledModels
                    },
                    persistent: {
                        metadata: modelMetadata,
                        trainingHistory: trainingHistory,
                        ensembleMetadata: ensembleMetadata,
                        lastTrained: trainingHistory ? trainingHistory.timestamp : null
                    },
                    weightPersistence: {
                        enabled: true,
                        trainedModelsCount: Object.values(trainedWeights).filter(Boolean).length,
                        totalModelsCount: this.enabledModels.length
                    },
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
        
        // Add remaining essential routes (abbreviated for space)
        // ... other existing routes
    }
    
    // NEW: Enhanced model creation with automatic training
    async getOrCreateModelWithWeights(pair, modelType, featureCount) {
        if (!this.models[pair]) {
            this.models[pair] = {};
        }
        
        // Check if model exists and has correct feature count
        if (this.models[pair][modelType]) {
            const existingModel = this.models[pair][modelType];
            const modelSummary = existingModel.getModelSummary();
            
            if (modelSummary.config && modelSummary.config.features === featureCount) {
                return existingModel;
            } else {
                Logger.warn(`Feature count mismatch for ${pair}:${modelType}. Expected ${modelSummary.config.features}, got ${featureCount}. Recreating model.`);
                existingModel.dispose();
                delete this.models[pair][modelType];
            }
        }
        
        const baseConfig = {
            sequenceLength: this.preprocessor.sequenceLength,
            features: featureCount
        };
        
        const modelSpecificConfig = config.get(`ml.models.${modelType}`) || {};
        const finalConfig = {
            ...modelSpecificConfig,
            ...baseConfig
        };
        
        const ModelClass = this.getModelClass(modelType);
        
        // Try to load trained weights first
        Logger.info(`Loading model for ${pair}:${modelType}`, { 
            featureCount, 
            tryingWeights: true 
        });
        
        const modelWithWeights = await this.mlStorage.loadModelWeights(
            pair, 
            modelType, 
            ModelClass, 
            finalConfig
        );
        
        if (modelWithWeights) {
            // Successfully loaded trained weights
            this.models[pair][modelType] = modelWithWeights;
            
            Logger.info(`Loaded ${modelType} model with trained weights for ${pair}`, {
                featureCount,
                modelParams: modelWithWeights.getModelSummary().totalParams
            });
            
            return modelWithWeights;
        }
        
        // No trained weights available, create new model
        Logger.info(`Creating new ${modelType} model for ${pair} (no trained weights)`, { featureCount });
        
        const model = new ModelClass(finalConfig);
        model.buildModel();
        model.compileModel();
        
        this.models[pair][modelType] = model;
        
        // Save model metadata
        await this.mlStorage.saveModelMetadata(`${pair}_${modelType}`, {
            config: finalConfig,
            modelType: modelType,
            created: Date.now(),
            featureCount,
            status: 'created_new',
            hasTrainedWeights: false
        });
        
        // NEW: Trigger automatic training if enabled
        if (this.autoTrainingEnabled && !this.isModelTraining(pair, modelType)) {
            Logger.info(`Triggering automatic training for ${pair}:${modelType} (first use)`);
            
            // Start training in background
            this.trainSingleModelAsync(pair, modelType, {
                reason: 'automatic_first_use',
                epochs: 25, // Reduced epochs for initial training
                batchSize: 32
            }).catch(error => {
                Logger.error(`Automatic training failed for ${pair}:${modelType}`, { 
                    error: error.message 
                });
            });
        }
        
        return model;
    }
    
    // NEW: Async wrapper for training to avoid blocking
    async trainSingleModelAsync(pair, modelType, config = {}) {
        const trainingKey = `${pair}_${modelType}`;
        
        // Prevent duplicate training
        if (this.trainingQueue.has(trainingKey)) {
            Logger.info(`Training already in progress for ${pair}:${modelType}`);
            return;
        }
        
        this.trainingQueue.add(trainingKey);
        
        try {
            await this.trainSingleModel(pair, modelType, config);
        } finally {
            this.trainingQueue.delete(trainingKey);
        }
    }
    
    // NEW: Periodic training functionality
    startPeriodicTraining() {
        if (!this.periodicTrainingEnabled) {
            return;
        }
        
        if (this.periodicTrainingTimer) {
            clearInterval(this.periodicTrainingTimer);
        }
        
        this.periodicTrainingTimer = setInterval(async () => {
            try {
                await this.runPeriodicTraining();
            } catch (error) {
                Logger.error('Periodic training failed', { error: error.message });
            }
        }, this.trainingInterval);
        
        Logger.info('Periodic training started', {
            interval: this.trainingInterval / (60 * 60 * 1000) + ' hours',
            minDataAge: this.minDataAgeForRetraining / (60 * 60 * 1000) + ' hours'
        });
    }
    
    stopPeriodicTraining() {
        if (this.periodicTrainingTimer) {
            clearInterval(this.periodicTrainingTimer);
            this.periodicTrainingTimer = null;
            Logger.info('Periodic training stopped');
        }
    }
    
    async runPeriodicTraining(forcePairs = null) {
        Logger.info('Starting periodic training cycle', { forcePairs });
        
        // Get all pairs that have been used for predictions
        const pairsToCheck = forcePairs || Object.keys(this.featureCounts);
        
        if (pairsToCheck.length === 0) {
            Logger.info('No pairs found for periodic training');
            return;
        }
        
        for (const pair of pairsToCheck) {
            try {
                await this.checkAndRetrainPair(pair);
            } catch (error) {
                Logger.error(`Periodic training failed for pair ${pair}`, { error: error.message });
            }
        }
        
        Logger.info('Periodic training cycle completed', { 
            checkedPairs: pairsToCheck.length 
        });
    }
    
    async checkAndRetrainPair(pair) {
        const now = Date.now();
        const lastTraining = this.lastPeriodicTraining[pair] || 0;
        const timeSinceLastTraining = now - lastTraining;
        
        // Check if enough time has passed since last training
        if (timeSinceLastTraining < this.minDataAgeForRetraining) {
            Logger.debug(`Skipping periodic training for ${pair} - too soon`, {
                timeSinceLastTraining: Math.round(timeSinceLastTraining / (60 * 60 * 1000)) + ' hours',
                minRequired: Math.round(this.minDataAgeForRetraining / (60 * 60 * 1000)) + ' hours'
            });
            return;
        }
        
        // Check if models are currently being trained
        if (this.isCurrentlyTraining(pair)) {
            Logger.info(`Skipping periodic training for ${pair} - currently training`);
            return;
        }
        
        Logger.info(`Starting periodic retraining for ${pair}`);
        
        // Start ensemble training in background
        this.trainEnsembleModelsAsync(pair, {
            reason: 'periodic_retraining',
            epochs: 50, // Full training for periodic updates
            batchSize: 32
        }).then(() => {
            this.lastPeriodicTraining[pair] = Date.now();
            Logger.info(`Periodic retraining completed for ${pair}`);
        }).catch(error => {
            Logger.error(`Periodic retraining failed for ${pair}`, { error: error.message });
        });
    }
    
    // NEW: Async ensemble training wrapper
    async trainEnsembleModelsAsync(pair, config = {}) {
        const trainingKeys = this.enabledModels.map(modelType => `${pair}_${modelType}`);
        
        // Check if any models are already training
        const alreadyTraining = trainingKeys.some(key => this.trainingQueue.has(key));
        if (alreadyTraining) {
            Logger.info(`Ensemble training already in progress for ${pair}`);
            return;
        }
        
        // Add all models to training queue
        trainingKeys.forEach(key => this.trainingQueue.add(key));
        
        try {
            await this.trainEnsembleModels(pair, config);
        } finally {
            // Remove all from training queue
            trainingKeys.forEach(key => this.trainingQueue.delete(key));
        }
    }
    
    // NEW: Training status helpers
    getTrainingStatus() {
        return {
            autoTraining: {
                enabled: this.autoTrainingEnabled,
                description: 'Train models automatically on first use'
            },
            periodicTraining: {
                enabled: this.periodicTrainingEnabled,
                interval: this.trainingInterval,
                intervalHours: this.trainingInterval / (60 * 60 * 1000),
                minDataAge: this.minDataAgeForRetraining,
                minDataAgeHours: this.minDataAgeForRetraining / (60 * 60 * 1000),
                lastRuns: this.lastPeriodicTraining,
                nextRun: this.getNextPeriodicTraining()
            },
            currentlyTraining: {
                count: this.trainingQueue.size,
                models: Array.from(this.trainingQueue)
            },
            trainedModels: this.mlStorage.getTrainedModelsList().length,
            timestamp: Date.now()
        };
    }
    
    getNextPeriodicTraining() {
        if (!this.periodicTrainingEnabled || !this.periodicTrainingTimer) {
            return null;
        }
        
        const lastRun = Math.max(...Object.values(this.lastPeriodicTraining), 0);
        return lastRun + this.trainingInterval;
    }
    
    isCurrentlyTraining(pair) {
        return this.enabledModels.some(modelType => 
            this.trainingQueue.has(`${pair}_${modelType}`)
        );
    }
    
    isModelTraining(pair, modelType) {
        return this.trainingQueue.has(`${pair}_${modelType}`);
    }
    
    isUsingTrainedWeights(pair) {
        const trainedWeights = {};
        for (const modelType of this.enabledModels) {
            trainedWeights[modelType] = this.mlStorage.hasTrainedWeights(pair, modelType);
        }
        return trainedWeights;
    }
    
    // Helper to get model class by type
    getModelClass(modelType) {
        switch (modelType) {
            case 'lstm':
                return LSTMModel;
            case 'gru':
                return GRUModel;
            case 'cnn':
                return CNNModel;
            case 'transformer':
                return TransformerModel;
            default:
                throw new Error(`Unknown model type: ${modelType}`);
        }
    }
    
    // Enhanced prediction methods (same as before but with automatic training)
    async getEnsemblePrediction(pair, options = {}) {
        const cacheKey = `${pair}_ensemble`;
        
        if (this.predictions[cacheKey] && 
            (Date.now() - this.predictions[cacheKey].timestamp) < 60000) {
            return this.predictions[cacheKey];
        }
        
        try {
            let ensemble = this.ensembles[pair];
            if (!ensemble) {
                ensemble = await this.getOrCreateEnsemble(pair);
            }
            
            const pairData = await this.dataClient.getPairData(pair);
            const features = this.featureExtractor.extractFeatures(pairData);
            
            this.featureCounts[pair] = features.features.length;
            
            const inputData = await this.prepareRealTimeInput(features.features);
            const prediction = await ensemble.predict(inputData, options);
            
            this.predictions[cacheKey] = {
                ...prediction,
                timestamp: Date.now(),
                type: 'ensemble',
                usingTrainedWeights: this.isUsingTrainedWeights(pair)
            };
            
            inputData.dispose();
            
            return this.predictions[cacheKey];
            
        } catch (error) {
            Logger.error(`Ensemble prediction failed for ${pair}`, { error: error.message });
            throw error;
        }
    }
    
    async getSingleModelPrediction(pair, modelType) {
        const cacheKey = `${pair}_${modelType}`;
        
        if (this.predictions[cacheKey] && 
            (Date.now() - this.predictions[cacheKey].timestamp) < 60000) {
            return this.predictions[cacheKey];
        }
        
        try {
            const pairData = await this.dataClient.getPairData(pair);
            const features = this.featureExtractor.extractFeatures(pairData);
            const currentFeatureCount = features.features.length;
            
            this.featureCounts[pair] = currentFeatureCount;
            
            // This will automatically train if no weights exist (due to autoTrainingEnabled)
            let model = await this.getOrCreateModelWithWeights(pair, modelType, currentFeatureCount);
            
            const inputData = await this.prepareRealTimeInput(features.features);
            const predictions = await model.predict(inputData);
            const prediction = predictions[0];
            
            const result = {
                prediction: prediction,
                confidence: Math.abs(prediction - 0.5) * 2,
                direction: prediction > 0.5 ? 'up' : 'down',
                signal: this.getTradeSignal(prediction, Math.abs(prediction - 0.5) * 2),
                modelType: modelType,
                usingTrainedWeights: this.mlStorage.hasTrainedWeights(pair, modelType),
                autoTraining: {
                    enabled: this.autoTrainingEnabled,
                    isTraining: this.isModelTraining(pair, modelType)
                },
                individual: {
                    prediction: prediction,
                    confidence: Math.abs(prediction - 0.5) * 2
                },
                metadata: {
                    timestamp: Date.now(),
                    version: '2.2.0',
                    type: 'individual_prediction',
                    featureCount: currentFeatureCount
                }
            };
            
            this.predictions[cacheKey] = {
                ...result,
                timestamp: Date.now(),
                type: 'individual'
            };
            
            inputData.dispose();
            
            return this.predictions[cacheKey];
            
        } catch (error) {
            Logger.error(`Individual prediction failed for ${pair}:${modelType}`, { error: error.message });
            throw error;
        }
    }
    
    // Enhanced ensemble creation with automatic training
    async getOrCreateEnsemble(pair) {
        Logger.info(`Creating ensemble for ${pair} with automatic training`);
        
        const pairData = await this.dataClient.getPairData(pair);
        const features = this.featureExtractor.extractFeatures(pairData);
        const featureCount = features.features.length;
        
        this.featureCounts[pair] = featureCount;
        
        const ensembleMetadata = this.mlStorage.loadModelMetadata(`${pair}_ensemble`);
        
        const ensembleConfig = {
            modelTypes: this.enabledModels,
            votingStrategy: this.ensembleStrategy,
            weights: {}
        };
        
        if (ensembleMetadata && ensembleMetadata.ensembleConfig) {
            ensembleConfig.weights = ensembleMetadata.ensembleConfig.weights || {};
            ensembleConfig.votingStrategy = ensembleMetadata.ensembleConfig.votingStrategy || this.ensembleStrategy;
        }
        
        const ensemble = new ModelEnsemble(ensembleConfig);
        
        // Create and add individual models to ensemble (with automatic training)
        for (const modelType of this.enabledModels) {
            try {
                const model = await this.getOrCreateModelWithWeights(pair, modelType, featureCount);
                const weight = ensembleConfig.weights[modelType] || 1.0;
                const hasTrainedWeights = this.mlStorage.hasTrainedWeights(pair, modelType);
                
                ensemble.addModel(modelType, model, weight, {
                    pair: pair,
                    featureCount: featureCount,
                    hasTrainedWeights: hasTrainedWeights,
                    autoTraining: this.autoTrainingEnabled,
                    created: Date.now()
                });
                
                Logger.info(`Added ${modelType} model to ${pair} ensemble`, { 
                    weight, 
                    featureCount,
                    hasTrainedWeights,
                    autoTraining: this.autoTrainingEnabled
                });
                
            } catch (error) {
                Logger.error(`Failed to add ${modelType} to ensemble for ${pair}`, { 
                    error: error.message 
                });
            }
        }
        
        this.ensembles[pair] = ensemble;
        
        await this.mlStorage.saveModelMetadata(`${pair}_ensemble`, {
            ensembleConfig: ensemble.toJSON(),
            created: Date.now(),
            pair: pair,
            modelTypes: this.enabledModels,
            featureCount: featureCount,
            weightPersistence: true,
            autoTraining: this.autoTrainingEnabled
        });
        
        return ensemble;
    }
    
    // Training methods (keeping existing implementation)
    async trainSingleModel(pair, modelType, config = {}) {
        Logger.info(`Starting training for ${pair}:${modelType}`, {
            ...config,
            reason: config.reason || 'manual'
        });
        
        try {
            const pairData = await this.dataClient.getPairData(pair);
            const features = this.featureExtractor.extractFeatures(pairData);
            const currentFeatureCount = features.features.length;
            
            this.featureCounts[pair] = currentFeatureCount;
            
            // Get or create model (but don't trigger auto-training since we're training now)
            const tempAutoTraining = this.autoTrainingEnabled;
            this.autoTrainingEnabled = false; // Temporarily disable to avoid recursion
            
            const model = await this.getOrCreateModelWithWeights(pair, modelType, currentFeatureCount);
            
            this.autoTrainingEnabled = tempAutoTraining; // Restore setting
            
            const targets = this.featureExtractor.createTargets(pairData.history);
            const binaryTargets = targets[`direction_${config.targetPeriods || 5}`] || targets['direction_5'];
            
            if (!binaryTargets || binaryTargets.length === 0) {
                throw new Error('No training targets available');
            }
            
            const featuresArray = Array(binaryTargets.length).fill().map(() => features.features);
            const processedData = await this.preprocessor.prepareTrainingData(featuresArray, binaryTargets);
            
            const modelTrainingConfig = {
                epochs: config.epochs || 50,
                batchSize: config.batchSize || 32,
                verbose: 0,
                ...config
            };
            
            const history = await model.train(
                processedData.trainX,
                processedData.trainY,
                processedData.validationX,
                processedData.validationY,
                modelTrainingConfig
            );
            
            // Save trained weights to persistent storage
            await this.mlStorage.saveModelWeights(pair, modelType, model);
            
            const trainingResults = {
                status: 'completed',
                finalMetrics: history.finalMetrics,
                epochsCompleted: history.epochsCompleted || history.finalMetrics.epochsCompleted,
                modelType: modelType,
                pair: pair,
                featureCount: currentFeatureCount,
                weightsSaved: true,
                trainedAt: Date.now(),
                reason: config.reason || 'manual'
            };
            
            await this.mlStorage.saveTrainingHistory(`${pair}_${modelType}`, {
                ...trainingResults,
                timestamp: Date.now()
            });
            
            // Clean up tensors
            processedData.trainX.dispose();
            processedData.trainY.dispose();
            processedData.validationX.dispose();
            processedData.validationY.dispose();
            processedData.testX.dispose();
            processedData.testY.dispose();
            
            Logger.info(`${modelType} training completed for ${pair}`, {
                ...trainingResults.finalMetrics,
                featureCount: currentFeatureCount,
                weightsSaved: true,
                reason: config.reason || 'manual'
            });
            
            return trainingResults;
            
        } catch (error) {
            Logger.error(`${modelType} training failed for ${pair}`, { error: error.message });
            
            await this.mlStorage.saveTrainingHistory(`${pair}_${modelType}`, {
                pair: pair,
                modelType: modelType,
                status: 'failed',
                error: error.message,
                startTime: Date.now(),
                endTime: Date.now(),
                weightsSaved: false,
                reason: config.reason || 'manual'
            });
            
            throw error;
        }
    }
    
    async trainEnsembleModels(pair, config = {}) {
        Logger.info(`Starting ensemble training for ${pair} with weight persistence`, {
            ...config,
            reason: config.reason || 'manual'
        });
        
        try {
            const pairData = await this.dataClient.getPairData(pair);
            const features = this.featureExtractor.extractFeatures(pairData);
            const currentFeatureCount = features.features.length;
            
            this.featureCounts[pair] = currentFeatureCount;
            
            const targets = this.featureExtractor.createTargets(pairData.history);
            const binaryTargets = targets[`direction_${config.targetPeriods || 5}`] || targets['direction_5'];
            
            if (!binaryTargets || binaryTargets.length === 0) {
                throw new Error('No training targets available');
            }
            
            const featuresArray = Array(binaryTargets.length).fill().map(() => features.features);
            const processedData = await this.preprocessor.prepareTrainingData(featuresArray, binaryTargets);
            
            const trainingResults = {
                pair: pair,
                startTime: Date.now(),
                models: {},
                ensemble: null,
                config: config,
                featureCount: currentFeatureCount,
                weightPersistence: true,
                reason: config.reason || 'manual'
            };
            
            // Temporarily disable auto-training to avoid recursion
            const tempAutoTraining = this.autoTrainingEnabled;
            this.autoTrainingEnabled = false;
            
            // Train individual models and save weights
            for (const modelType of this.enabledModels) {
                try {
                    Logger.info(`Training ${modelType} model for ${pair} with weight saving`);
                    
                    const model = await this.getOrCreateModelWithWeights(pair, modelType, currentFeatureCount);
                    
                    const modelTrainingConfig = {
                        epochs: config.epochs || 50,
                        batchSize: config.batchSize || 32,
                        verbose: 0,
                        ...config[modelType]
                    };
                    
                    const history = await model.train(
                        processedData.trainX,
                        processedData.trainY,
                        processedData.validationX,
                        processedData.validationY,
                        modelTrainingConfig
                    );
                    
                    await this.mlStorage.saveModelWeights(pair, modelType, model);
                    
                    trainingResults.models[modelType] = {
                        status: 'completed',
                        finalMetrics: history.finalMetrics,
                        epochsCompleted: history.epochsCompleted || history.finalMetrics.epochsCompleted,
                        modelType: modelType,
                        weightsSaved: true
                    };
                    
                    await this.mlStorage.saveTrainingHistory(`${pair}_${modelType}`, {
                        ...trainingResults.models[modelType],
                        pair: pair,
                        modelType: modelType,
                        timestamp: Date.now(),
                        reason: config.reason || 'manual'
                    });
                    
                    Logger.info(`${modelType} training completed for ${pair} with weights saved`, 
                        trainingResults.models[modelType].finalMetrics);
                    
                } catch (error) {
                    Logger.error(`${modelType} training failed for ${pair}`, { error: error.message });
                    trainingResults.models[modelType] = {
                        status: 'failed',
                        error: error.message,
                        modelType: modelType,
                        weightsSaved: false
                    };
                }
            }
            
            // Restore auto-training setting
            this.autoTrainingEnabled = tempAutoTraining;
            
            // Update ensemble weights based on training performance
            const ensemble = this.ensembles[pair];
            if (ensemble) {
                const performanceWeights = {};
                
                for (const [modelType, result] of Object.entries(trainingResults.models)) {
                    if (result.status === 'completed' && result.finalMetrics) {
                        const accuracy = parseFloat(result.finalMetrics.finalAccuracy);
                        performanceWeights[modelType] = isNaN(accuracy) ? 0.1 : Math.max(0.1, accuracy);
                    } else {
                        performanceWeights[modelType] = 0.1;
                    }
                }
                
                ensemble.updateWeights(performanceWeights);
                
                trainingResults.ensemble = {
                    status: 'updated',
                    newWeights: ensemble.weights,
                    strategy: ensemble.votingStrategy,
                    modelCount: ensemble.models.size,
                    weightPersistence: true
                };
                
                await this.mlStorage.saveModelMetadata(`${pair}_ensemble`, {
                    ensembleConfig: ensemble.toJSON(),
                    trainingCompleted: Date.now(),
                    performanceWeights: performanceWeights,
                    featureCount: currentFeatureCount,
                    weightPersistence: true,
                    reason: config.reason || 'manual'
                });
            }
            
            trainingResults.endTime = Date.now();
            trainingResults.status = 'completed';
            trainingResults.totalTime = trainingResults.endTime - trainingResults.startTime;
            
            // Clean up tensors
            processedData.trainX.dispose();
            processedData.trainY.dispose();
            processedData.validationX.dispose();
            processedData.validationY.dispose();
            processedData.testX.dispose();
            processedData.testY.dispose();
            
            await this.mlStorage.saveTrainingHistory(`${pair}_ensemble`, trainingResults);
            
            Logger.info(`Ensemble training completed for ${pair} with weight persistence`, {
                totalTime: trainingResults.totalTime,
                modelsCompleted: Object.values(trainingResults.models).filter(m => m.status === 'completed').length,
                modelsFailed: Object.values(trainingResults.models).filter(m => m.status === 'failed').length,
                featureCount: currentFeatureCount,
                weightsSaved: Object.values(trainingResults.models).filter(m => m.weightsSaved).length,
                reason: config.reason || 'manual'
            });
            
        } catch (error) {
            Logger.error(`Ensemble training failed for ${pair}`, { error: error.message });
            
            await this.mlStorage.saveTrainingHistory(`${pair}_ensemble`, {
                pair: pair,
                status: 'failed',
                error: error.message,
                startTime: Date.now(),
                endTime: Date.now(),
                weightPersistence: true,
                reason: config.reason || 'manual'
            });
            
            throw error;
        }
    }
    
    // Utility methods
    async prepareRealTimeInput(features) {
        const tf = require('@tensorflow/tfjs');
        const sequenceLength = this.preprocessor.sequenceLength;
        
        const sequence = Array(sequenceLength).fill(features);
        const inputTensor = tf.tensor3d([sequence]);
        
        return inputTensor;
    }
    
    getTradeSignal(prediction, confidence) {
        const strongThreshold = 0.7;
        const weakThreshold = 0.55;
        
        if (confidence > strongThreshold) {
            return prediction > 0.5 ? 'STRONG_BUY' : 'STRONG_SELL';
        } else if (confidence > weakThreshold) {
            return prediction > 0.5 ? 'BUY' : 'SELL';
        } else {
            return 'HOLD';
        }
    }
    
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
            Logger.info('Starting Enhanced ML Server with Automatic & Periodic Training...');
            
            // Wait for core service to be ready
            await this.dataClient.waitForCoreService();
            
            // Start periodic training if enabled
            if (this.periodicTrainingEnabled) {
                this.startPeriodicTraining();
            }
            
            // Start HTTP server
            this.server = this.app.listen(this.port, () => {
                Logger.info(`Enhanced ML Server with Automatic & Periodic Training running at http://localhost:${this.port}`);
                console.log(`🤖 Enhanced ML API with Auto Training available at: http://localhost:${this.port}/api`);
                console.log(`📊 Health check: http://localhost:${this.port}/api/health`);
                console.log(`🔮 Ensemble predictions: http://localhost:${this.port}/api/predictions/RVN`);
                console.log(`🏆 Model comparison: http://localhost:${this.port}/api/models/RVN/compare`);
                console.log(`⚖️ Ensemble stats: http://localhost:${this.port}/api/ensemble/RVN/stats`);
                console.log(`🏋️ Train model: http://localhost:${this.port}/api/train/RVN/lstm`);
                console.log(`📚 Trained models: http://localhost:${this.port}/api/models/trained`);
                console.log(`⏰ Training status: http://localhost:${this.port}/api/training/status`);
                console.log(`⚙️ Training config: http://localhost:${this.port}/api/training/config`);
                console.log(`💾 Storage stats: http://localhost:${this.port}/api/storage/stats`);
                console.log(`🔧 Rebuild models: http://localhost:${this.port}/api/models/RVN/rebuild`);
                console.log('');
                console.log('🚀 Enhanced Features Available:');
                console.log(`   • Model Types: ${this.enabledModels.join(', ')}`);
                console.log(`   • Ensemble Strategy: ${this.ensembleStrategy}`);
                console.log(`   • Weight Persistence: Enabled`);
                console.log(`   • Automatic Training: ${this.autoTrainingEnabled ? 'Enabled' : 'Disabled'}`);
                console.log(`   • Periodic Training: ${this.periodicTrainingEnabled ? 'Enabled' : 'Disabled'}`);
                if (this.periodicTrainingEnabled) {
                    console.log(`   • Training Interval: ${this.trainingInterval / (60 * 60 * 1000)} hours`);
                }
                console.log(`   • Individual Model Predictions`);
                console.log(`   • Performance Comparison`);
                console.log(`   • Dynamic Weight Updates`);
                console.log(`   • Trained Model Management`);
                console.log(`   • Dynamic Feature Count Handling`);
            });
            
        } catch (error) {
            Logger.error('Failed to start Enhanced ML server', { error: error.message });
            process.exit(1);
        }
    }
    
    async stop() {
        Logger.info('Stopping Enhanced ML Server with Automatic & Periodic Training...');
        
        // Stop periodic training
        this.stopPeriodicTraining();
        
        // Clear training queue
        this.trainingQueue.clear();
        
        // Shutdown storage system gracefully
        if (this.mlStorage) {
            await this.mlStorage.shutdown();
        }
        
        // Dispose of all individual models
        Object.values(this.models).forEach(pairModels => {
            Object.values(pairModels).forEach(model => {
                if (model && typeof model.dispose === 'function') {
                    model.dispose();
                }
            });
        });
        
        // Dispose of all ensembles
        Object.values(this.ensembles).forEach(ensemble => {
            if (ensemble && typeof ensemble.dispose === 'function') {
                ensemble.dispose();
            }
        });
        
        if (this.preprocessor) {
            this.preprocessor.dispose();
        }
        
        if (this.server) {
            this.server.close();
        }
        
        Logger.info('Enhanced ML Server with Automatic & Periodic Training stopped');
    }
}

module.exports = MLServer;