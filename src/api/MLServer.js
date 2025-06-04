const express = require('express');
const config = require('config');
const fs = require('fs');
const path = require('path');
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
        this.mlStorage = null; // Advanced persistence
        this.featureCounts = {}; // Track feature count per pair
        this.trainingInProgress = new Set(); // Track ongoing training
        
        // 🆕 NEW: Periodic training properties
        this.periodicTrainingConfig = null;
        this.periodicTrainingTimer = null;
        this.periodicTrainingStats = null;
        
        // Model configuration
        this.modelTypes = ['lstm', 'gru', 'cnn', 'transformer'];
        this.enabledModels = config.get('ml.ensemble.enabledModels') || this.modelTypes;
        this.ensembleStrategy = config.get('ml.ensemble.strategy') || 'weighted';
        
        this.initializeServices();
        this.setupRoutes();
        this.setupMiddleware();
    }
    
    initializeServices() {
        Logger.info('Initializing Enhanced ML services with Auto-Training, Weight Persistence, and Periodic Training...');
        
        // Initialize data client
        this.dataClient = new DataClient();
        
        // Initialize feature extractor
        this.featureExtractor = new FeatureExtractor(config.get('ml.features'));
        
        // Initialize data preprocessor
        this.preprocessor = new DataPreprocessor(config.get('ml.models.lstm'));
        
        // Initialize advanced ML storage
        this.mlStorage = new MLStorage({
            baseDir: config.get('ml.storage.baseDir'),
            saveInterval: config.get('ml.storage.saveInterval'),
            maxAgeHours: config.get('ml.storage.maxAgeHours'),
            enableCache: config.get('ml.storage.enableCache')
        });
        
        // 🆕 Initialize periodic training system
        this.periodicTrainingConfig = {
            enabled: config.get('ml.training.periodicTraining') !== false, // Default enabled
            interval: config.get('ml.training.periodicInterval') || 3600000, // 1 hour default
            epochs: config.get('ml.training.periodicEpochs') || 15, // Faster periodic training
            maxConcurrent: config.get('ml.training.maxConcurrent') || 2, // Max 2 pairs training at once
            quietHours: config.get('ml.training.quietHours') || null, // Optional quiet hours
            lastTrainedThreshold: config.get('ml.training.retrainThreshold') || 3600000 // 1 hour min between retrains
        };
        
        this.periodicTrainingTimer = null;
        this.periodicTrainingStats = {
            totalRuns: 0,
            successfulRuns: 0,
            failedRuns: 0,
            lastRun: null,
            nextRun: null,
            trainingHistory: []
        };
        
        Logger.info('Enhanced ML services initialized successfully', {
            enabledModels: this.enabledModels,
            ensembleStrategy: this.ensembleStrategy,
            autoTraining: true,
            weightPersistence: true,
            periodicTraining: this.periodicTrainingConfig.enabled,
            periodicInterval: `${this.periodicTrainingConfig.interval / 60000} minutes`
        });
    }

    // 🆕 NEW: Start periodic training system
    startPeriodicTraining() {
        if (!this.periodicTrainingConfig.enabled) {
            Logger.info('Periodic training is disabled');
            return;
        }
        
        if (this.periodicTrainingTimer) {
            clearInterval(this.periodicTrainingTimer);
        }
        
        Logger.info('🕐 Starting periodic training system', {
            interval: `${this.periodicTrainingConfig.interval / 60000} minutes`,
            epochs: this.periodicTrainingConfig.epochs,
            maxConcurrent: this.periodicTrainingConfig.maxConcurrent
        });
        
        // Set next run time
        this.periodicTrainingStats.nextRun = Date.now() + this.periodicTrainingConfig.interval;
        
        this.periodicTrainingTimer = setInterval(async () => {
            await this.runPeriodicTraining();
        }, this.periodicTrainingConfig.interval);
        
        // Also run a check 5 minutes after startup (gives core service time to stabilize)
        setTimeout(async () => {
            Logger.info('🚀 Running initial periodic training check...');
            await this.runPeriodicTraining();
        }, 300000); // 5 minutes
    }
    
    // 🆕 NEW: Run periodic training cycle
    async runPeriodicTraining() {
        const runId = `periodic_${Date.now()}`;
        
        try {
            Logger.info('🕐 Starting periodic training cycle', { runId });
            
            this.periodicTrainingStats.totalRuns++;
            this.periodicTrainingStats.lastRun = Date.now();
            this.periodicTrainingStats.nextRun = Date.now() + this.periodicTrainingConfig.interval;
            
            // Check quiet hours if configured
            if (this.isQuietHour()) {
                Logger.info('⏰ Skipping periodic training due to quiet hours');
                return;
            }
            
            // Get all pairs that have models
            const eligiblePairs = this.getEligiblePairsForRetraining();
            
            if (eligiblePairs.length === 0) {
                Logger.info('📭 No pairs eligible for periodic retraining');
                return;
            }
            
            Logger.info('🎯 Pairs eligible for periodic retraining', {
                pairs: eligiblePairs,
                maxConcurrent: this.periodicTrainingConfig.maxConcurrent
            });
            
            // Train pairs with concurrency limit
            const results = await this.trainPairsWithConcurrencyLimit(eligiblePairs, runId);
            
            // Update stats
            const successful = results.filter(r => r.status === 'completed').length;
            const failed = results.filter(r => r.status === 'failed').length;
            
            this.periodicTrainingStats.successfulRuns += successful > 0 ? 1 : 0;
            this.periodicTrainingStats.failedRuns += failed > 0 ? 1 : 0;
            
            // Keep recent history (last 24 runs)
            this.periodicTrainingStats.trainingHistory.push({
                runId,
                timestamp: Date.now(),
                pairs: eligiblePairs,
                results: results,
                successful,
                failed,
                duration: Date.now() - this.periodicTrainingStats.lastRun
            });
            
            if (this.periodicTrainingStats.trainingHistory.length > 24) {
                this.periodicTrainingStats.trainingHistory = this.periodicTrainingStats.trainingHistory.slice(-24);
            }
            
            Logger.info('✅ Periodic training cycle completed', {
                runId,
                successful,
                failed,
                duration: `${(Date.now() - this.periodicTrainingStats.lastRun) / 1000}s`,
                nextRun: new Date(this.periodicTrainingStats.nextRun).toLocaleTimeString()
            });
            
        } catch (error) {
            Logger.error('❌ Periodic training cycle failed', {
                runId,
                error: error.message
            });
            this.periodicTrainingStats.failedRuns++;
        }
    }
    
    // 🆕 NEW: Get pairs eligible for retraining
    getEligiblePairsForRetraining() {
        const eligible = [];
        const now = Date.now();
        
        // Check all pairs that have ensembles (indicating they've been used)
        for (const pair of Object.keys(this.ensembles)) {
            // Skip if currently training
            if (this.trainingInProgress.has(pair)) {
                Logger.debug(`Skipping ${pair} - training in progress`);
                continue;
            }
            
            // Check when this pair was last trained
            const trainingHistory = this.mlStorage.loadTrainingHistory(`${pair}_ensemble`);
            
            if (!trainingHistory || !trainingHistory.trainingResults) {
                // Never been trained, but has ensemble - eligible for initial training
                eligible.push(pair);
                continue;
            }
            
            const lastTrained = trainingHistory.timestamp;
            const timeSinceLastTrain = now - lastTrained;
            
            // Check if enough time has passed since last training
            if (timeSinceLastTrain >= this.periodicTrainingConfig.lastTrainedThreshold) {
                eligible.push(pair);
                Logger.debug(`${pair} eligible - last trained ${Math.round(timeSinceLastTrain / 60000)} minutes ago`);
            } else {
                Logger.debug(`${pair} too recently trained - ${Math.round(timeSinceLastTrain / 60000)} minutes ago`);
            }
        }
        
        return eligible;
    }
    
    // 🆕 NEW: Train multiple pairs with concurrency control
    async trainPairsWithConcurrencyLimit(pairs, runId) {
        const results = [];
        const maxConcurrent = this.periodicTrainingConfig.maxConcurrent;
        
        // Process pairs in batches
        for (let i = 0; i < pairs.length; i += maxConcurrent) {
            const batch = pairs.slice(i, i + maxConcurrent);
            
            Logger.info(`🔄 Training batch ${Math.floor(i / maxConcurrent) + 1}`, {
                pairs: batch,
                runId
            });
            
            // Train pairs in this batch concurrently
            const batchPromises = batch.map(async (pair) => {
                const startTime = Date.now();
                
                try {
                    // Use reduced epochs for periodic training
                    const periodicConfig = {
                        epochs: this.periodicTrainingConfig.epochs,
                        batchSize: 32,
                        verbose: 0,
                        periodicTraining: true,
                        runId: runId
                    };
                    
                    await this.trainEnsembleModels(pair, periodicConfig);
                    
                    return {
                        pair,
                        status: 'completed',
                        duration: Date.now() - startTime,
                        runId
                    };
                    
                } catch (error) {
                    Logger.error(`Periodic training failed for ${pair}`, {
                        error: error.message,
                        runId
                    });
                    
                    return {
                        pair,
                        status: 'failed',
                        error: error.message,
                        duration: Date.now() - startTime,
                        runId
                    };
                }
            });
            
            // Wait for this batch to complete
            const batchResults = await Promise.all(batchPromises);
            results.push(...batchResults);
            
            // Brief pause between batches to prevent overwhelming the system
            if (i + maxConcurrent < pairs.length) {
                await new Promise(resolve => setTimeout(resolve, 10000)); // 10 second pause
            }
        }
        
        return results;
    }
    
    // 🆕 NEW: Check if current time is in quiet hours
    isQuietHour() {
        if (!this.periodicTrainingConfig.quietHours) {
            return false;
        }
        
        const now = new Date();
        const currentHour = now.getHours();
        const { start, end } = this.periodicTrainingConfig.quietHours;
        
        // Handle quiet hours that span midnight
        if (start > end) {
            return currentHour >= start || currentHour < end;
        } else {
            return currentHour >= start && currentHour < end;
        }
    }
    
    // 🆕 NEW: Get periodic training statistics
    getPeriodicTrainingStats() {
        return {
            config: this.periodicTrainingConfig,
            stats: this.periodicTrainingStats,
            status: {
                isEnabled: this.periodicTrainingConfig.enabled,
                isRunning: !!this.periodicTrainingTimer,
                nextRun: this.periodicTrainingStats.nextRun ? 
                    new Date(this.periodicTrainingStats.nextRun).toISOString() : null,
                lastRun: this.periodicTrainingStats.lastRun ? 
                    new Date(this.periodicTrainingStats.lastRun).toISOString() : null,
                eligiblePairs: this.getEligiblePairsForRetraining(),
                currentlyTraining: Array.from(this.trainingInProgress),
                isQuietHour: this.isQuietHour()
            }
        };
    }
    
    // 🆕 NEW: Stop periodic training
    stopPeriodicTraining() {
        if (this.periodicTrainingTimer) {
            clearInterval(this.periodicTrainingTimer);
            this.periodicTrainingTimer = null;
            Logger.info('🛑 Periodic training stopped');
        }
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
        // Enhanced health check with ensemble information
        this.app.get('/api/health', async (req, res) => {
            try {
                const coreHealth = await this.dataClient.checkCoreHealth();
                const storageStats = this.mlStorage.getStorageStats();
                
                // Get ensemble statistics
                const ensembleStats = {};
                for (const [pair, ensemble] of Object.entries(this.ensembles)) {
                    ensembleStats[pair] = ensemble.getEnsembleStats();
                }
                
                res.json({
                    status: 'healthy',
                    service: 'trading-bot-ml-enhanced',
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
                        enabledTypes: this.enabledModels,
                        strategy: this.ensembleStrategy,
                        featureCounts: this.featureCounts,
                        trainedWeights: storageStats.trainedModels,
                        trainingInProgress: Array.from(this.trainingInProgress)
                    },
                    predictions: {
                        cached: Object.keys(this.predictions).length,
                        lastUpdate: this.getLastPredictionTime()
                    },
                    storage: {
                        enabled: true,
                        stats: storageStats,
                        cacheSize: storageStats.cache
                    },
                    features: {
                        autoTraining: true,
                        weightPersistence: true,
                        backgroundTraining: true
                    }
                });
            } catch (error) {
                Logger.error('Health check failed', { error: error.message });
                res.status(500).json({
                    status: 'unhealthy',
                    service: 'trading-bot-ml-enhanced',
                    error: error.message,
                    timestamp: Date.now()
                });
            }
        });
        
        // Enhanced prediction endpoint with auto-training
        this.app.get('/api/predictions/:pair', async (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                const useEnsemble = req.query.ensemble !== 'false'; // Default to ensemble
                const strategy = req.query.strategy || this.ensembleStrategy;
                
                let prediction;
                if (useEnsemble) {
                    // 🆕 USE AUTO-TRAINING VERSION
                    prediction = await this.getEnsemblePredictionWithAutoTrain(pair, { strategy });
                } else {
                    // 🆕 USE AUTO-TRAINING VERSION  
                    prediction = await this.getSingleModelPredictionWithAutoTrain(pair, req.query.model || 'lstm');
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
                           (Date.now() - this.predictions[pair].timestamp) < 60000
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
        
        // Get ensemble statistics
        this.app.get('/api/ensemble/:pair/stats', (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                const ensemble = this.ensembles[pair];
                
                if (!ensemble) {
                    res.status(404).json({
                        error: 'Ensemble not found',
                        pair: pair,
                        available: Object.keys(this.ensembles)
                    });
                    return;
                }
                
                const stats = ensemble.getEnsembleStats();
                res.json({
                    pair,
                    ensemble: stats,
                    timestamp: Date.now()
                });
                
            } catch (error) {
                Logger.error(`Ensemble stats failed for ${req.params.pair}`, { 
                    error: error.message 
                });
                res.status(500).json({
                    error: 'Ensemble stats failed',
                    message: error.message,
                    pair: req.params.pair.toUpperCase()
                });
            }
        });
        
        // Update ensemble weights
        this.app.post('/api/ensemble/:pair/weights', async (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                const weights = req.body.weights;
                
                if (!weights || typeof weights !== 'object') {
                    res.status(400).json({
                        error: 'Invalid weights format',
                        expected: 'Object with model names as keys and weights as values'
                    });
                    return;
                }
                
                const ensemble = this.ensembles[pair];
                if (!ensemble) {
                    res.status(404).json({
                        error: 'Ensemble not found',
                        pair: pair
                    });
                    return;
                }
                
                ensemble.updateWeights(weights);
                
                // Save updated ensemble configuration
                await this.mlStorage.saveModelMetadata(`${pair}_ensemble`, {
                    ensembleConfig: ensemble.toJSON(),
                    weightsUpdated: Date.now(),
                    updatedBy: 'api'
                });
                
                res.json({
                    success: true,
                    pair: pair,
                    newWeights: ensemble.weights,
                    timestamp: Date.now()
                });
                
            } catch (error) {
                Logger.error(`Weight update failed for ${req.params.pair}`, { 
                    error: error.message 
                });
                res.status(500).json({
                    error: 'Weight update failed',
                    message: error.message
                });
            }
        });
        
        // Train ensemble models (manual training)
        this.app.post('/api/train/:pair/ensemble', async (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                const trainingConfig = req.body || {};
                
                // Check if training is already in progress
                if (this.trainingInProgress.has(pair)) {
                    res.json({
                        message: `Training already in progress for ${pair}`,
                        pair,
                        status: 'already_training',
                        timestamp: Date.now()
                    });
                    return;
                }
                
                // Start ensemble training in background
                this.trainEnsembleModels(pair, { 
                    ...trainingConfig,
                    epochs: trainingConfig.epochs || 50, // Full training epochs
                    manualTraining: true
                }).catch(error => {
                    Logger.error(`Background ensemble training failed for ${pair}`, { 
                        error: error.message 
                    });
                });
                
                res.json({
                    message: `Manual ensemble training started for ${pair}`,
                    pair,
                    modelTypes: this.enabledModels,
                    config: trainingConfig,
                    trainingType: 'manual',
                    timestamp: Date.now()
                });
                
            } catch (error) {
                Logger.error(`Ensemble training start failed for ${req.params.pair}`, { 
                    error: error.message 
                });
                res.status(500).json({
                    error: 'Ensemble training start failed',
                    message: error.message,
                    pair: req.params.pair.toUpperCase()
                });
            }
        });
        
        // Compare model performance
        this.app.get('/api/models/:pair/compare', async (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                const comparison = await this.compareModelPerformance(pair);
                
                res.json({
                    pair,
                    comparison,
                    timestamp: Date.now()
                });
                
            } catch (error) {
                Logger.error(`Model comparison failed for ${req.params.pair}`, { 
                    error: error.message 
                });
                res.status(500).json({
                    error: 'Model comparison failed',
                    message: error.message,
                    pair: req.params.pair.toUpperCase()
                });
            }
        });
        
        // Get individual model prediction with auto-training
        this.app.get('/api/models/:pair/:modelType/predict', async (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                const modelType = req.params.modelType.toLowerCase();
                
                if (!this.enabledModels.includes(modelType)) {
                    res.status(400).json({
                        error: 'Model type not enabled',
                        modelType: modelType,
                        enabled: this.enabledModels
                    });
                    return;
                }
                
                // 🆕 USE AUTO-TRAINING VERSION
                const prediction = await this.getSingleModelPredictionWithAutoTrain(pair, modelType);
                
                res.json({
                    pair,
                    modelType,
                    prediction,
                    timestamp: Date.now()
                });
                
            } catch (error) {
                Logger.error(`Individual model prediction failed`, { 
                    error: error.message,
                    pair: req.params.pair,
                    modelType: req.params.modelType
                });
                res.status(500).json({
                    error: 'Individual model prediction failed',
                    message: error.message
                });
            }
        });
        
        // Check training status
        this.app.get('/api/training/:pair/status', (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                const isTraining = this.trainingInProgress.has(pair);
                
                const weightsStatus = {};
                for (const modelType of this.enabledModels) {
                    weightsStatus[modelType] = {
                        hasTrainedWeights: this.mlStorage.hasTrainedWeights(pair, modelType)
                    };
                }
                
                res.json({
                    pair,
                    isTraining,
                    allTrainingInProgress: Array.from(this.trainingInProgress),
                    weightsStatus,
                    needsTraining: Object.values(weightsStatus).some(w => !w.hasTrainedWeights),
                    timestamp: Date.now()
                });
                
            } catch (error) {
                Logger.error(`Training status check failed for ${req.params.pair}`, { 
                    error: error.message 
                });
                res.status(500).json({
                    error: 'Training status check failed',
                    message: error.message
                });
            }
        });
        
        // Existing routes (enhanced with ensemble support)
        this.app.get('/api/storage/stats', (req, res) => {
            try {
                const stats = this.mlStorage.getStorageStats();
                res.json({
                    storage: stats,
                    timestamp: Date.now()
                });
            } catch (error) {
                Logger.error('Storage stats failed', { error: error.message });
                res.status(500).json({
                    error: 'Failed to get storage stats',
                    message: error.message
                });
            }
        });
        
        this.app.post('/api/storage/save', async (req, res) => {
            try {
                const savedCount = await this.mlStorage.forceSave();
                res.json({
                    success: true,
                    message: 'ML data saved successfully with atomic writes',
                    savedCount,
                    timestamp: Date.now()
                });
            } catch (error) {
                Logger.error('Force save failed', { error: error.message });
                res.status(500).json({
                    error: 'Force save failed',
                    message: error.message
                });
            }
        });
        
        this.app.post('/api/storage/cleanup', async (req, res) => {
            try {
                const maxAgeHours = req.body.maxAgeHours || 168;
                const cleanedCount = await this.mlStorage.cleanup(maxAgeHours);
                
                res.json({
                    success: true,
                    message: `Cleaned up ${cleanedCount} old ML files`,
                    cleanedCount,
                    maxAgeHours,
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
        
        // Enhanced features endpoint
        this.app.get('/api/features/:pair', async (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                
                // Try to load from cache first
                const cachedFeatures = this.mlStorage.loadFeatureCache(pair);
                if (cachedFeatures && (Date.now() - cachedFeatures.timestamp) < 300000) {
                    res.json({
                        pair,
                        features: cachedFeatures.features,
                        timestamp: cachedFeatures.timestamp,
                        cached: true
                    });
                    return;
                }
                
                // Extract fresh features
                const pairData = await this.dataClient.getPairData(pair);
                const features = this.featureExtractor.extractFeatures(pairData);
                
                // Update feature count tracking
                this.featureCounts[pair] = features.features.length;
                
                // Save to cache
                await this.mlStorage.saveFeatureCache(pair, {
                    count: features.features.length,
                    names: features.featureNames,
                    values: features.features.slice(0, 10),
                    metadata: features.metadata
                });
                
                res.json({
                    pair,
                    features: {
                        count: features.features.length,
                        names: features.featureNames,
                        values: features.features.slice(0, 10),
                        metadata: features.metadata
                    },
                    timestamp: Date.now(),
                    cached: false
                });
                
            } catch (error) {
                Logger.error(`Feature extraction failed for ${req.params.pair}`, { 
                    error: error.message 
                });
                res.status(500).json({
                    error: 'Feature extraction failed',
                    message: error.message,
                    pair: req.params.pair.toUpperCase()
                });
            }
        });
        
        // Enhanced model status with ensemble info
        this.app.get('/api/models/:pair/status', async (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                const pairModels = this.models[pair] || {};
                const ensemble = this.ensembles[pair];
                const isTraining = this.trainingInProgress.has(pair);
                
                // Load persistent model metadata
                const modelMetadata = this.mlStorage.loadModelMetadata(pair);
                const trainingHistory = this.mlStorage.loadTrainingHistory(pair);
                const ensembleMetadata = this.mlStorage.loadModelMetadata(`${pair}_ensemble`);
                
                const individualModels = {};
                for (const modelType of this.enabledModels) {
                    const model = pairModels[modelType];
                    const hasWeights = this.mlStorage.hasTrainedWeights(pair, modelType);
                    
                    individualModels[modelType] = {
                        hasModel: !!model,
                        hasTrainedWeights: hasWeights,
                        modelInfo: model ? model.getModelSummary() : null,
                        weightsStatus: hasWeights ? 'saved' : 'none',
                        needsTraining: !hasWeights
                    };
                }
                
                res.json({
                    pair,
                    featureCount: this.featureCounts[pair] || 'unknown',
                    isTraining: isTraining,
                    individual: individualModels,
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
                        lastTrained: trainingHistory ? trainingHistory.timestamp : null,
                        trainedModels: this.mlStorage.getTrainedModelsList().filter(m => m.pair === pair)
                    },
                    autoTraining: {
                        enabled: true,
                        needsTraining: Object.values(individualModels).some(m => m.needsTraining),
                        trainingInProgress: isTraining
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
        
        // Enhanced prediction history
        this.app.get('/api/predictions/:pair/history', (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                const history = this.mlStorage.loadPredictionHistory(pair);
                
                if (!history) {
                    res.json({
                        pair,
                        predictions: [],
                        count: 0,
                        message: 'No prediction history found',
                        timestamp: Date.now()
                    });
                    return;
                }
                
                // Apply optional filters
                let predictions = history.predictions || [];
                const limit = parseInt(req.query.limit) || 100;
                const since = req.query.since ? parseInt(req.query.since) : null;
                const ensemble = req.query.ensemble; // 'true', 'false', or undefined
                
                if (since) {
                    predictions = predictions.filter(p => p.timestamp >= since);
                }
                
                if (ensemble !== undefined) {
                    const useEnsemble = ensemble === 'true';
                    predictions = predictions.filter(p => p.useEnsemble === useEnsemble);
                }
                
                predictions = predictions.slice(-limit);
                
                res.json({
                    pair,
                    predictions,
                    count: predictions.length,
                    totalCount: history.count,
                    filters: {
                        limit,
                        since,
                        ensemble: ensemble !== undefined ? ensemble === 'true' : null
                    },
                    timestamp: Date.now()
                });
                
            } catch (error) {
                Logger.error(`Prediction history failed for ${req.params.pair}`, { 
                    error: error.message 
                });
                res.status(500).json({
                    error: 'Prediction history failed',
                    message: error.message,
                    pair: req.params.pair.toUpperCase()
                });
            }
        });
        
        // Rebuild models endpoint
        this.app.post('/api/models/:pair/rebuild', async (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                
                Logger.info(`Rebuilding models for ${pair} with current feature count`);
                
                // Clear existing models for this pair
                if (this.models[pair]) {
                    Object.values(this.models[pair]).forEach(model => {
                        if (model && typeof model.dispose === 'function') {
                            model.dispose();
                        }
                    });
                    delete this.models[pair];
                }
                
                // Clear ensemble
                if (this.ensembles[pair]) {
                    if (typeof this.ensembles[pair].dispose === 'function') {
                        this.ensembles[pair].dispose();
                    }
                    delete this.ensembles[pair];
                }
                
                // Clear cache for this pair
                delete this.predictions[pair];
                delete this.featureCounts[pair];
                
                // Remove from training progress if present
                this.trainingInProgress.delete(pair);
                
                // Get current feature count
                const pairData = await this.dataClient.getPairData(pair);
                const features = this.featureExtractor.extractFeatures(pairData);
                const currentFeatureCount = features.features.length;
                
                this.featureCounts[pair] = currentFeatureCount;
                
                Logger.info(`Rebuilding models for ${pair} with ${currentFeatureCount} features`);
                
                res.json({
                    success: true,
                    message: `Models rebuilt for ${pair}`,
                    pair: pair,
                    newFeatureCount: currentFeatureCount,
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
        
        // Enhanced API endpoints list
        this.app.get('/api', (req, res) => {
            res.json({
                service: 'trading-bot-ml-enhanced',
                version: '2.1.0',
                features: [
                    'Model Ensemble System',
                    'Multiple Model Types (LSTM, GRU, CNN, Transformer)',
                    '🆕 Automatic Training on First Use',
                    '🆕 Background Training',
                    'Weight Persistence & Loading',
                    'Weighted Voting Strategies',
                    'Performance Tracking',
                    'Advanced Persistence',
                    'Dynamic Feature Count Handling'
                ],
                endpoints: [
                    'GET /api/health - Enhanced service health with auto-training status',
                    'GET /api/predictions/:pair - Auto-training ensemble predictions',
                    'GET /api/predictions/:pair/history - Enhanced prediction history with filters',
                    'GET /api/ensemble/:pair/stats - Ensemble statistics and performance',
                    'POST /api/ensemble/:pair/weights - Update ensemble weights',
                    'POST /api/train/:pair/ensemble - Manual training (full epochs)',
                    'GET /api/training/:pair/status - Check training status',
                    'GET /api/models/:pair/compare - Compare individual model performance',
                    'GET /api/models/:pair/:modelType/predict - Auto-training individual predictions',
                    'GET /api/models/:pair/status - Enhanced model status with training info',
                    'POST /api/models/:pair/rebuild - Rebuild models with current feature count',
                    'GET /api/features/:pair - Feature extraction with caching',
                    'GET /api/storage/stats - Storage statistics including trained weights',
                    'POST /api/storage/save - Force save all data',
                    'POST /api/storage/cleanup - Clean up old files'
                ],
                modelTypes: this.enabledModels,
                ensembleStrategies: ['weighted', 'majority', 'average', 'confidence_weighted'],
                currentStrategy: this.ensembleStrategy,
                training: {
                    autoTraining: true,
                    backgroundTraining: true,
                    weightPersistence: true,
                    manualTraining: true
                },
                storage: {
                    enabled: true,
                    features: [
                        'Atomic file writes',
                        'Ensemble configuration persistence',
                        'Individual model metadata',
                        'Model weight persistence & loading',
                        'Performance history tracking',
                        'Prediction history with ensemble info',
                        'Training progress tracking'
                    ]
                },
                timestamp: Date.now()
            });
        });

        this.app.get('/api/training/periodic/status', (req, res) => {
            try {
                const stats = this.getPeriodicTrainingStats();
                res.json({
                    periodicTraining: stats,
                    timestamp: Date.now()
                });
            } catch (error) {
                Logger.error('Failed to get periodic training status', { error: error.message });
                res.status(500).json({
                    error: 'Failed to get periodic training status',
                    message: error.message
                });
            }
        });
        
        // Start/stop periodic training
        this.app.post('/api/training/periodic/control', (req, res) => {
            try {
                const { action } = req.body;
                
                if (action === 'start') {
                    this.startPeriodicTraining();
                    res.json({
                        success: true,
                        message: 'Periodic training started',
                        status: 'started',
                        nextRun: new Date(this.periodicTrainingStats.nextRun).toISOString(),
                        timestamp: Date.now()
                    });
                } else if (action === 'stop') {
                    this.stopPeriodicTraining();
                    res.json({
                        success: true,
                        message: 'Periodic training stopped',
                        status: 'stopped',
                        timestamp: Date.now()
                    });
                } else if (action === 'restart') {
                    this.stopPeriodicTraining();
                    setTimeout(() => {
                        this.startPeriodicTraining();
                    }, 1000);
                    res.json({
                        success: true,
                        message: 'Periodic training restarted',
                        status: 'restarted',
                        timestamp: Date.now()
                    });
                } else {
                    res.status(400).json({
                        error: 'Invalid action',
                        validActions: ['start', 'stop', 'restart']
                    });
                }
            } catch (error) {
                Logger.error('Failed to control periodic training', { error: error.message });
                res.status(500).json({
                    error: 'Failed to control periodic training',
                    message: error.message
                });
            }
        });
        
        // Force run periodic training immediately
        this.app.post('/api/training/periodic/run', async (req, res) => {
            try {
                if (this.trainingInProgress.size >= this.periodicTrainingConfig.maxConcurrent) {
                    res.status(429).json({
                        error: 'Too many training sessions in progress',
                        currentlyTraining: Array.from(this.trainingInProgress),
                        maxConcurrent: this.periodicTrainingConfig.maxConcurrent
                    });
                    return;
                }
                
                // Start periodic training run in background
                this.runPeriodicTraining().catch(error => {
                    Logger.error('Manual periodic training run failed', { error: error.message });
                });
                
                res.json({
                    success: true,
                    message: 'Periodic training run started manually',
                    eligiblePairs: this.getEligiblePairsForRetraining(),
                    timestamp: Date.now()
                });
                
            } catch (error) {
                Logger.error('Failed to run periodic training', { error: error.message });
                res.status(500).json({
                    error: 'Failed to run periodic training',
                    message: error.message
                });
            }
        });
        
        // Update periodic training configuration
        this.app.post('/api/training/periodic/config', (req, res) => {
            try {
                const { interval, epochs, maxConcurrent, quietHours, enabled } = req.body;
                
                const oldConfig = { ...this.periodicTrainingConfig };
                
                // Update configuration
                if (interval !== undefined) {
                    this.periodicTrainingConfig.interval = Math.max(300000, interval); // Min 5 minutes
                }
                if (epochs !== undefined) {
                    this.periodicTrainingConfig.epochs = Math.max(5, Math.min(100, epochs)); // 5-100 range
                }
                if (maxConcurrent !== undefined) {
                    this.periodicTrainingConfig.maxConcurrent = Math.max(1, Math.min(5, maxConcurrent)); // 1-5 range
                }
                if (quietHours !== undefined) {
                    this.periodicTrainingConfig.quietHours = quietHours;
                }
                if (enabled !== undefined) {
                    this.periodicTrainingConfig.enabled = enabled;
                }
                
                // Restart periodic training if interval changed
                if (oldConfig.interval !== this.periodicTrainingConfig.interval || 
                    oldConfig.enabled !== this.periodicTrainingConfig.enabled) {
                    this.stopPeriodicTraining();
                    if (this.periodicTrainingConfig.enabled) {
                        setTimeout(() => {
                            this.startPeriodicTraining();
                        }, 1000);
                    }
                }
                
                Logger.info('Periodic training configuration updated', {
                    oldConfig,
                    newConfig: this.periodicTrainingConfig
                });
                
                res.json({
                    success: true,
                    message: 'Periodic training configuration updated',
                    oldConfig,
                    newConfig: this.periodicTrainingConfig,
                    timestamp: Date.now()
                });
                
            } catch (error) {
                Logger.error('Failed to update periodic training config', { error: error.message });
                res.status(500).json({
                    error: 'Failed to update periodic training config',
                    message: error.message
                });
            }
        });

    }
    
    // 🆕 Check if models need training and trigger if necessary
    async checkAndAutoTrain(pair) {
        try {
            // Check if training is already in progress
            if (this.trainingInProgress.has(pair)) {
                return {
                    autoTrainingStarted: false,
                    message: 'Training already in progress',
                    trainingInProgress: true
                };
            }
            
            // Check if any models need training
            const modelsNeedingTraining = [];
            
            for (const modelType of this.enabledModels) {
                const hasWeights = this.mlStorage.hasTrainedWeights(pair, modelType);
                if (!hasWeights) {
                    modelsNeedingTraining.push(modelType);
                }
            }
            
            if (modelsNeedingTraining.length > 0) {
                Logger.info(`🤖 Auto-training needed for ${pair}`, {
                    modelsNeedingTraining,
                    totalModels: this.enabledModels.length
                });
                
                // Start training in background with reduced epochs for faster initial training
                const autoTrainingConfig = {
                    epochs: 25,  // Reduced for faster auto-training
                    batchSize: 32,
                    verbose: 0,
                    autoTraining: true
                };
                
                // Start training but don't wait for it
                this.trainEnsembleModels(pair, autoTrainingConfig)
                    .then(() => {
                        Logger.info(`✅ Auto-training completed for ${pair}`);
                    })
                    .catch(error => {
                        Logger.error(`❌ Auto-training failed for ${pair}`, { error: error.message });
                    });
                
                return {
                    autoTrainingStarted: true,
                    modelsNeedingTraining,
                    message: 'Auto-training started in background',
                    trainingType: 'auto',
                    epochs: autoTrainingConfig.epochs
                };
            }
            
            return {
                autoTrainingStarted: false,
                message: 'All models have trained weights',
                trainedModels: this.enabledModels.length
            };
            
        } catch (error) {
            Logger.error(`Auto-training check failed for ${pair}`, { error: error.message });
            return {
                autoTrainingStarted: false,
                error: error.message
            };
        }
    }
    
    // Enhanced ensemble prediction with auto-training and error recovery
    async getEnsemblePredictionWithAutoTrain(pair, options = {}) {
        const cacheKey = `${pair}_ensemble`;
        
        // Check cache first
        if (this.predictions[cacheKey] && 
            (Date.now() - this.predictions[cacheKey].timestamp) < 60000) {
            return this.predictions[cacheKey];
        }
        
        try {
            // CHECK AND TRIGGER AUTO-TRAINING IF NEEDED
            const autoTrainStatus = await this.checkAndAutoTrain(pair);
            
            // Get or create ensemble
            let ensemble = this.ensembles[pair];
            if (!ensemble) {
                ensemble = await this.getOrCreateEnsemble(pair);
            }
            
            // 🆕 ADDED: Check for disposed models and recreate if necessary
            await this.validateAndRepairEnsemble(pair, ensemble);
            
            // Get data and extract features
            const pairData = await this.dataClient.getPairData(pair);
            const features = this.featureExtractor.extractFeatures(pairData);
            
            // Update feature count
            this.featureCounts[pair] = features.features.length;
            
            // Prepare input for prediction
            const inputData = await this.prepareRealTimeInput(features.features);
            
            // Make ensemble prediction
            const prediction = await ensemble.predict(inputData, options);
            
            // ADD AUTO-TRAINING STATUS TO RESPONSE
            prediction.autoTraining = autoTrainStatus;
            
            // Cache result
            this.predictions[cacheKey] = {
                ...prediction,
                timestamp: Date.now(),
                type: 'ensemble',
                autoTraining: autoTrainStatus
            };
            
            // Clean up input tensor
            inputData.dispose();
            
            return this.predictions[cacheKey];
            
        } catch (error) {
            Logger.error(`Ensemble prediction with auto-train failed for ${pair}`, { error: error.message });
            throw error;
        }
    }

    // 🆕 NEW: Validate ensemble models and repair if disposed
    async validateAndRepairEnsemble(pair, ensemble) {
        const currentFeatureCount = this.featureCounts[pair];
        
        for (const modelType of this.enabledModels) {
            try {
                // Check if model exists in our collection
                const model = this.models[pair] && this.models[pair][modelType];
                
                if (!model) {
                    Logger.info(`Model ${modelType} missing for ${pair}, recreating...`);
                    const newModel = await this.getOrCreateModel(pair, modelType, currentFeatureCount);
                    
                    // Update ensemble with new model
                    if (ensemble.models.has(modelType)) {
                        ensemble.removeModel(modelType);
                    }
                    ensemble.addModel(modelType, newModel, 1.0, {
                        pair: pair,
                        featureCount: currentFeatureCount,
                        recreated: Date.now()
                    });
                    
                    Logger.info(`✅ Recreated and re-added ${modelType} to ensemble`);
                }
                
            } catch (error) {
                Logger.warn(`Failed to validate/repair ${modelType} for ${pair}`, { 
                    error: error.message 
                });
                
                // Remove problematic model from ensemble
                if (ensemble.models.has(modelType)) {
                    ensemble.removeModel(modelType);
                    Logger.info(`Removed problematic ${modelType} from ensemble`);
                }
            }
        }
    }
    
    // 🆕 Enhanced single model prediction with auto-training  
    async getSingleModelPredictionWithAutoTrain(pair, modelType) {
        const cacheKey = `${pair}_${modelType}`;
        
        // Check cache first
        if (this.predictions[cacheKey] && 
            (Date.now() - this.predictions[cacheKey].timestamp) < 60000) {
            return this.predictions[cacheKey];
        }
        
        try {
            // 🆕 CHECK IF THIS SPECIFIC MODEL NEEDS TRAINING
            const hasWeights = this.mlStorage.hasTrainedWeights(pair, modelType);
            let autoTrainStatus = { autoTrainingStarted: false };
            
            if (!hasWeights && !this.trainingInProgress.has(pair)) {
                Logger.info(`🤖 Auto-training needed for ${pair}:${modelType}`);
                
                // Start training for this specific model type
                const autoTrainingConfig = {
                    epochs: 25,
                    batchSize: 32,
                    verbose: 0,
                    autoTraining: true,
                    [modelType]: { epochs: 25 } // Specific config for this model
                };
                
                // Start training but don't wait
                this.trainEnsembleModels(pair, autoTrainingConfig)
                    .then(() => {
                        Logger.info(`✅ Auto-training completed for ${pair}:${modelType}`);
                    })
                    .catch(error => {
                        Logger.error(`❌ Auto-training failed for ${pair}:${modelType}`, { error: error.message });
                    });
                
                autoTrainStatus = {
                    autoTrainingStarted: true,
                    modelType: modelType,
                    message: `Auto-training started for ${modelType} model`,
                    trainingType: 'auto',
                    epochs: 25
                };
            } else if (this.trainingInProgress.has(pair)) {
                autoTrainStatus = {
                    autoTrainingStarted: false,
                    message: 'Training already in progress',
                    trainingInProgress: true
                };
            }
            
            // Get data and extract features
            const pairData = await this.dataClient.getPairData(pair);
            const features = this.featureExtractor.extractFeatures(pairData);
            const currentFeatureCount = features.features.length;
            
            // Update feature count tracking
            this.featureCounts[pair] = currentFeatureCount;
            
            // Get or create model with correct feature count
            const model = await this.getOrCreateModel(pair, modelType, currentFeatureCount);
            
            // Prepare input for prediction
            const inputData = await this.prepareRealTimeInput(features.features);
            
            // Make prediction
            const predictions = await model.predict(inputData);
            const prediction = predictions[0];
            
            const result = {
                prediction: prediction,
                confidence: Math.abs(prediction - 0.5) * 2,
                direction: prediction > 0.5 ? 'up' : 'down',
                signal: this.getTradeSignal(prediction, Math.abs(prediction - 0.5) * 2),
                modelType: modelType,
                hasTrainedWeights: hasWeights,
                autoTraining: autoTrainStatus,
                individual: {
                    prediction: prediction,
                    confidence: Math.abs(prediction - 0.5) * 2
                },
                metadata: {
                    timestamp: Date.now(),
                    version: '2.1.0',
                    type: 'individual_prediction',
                    featureCount: currentFeatureCount
                }
            };
            
            // Cache result
            this.predictions[cacheKey] = {
                ...result,
                timestamp: Date.now(),
                type: 'individual'
            };
            
            // Clean up input tensor
            inputData.dispose();
            
            return this.predictions[cacheKey];
            
        } catch (error) {
            Logger.error(`Individual prediction with auto-train failed for ${pair}:${modelType}`, { error: error.message });
            throw error;
        }
    }
    
    // Create or get ensemble for a pair
    async getOrCreateEnsemble(pair) {
        Logger.info(`Creating ensemble for ${pair}`);
        
        // Get current feature count
        const pairData = await this.dataClient.getPairData(pair);
        const features = this.featureExtractor.extractFeatures(pairData);
        const featureCount = features.features.length;
        
        // Update tracking
        this.featureCounts[pair] = featureCount;
        
        // Check if we have persistent ensemble metadata
        const ensembleMetadata = this.mlStorage.loadModelMetadata(`${pair}_ensemble`);
        
        const ensembleConfig = {
            modelTypes: this.enabledModels,
            votingStrategy: this.ensembleStrategy,
            weights: {}
        };
        
        // Restore weights from metadata if available
        if (ensembleMetadata && ensembleMetadata.ensembleConfig) {
            ensembleConfig.weights = ensembleMetadata.ensembleConfig.weights || {};
            ensembleConfig.votingStrategy = ensembleMetadata.ensembleConfig.votingStrategy || this.ensembleStrategy;
        }
        
        const ensemble = new ModelEnsemble(ensembleConfig);
        
        // Create and add individual models to ensemble
        for (const modelType of this.enabledModels) {
            try {
                const model = await this.getOrCreateModel(pair, modelType, featureCount);
                const weight = ensembleConfig.weights[modelType] || 1.0;
                
                ensemble.addModel(modelType, model, weight, {
                    pair: pair,
                    featureCount: featureCount,
                    created: Date.now(),
                    hasTrainedWeights: this.mlStorage.hasTrainedWeights(pair, modelType)
                });
                
                Logger.info(`Added ${modelType} model to ${pair} ensemble`, { 
                    weight, 
                    featureCount,
                    hasTrainedWeights: this.mlStorage.hasTrainedWeights(pair, modelType)
                });
                
            } catch (error) {
                Logger.error(`Failed to add ${modelType} to ensemble for ${pair}`, { 
                    error: error.message 
                });
            }
        }
        
        this.ensembles[pair] = ensemble;
        
        // Save ensemble metadata
        await this.mlStorage.saveModelMetadata(`${pair}_ensemble`, {
            ensembleConfig: ensemble.toJSON(),
            created: Date.now(),
            pair: pair,
            modelTypes: this.enabledModels,
            featureCount: featureCount
        });
        
        return ensemble;
    }
    
    // Create individual model - ENHANCED: With weight loading
    async getOrCreateModel(pair, modelType, featureCount) {
        if (!this.models[pair]) {
            this.models[pair] = {};
        }
        
        // Check if model exists and has correct feature count
        if (this.models[pair][modelType]) {
            const existingModel = this.models[pair][modelType];
            const modelSummary = existingModel.getModelSummary();
            
            if (modelSummary.config && modelSummary.config.features === featureCount) {
                return existingModel; // Model is correct, return it
            } else {
                // Feature count mismatch, dispose and recreate
                Logger.warn(`Feature count mismatch for ${pair}:${modelType}. Expected ${modelSummary.config.features}, got ${featureCount}. Recreating model.`);
                existingModel.dispose();
                delete this.models[pair][modelType];
            }
        }
        
        Logger.info(`Creating ${modelType} model for ${pair}`, { featureCount });
        
        // Create base config with the ACTUAL feature count
        const baseConfig = {
            sequenceLength: this.preprocessor.sequenceLength,
            features: featureCount  // Use the passed featureCount parameter
        };
        
        // Get model-specific config and OVERRIDE features with actual count
        const modelSpecificConfig = config.get(`ml.models.${modelType}`) || {};
        const finalConfig = {
            ...modelSpecificConfig,  // Start with config defaults
            ...baseConfig           // Override with actual values (especially features)
        };
        
        let model;
        let ModelClass;
        
        // Determine the model class
        switch (modelType) {
            case 'lstm':
                ModelClass = LSTMModel;
                break;
            case 'gru':
                ModelClass = GRUModel;
                break;
            case 'cnn':
                ModelClass = CNNModel;
                break;
            case 'transformer':
                ModelClass = TransformerModel;
                break;
            default:
                throw new Error(`Unknown model type: ${modelType}`);
        }
        
        // 🆕 TRY TO LOAD TRAINED WEIGHTS FIRST
        try {
            Logger.info(`Attempting to load trained weights for ${pair}:${modelType}`);
            model = await this.mlStorage.loadModelWeights(pair, modelType, ModelClass, finalConfig);
            
            if (model) {
                Logger.info(`✅ Successfully loaded trained weights for ${pair}:${modelType}`, {
                    featureCount: finalConfig.features,
                    totalParams: model.model.countParams()
                });
                
                this.models[pair][modelType] = model;
                
                // Save model metadata (but weights are already saved)
                await this.mlStorage.saveModelMetadata(`${pair}_${modelType}`, {
                    config: finalConfig,
                    modelType: modelType,
                    loaded: Date.now(),
                    featureCount,
                    status: 'loaded_with_weights',
                    source: 'persistent_storage'
                });
                
                return model;
            }
        } catch (error) {
            Logger.warn(`Failed to load trained weights for ${pair}:${modelType}`, { 
                error: error.message 
            });
        }
        
        // 🆕 CREATE NEW MODEL IF NO WEIGHTS FOUND
        Logger.info(`No trained weights found. Creating new ${modelType} model for ${pair}`);
        
        model = new ModelClass(finalConfig);
        model.buildModel();
        model.compileModel();
        
        this.models[pair][modelType] = model;
        
        // Save model metadata with status indicating it needs training
        await this.mlStorage.saveModelMetadata(`${pair}_${modelType}`, {
            config: finalConfig,
            modelType: modelType,
            created: Date.now(),
            featureCount,
            status: 'created_needs_training',
            hasSavedWeights: false
        });
        
        Logger.info(`✅ New ${modelType} model created for ${pair} (needs training)`, {
            featureCount: finalConfig.features,
            totalParams: model.model.countParams()
        });
        
        return model;
    }
    
    // Prepare real-time input from features
    async prepareRealTimeInput(features) {
        const tf = require('@tensorflow/tfjs');
        const sequenceLength = this.preprocessor.sequenceLength;
        
        // Create a mock sequence by repeating the current features
        // In production, you'd want to use actual historical feature sequences
        const sequence = Array(sequenceLength).fill(features);
        
        // Convert to tensor
        const inputTensor = tf.tensor3d([sequence]); // Shape: [1, sequenceLength, features]
        
        return inputTensor;
    }
    
    // Compare model performance
    async compareModelPerformance(pair) {
        const comparison = {
            pair: pair,
            models: {},
            ensemble: null,
            featureCount: this.featureCounts[pair] || 'unknown',
            weightsStatus: {},
            trainingStatus: {
                isTraining: this.trainingInProgress.has(pair),
                needsTraining: false
            },
            timestamp: Date.now()
        };
        
        // Get sample data for comparison
        try {
            const pairData = await this.dataClient.getPairData(pair);
            const features = this.featureExtractor.extractFeatures(pairData);
            const inputData = await this.prepareRealTimeInput(features.features);
            
            // Update feature count
            this.featureCounts[pair] = features.features.length;
            comparison.featureCount = features.features.length;
            
            // Check weights status for all models
            for (const modelType of this.enabledModels) {
                const hasWeights = this.mlStorage.hasTrainedWeights(pair, modelType);
                comparison.weightsStatus[modelType] = {
                    hasTrainedWeights: hasWeights,
                    weightsLocation: hasWeights ? 
                        path.join(this.mlStorage.weightsDir, `${pair.toLowerCase()}_${modelType}`) : null
                };
                
                if (!hasWeights) {
                    comparison.trainingStatus.needsTraining = true;
                }
            }
            
            // Compare individual models
            const pairModels = this.models[pair] || {};
            for (const modelType of this.enabledModels) {
                if (pairModels[modelType]) {
                    try {
                        const startTime = Date.now();
                        const predictions = await pairModels[modelType].predict(inputData);
                        const predictionTime = Date.now() - startTime;
                        
                        const prediction = predictions[0];
                        const confidence = Math.abs(prediction - 0.5) * 2;
                        
                        comparison.models[modelType] = {
                            prediction: prediction,
                            confidence: confidence,
                            direction: prediction > 0.5 ? 'up' : 'down',
                            predictionTime: predictionTime,
                            modelSummary: pairModels[modelType].getModelSummary(),
                            available: true,
                            hasTrainedWeights: comparison.weightsStatus[modelType].hasTrainedWeights
                        };
                        
                    } catch (error) {
                        comparison.models[modelType] = {
                            error: error.message,
                            available: false,
                            hasTrainedWeights: comparison.weightsStatus[modelType].hasTrainedWeights
                        };
                    }
                } else {
                    comparison.models[modelType] = {
                        available: false,
                        message: 'Model not created',
                        hasTrainedWeights: comparison.weightsStatus[modelType].hasTrainedWeights
                    };
                }
            }
            
            // Compare ensemble if available
            const ensemble = this.ensembles[pair];
            if (ensemble) {
                try {
                    const startTime = Date.now();
                    const ensemblePrediction = await ensemble.predict(inputData);
                    const predictionTime = Date.now() - startTime;
                    
                    comparison.ensemble = {
                        ...ensemblePrediction,
                        predictionTime: predictionTime,
                        stats: ensemble.getEnsembleStats(),
                        available: true
                    };
                    
                } catch (error) {
                    comparison.ensemble = {
                        error: error.message,
                        available: false
                    };
                }
            } else {
                comparison.ensemble = {
                    available: false,
                    message: 'Ensemble not created'
                };
            }
            
            // Clean up tensor
            inputData.dispose();
            
        } catch (error) {
            Logger.error(`Model comparison failed for ${pair}`, { error: error.message });
            comparison.error = error.message;
        }
        
        return comparison;
    }
    
    // Train ensemble models - ENHANCED: With weight saving and training tracking
    async trainEnsembleModels(pair, config = {}) {
        const trainingKey = pair.toUpperCase();
        
        // Add to training progress tracking
        this.trainingInProgress.add(trainingKey);
        
        Logger.info(`Starting ensemble training for ${pair}`, {
            config,
            trainingType: config.autoTraining ? 'auto' : 'manual',
            trainingInProgress: Array.from(this.trainingInProgress)
        });
        
        try {
            // Get historical data
            const pairData = await this.dataClient.getPairData(pair);
            const features = this.featureExtractor.extractFeatures(pairData);
            const currentFeatureCount = features.features.length;
            
            // Update feature count
            this.featureCounts[pair] = currentFeatureCount;
            
            // Create targets for training
            const targets = this.featureExtractor.createTargets(pairData.history);
            const binaryTargets = targets[`direction_${config.targetPeriods || 5}`] || targets['direction_5'];
            
            if (!binaryTargets || binaryTargets.length === 0) {
                throw new Error('No training targets available');
            }
            
            // Prepare training data
            const featuresArray = Array(binaryTargets.length).fill().map(() => features.features);
            const processedData = await this.preprocessor.prepareTrainingData(featuresArray, binaryTargets);
            
            const trainingResults = {
                pair: pair,
                startTime: Date.now(),
                models: {},
                ensemble: null,
                config: config,
                featureCount: currentFeatureCount,
                trainingType: config.autoTraining ? 'auto' : 'manual'
            };
            
            // 🆕 CLEAN UP OLD WEIGHTS IF FEATURE COUNT CHANGED
            for (const modelType of this.enabledModels) {
                if (this.mlStorage.hasTrainedWeights(pair, modelType)) {
                    // Check if existing weights are compatible
                    const weightsDir = path.join(this.mlStorage.weightsDir, `${pair.toLowerCase()}_${modelType}`);
                    const metadataPath = path.join(weightsDir, 'metadata.json');
                    
                    if (fs.existsSync(metadataPath)) {
                        try {
                            const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
                            const savedFeatureCount = metadata.modelSummary?.config?.features;
                            
                            if (savedFeatureCount && savedFeatureCount !== currentFeatureCount) {
                                Logger.warn(`Feature count changed for ${pair}:${modelType}. Old: ${savedFeatureCount}, New: ${currentFeatureCount}. Cleaning up old weights.`);
                                
                                // Remove old weights directory
                                const rimraf = require('rimraf');
                                rimraf.sync(weightsDir);
                                
                                // Clear from cache
                                const cacheKey = `${pair.toUpperCase()}_${modelType}`;
                                this.mlStorage.weightsCache.delete(cacheKey);
                                
                                Logger.info(`Cleaned up incompatible weights for ${pair}:${modelType}`);
                            }
                        } catch (error) {
                            Logger.warn(`Failed to check weights compatibility for ${pair}:${modelType}`, { error: error.message });
                        }
                    }
                }
            }
            
            // Train individual models
            for (const modelType of this.enabledModels) {
                try {
                    Logger.info(`Training ${modelType} model for ${pair}`, {
                        trainingType: config.autoTraining ? 'auto' : 'manual'
                    });
                    
                    const model = await this.getOrCreateModel(pair, modelType, currentFeatureCount);
                    
                    const modelTrainingConfig = {
                        epochs: config.epochs || (config.autoTraining ? 25 : 50),
                        batchSize: config.batchSize || 32,
                        verbose: 0,
                        ...config[modelType] // Model-specific config
                    };
                    
                    const history = await model.train(
                        processedData.trainX,
                        processedData.trainY,
                        processedData.validationX,
                        processedData.validationY,
                        modelTrainingConfig
                    );
                    
                    // 🆕 SAVE TRAINED WEIGHTS AFTER SUCCESSFUL TRAINING
                    try {
                        Logger.info(`Saving trained weights for ${pair}:${modelType}`);
                        await this.mlStorage.saveModelWeights(pair, modelType, model);
                        
                        trainingResults.models[modelType] = {
                            status: 'completed',
                            finalMetrics: history.finalMetrics,
                            epochsCompleted: history.epoch ? history.epoch.length : (history.finalMetrics?.epochsCompleted || 0),
                            modelType: modelType,
                            weightsSaved: true,
                            weightsTimestamp: Date.now(),
                            trainingType: config.autoTraining ? 'auto' : 'manual'
                        };
                        
                        Logger.info(`✅ ${modelType} training and weight saving completed for ${pair}`, 
                            trainingResults.models[modelType].finalMetrics);
                        
                    } catch (weightSaveError) {
                        Logger.error(`Failed to save weights for ${pair}:${modelType}`, { 
                            error: weightSaveError.message 
                        });
                        
                        trainingResults.models[modelType] = {
                            status: 'completed_weights_failed',
                            finalMetrics: history.finalMetrics,
                            epochsCompleted: history.epoch ? history.epoch.length : (history.finalMetrics?.epochsCompleted || 0),
                            modelType: modelType,
                            weightsSaved: false,
                            weightError: weightSaveError.message,
                            trainingType: config.autoTraining ? 'auto' : 'manual'
                        };
                    }
                    
                    // Save individual model training history
                    await this.mlStorage.saveTrainingHistory(`${pair}_${modelType}`, {
                        ...trainingResults.models[modelType],
                        pair: pair,
                        modelType: modelType,
                        timestamp: Date.now()
                    });
                    
                } catch (error) {
                    Logger.error(`${modelType} training failed for ${pair}`, { error: error.message });
                    trainingResults.models[modelType] = {
                        status: 'failed',
                        error: error.message,
                        modelType: modelType,
                        weightsSaved: false,
                        trainingType: config.autoTraining ? 'auto' : 'manual'
                    };
                }
            }
            
            // Update ensemble weights based on training performance
            const ensemble = this.ensembles[pair];
            if (ensemble) {
                const performanceWeights = {};
                
                for (const [modelType, result] of Object.entries(trainingResults.models)) {
                    if (result.status === 'completed' || result.status === 'completed_weights_failed') {
                        // Use accuracy as performance metric
                        const accuracy = parseFloat(result.finalMetrics.finalAccuracy);
                        performanceWeights[modelType] = isNaN(accuracy) ? 0.1 : Math.max(0.1, accuracy);
                    } else {
                        performanceWeights[modelType] = 0.1; // Minimum weight for failed models
                    }
                }
                
                ensemble.updateWeights(performanceWeights);
                
                trainingResults.ensemble = {
                    status: 'updated',
                    newWeights: ensemble.weights,
                    strategy: ensemble.votingStrategy,
                    modelCount: ensemble.models.size
                };
                
                // Save updated ensemble configuration
                await this.mlStorage.saveModelMetadata(`${pair}_ensemble`, {
                    ensembleConfig: ensemble.toJSON(),
                    trainingCompleted: Date.now(),
                    performanceWeights: performanceWeights,
                    featureCount: currentFeatureCount,
                    weightsStatus: trainingResults.models,
                    trainingType: config.autoTraining ? 'auto' : 'manual'
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
            
            // Save overall ensemble training history
            await this.mlStorage.saveTrainingHistory(`${pair}_ensemble`, trainingResults);
            
            // 🆕 LOG TRAINING SUMMARY WITH WEIGHTS STATUS
            const completedModels = Object.values(trainingResults.models).filter(m => 
                m.status === 'completed' || m.status === 'completed_weights_failed'
            ).length;
            const savedWeights = Object.values(trainingResults.models).filter(m => m.weightsSaved).length;
            
            Logger.info(`✅ Ensemble training completed for ${pair}`, {
                totalTime: trainingResults.totalTime,
                modelsCompleted: completedModels,
                modelsFailed: Object.values(trainingResults.models).filter(m => m.status === 'failed').length,
                weightsSaved: savedWeights,
                weightsTotal: this.enabledModels.length,
                featureCount: currentFeatureCount,
                trainingType: config.autoTraining ? 'auto' : 'manual'
            });
            
        } catch (error) {
            Logger.error(`Ensemble training failed for ${pair}`, { error: error.message });
            
            // Save failed training attempt
            await this.mlStorage.saveTrainingHistory(`${pair}_ensemble`, {
                pair: pair,
                status: 'failed',
                error: error.message,
                startTime: Date.now(),
                endTime: Date.now(),
                trainingType: config.autoTraining ? 'auto' : 'manual'
            });
            
            throw error;
        } finally {
            // Remove from training progress tracking
            this.trainingInProgress.delete(trainingKey);
            Logger.info(`Training completed for ${pair}. Remaining in progress:`, Array.from(this.trainingInProgress));
        }
    }
    
    // Get trade signal based on prediction and confidence
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
            Logger.info('Starting Enhanced ML Server with Auto-Training, Weight Persistence, Tensor Fix, and Periodic Training...');
            
            // Wait for core service to be ready
            await this.dataClient.waitForCoreService();
            
            // 🆕 NEW: Start periodic training system
            if (this.periodicTrainingConfig.enabled) {
                this.startPeriodicTraining();
            }
            
            // Start HTTP server
            this.server = this.app.listen(this.port, () => {
                Logger.info(`Enhanced ML Server running at http://localhost:${this.port}`);
                console.log(`🤖 Enhanced ML API available at: http://localhost:${this.port}/api`);
                console.log(`📊 Health check: http://localhost:${this.port}/api/health`);
                console.log(`🔮 Auto-training predictions: http://localhost:${this.port}/api/predictions/RVN`);
                console.log(`🏆 Model comparison: http://localhost:${this.port}/api/models/RVN/compare`);
                console.log(`⚖️ Ensemble stats: http://localhost:${this.port}/api/ensemble/RVN/stats`);
                console.log(`💾 Storage stats: http://localhost:${this.port}/api/storage/stats`);
                console.log(`🔧 Rebuild models: http://localhost:${this.port}/api/models/RVN/rebuild`);
                console.log(`🎯 Manual training: http://localhost:${this.port}/api/train/RVN/ensemble`);
                console.log(`⏱️ Training status: http://localhost:${this.port}/api/training/RVN/status`);
                console.log(`🕐 Periodic training status: http://localhost:${this.port}/api/training/periodic/status`);
                console.log(`⚙️ Periodic training control: http://localhost:${this.port}/api/training/periodic/control`);
                console.log('');
                console.log('🚀 Advanced Features Available:');
                console.log(`   • Model Types: ${this.enabledModels.join(', ')}`);
                console.log(`   • Ensemble Strategy: ${this.ensembleStrategy}`);
                console.log(`   • 🆕 Auto-Training: Enabled`);
                console.log(`   • 🆕 Background Training: Enabled`);
                console.log(`   • 🆕 Tensor Disposal Fix: Applied`);
                console.log(`   • 🆕 Periodic Training: ${this.periodicTrainingConfig.enabled ? 'Enabled' : 'Disabled'}`);
                if (this.periodicTrainingConfig.enabled) {
                    console.log(`   • 🕐 Training Interval: ${this.periodicTrainingConfig.interval / 60000} minutes`);
                    console.log(`   • ⚡ Periodic Epochs: ${this.periodicTrainingConfig.epochs}`);
                    console.log(`   • 🔄 Max Concurrent: ${this.periodicTrainingConfig.maxConcurrent}`);
                }
                console.log(`   • Weight Persistence: Enabled`);
                console.log(`   • Individual Model Predictions`);
                console.log(`   • Performance Comparison`);
                console.log(`   • Dynamic Weight Updates`);
                console.log(`   • Dynamic Feature Count Handling`);
                console.log(`   • Trained Model Loading`);
                console.log(`   • Training Progress Tracking`);
                console.log(`   • Ensemble Error Recovery`);
            });
            
        } catch (error) {
            Logger.error('Failed to start Enhanced ML server', { error: error.message });
            process.exit(1);
        }
    }
    
    async stop() {
        Logger.info('Stopping Enhanced ML Server...');
        
        // 🆕 NEW: Stop periodic training
        this.stopPeriodicTraining();
        
        // Wait for any ongoing training to complete
        if (this.trainingInProgress.size > 0) {
            Logger.info('Waiting for ongoing training to complete...', {
                trainingInProgress: Array.from(this.trainingInProgress)
            });
            
            // Wait up to 60 seconds for training to complete (increased for periodic training)
            let waitTime = 0;
            while (this.trainingInProgress.size > 0 && waitTime < 60000) {
                await new Promise(resolve => setTimeout(resolve, 1000));
                waitTime += 1000;
            }
            
            if (this.trainingInProgress.size > 0) {
                Logger.warn('Training still in progress during shutdown', {
                    trainingInProgress: Array.from(this.trainingInProgress)
                });
            }
        }
        
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
        
        // Clear training progress tracking
        this.trainingInProgress.clear();
        
        Logger.info('Enhanced ML Server stopped');
    }
}

module.exports = MLServer;