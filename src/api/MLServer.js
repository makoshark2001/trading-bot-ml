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
        this.mlStorage = null; // Advanced persistence
        
        // Model configuration
        this.modelTypes = ['lstm', 'gru', 'cnn', 'transformer'];
        this.enabledModels = config.get('ml.ensemble.enabledModels') || this.modelTypes;
        this.ensembleStrategy = config.get('ml.ensemble.strategy') || 'weighted';
        
        this.initializeServices();
        this.setupRoutes();
        this.setupMiddleware();
    }
    
    initializeServices() {
        Logger.info('Initializing Enhanced ML services with Model Ensemble...');
        
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
        
        Logger.info('Enhanced ML services initialized successfully', {
            enabledModels: this.enabledModels,
            ensembleStrategy: this.ensembleStrategy
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
                        strategy: this.ensembleStrategy
                    },
                    predictions: {
                        cached: Object.keys(this.predictions).length,
                        lastUpdate: this.getLastPredictionTime()
                    },
                    storage: {
                        enabled: true,
                        stats: storageStats,
                        cacheSize: storageStats.cache
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
        
        // Enhanced prediction endpoint with ensemble
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
        
        // Train ensemble models
        this.app.post('/api/train/:pair/ensemble', async (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                const trainingConfig = req.body || {};
                
                // Start ensemble training in background
                this.trainEnsembleModels(pair, trainingConfig).catch(error => {
                    Logger.error(`Background ensemble training failed for ${pair}`, { 
                        error: error.message 
                    });
                });
                
                res.json({
                    message: `Ensemble training started for ${pair}`,
                    pair,
                    modelTypes: this.enabledModels,
                    config: trainingConfig,
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
        
        // Get individual model prediction
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
                
                const prediction = await this.getSingleModelPrediction(pair, modelType);
                
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
                
                // Load persistent model metadata
                const modelMetadata = this.mlStorage.loadModelMetadata(pair);
                const trainingHistory = this.mlStorage.loadTrainingHistory(pair);
                const ensembleMetadata = this.mlStorage.loadModelMetadata(`${pair}_ensemble`);
                
                const individualModels = {};
                for (const modelType of this.enabledModels) {
                    const model = pairModels[modelType];
                    individualModels[modelType] = {
                        hasModel: !!model,
                        modelInfo: model ? model.getModelSummary() : null
                    };
                }
                
                res.json({
                    pair,
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
                        lastTrained: trainingHistory ? trainingHistory.timestamp : null
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
        
        // Enhanced API endpoints list
        this.app.get('/api', (req, res) => {
            res.json({
                service: 'trading-bot-ml-enhanced',
                version: '2.0.0',
                features: [
                    'Model Ensemble System',
                    'Multiple Model Types (LSTM, GRU, CNN, Transformer)',
                    'Weighted Voting Strategies',
                    'Performance Tracking',
                    'Advanced Persistence'
                ],
                endpoints: [
                    'GET /api/health - Enhanced service health with ensemble info',
                    'GET /api/predictions/:pair - Ensemble predictions with strategy options',
                    'GET /api/predictions/:pair/history - Enhanced prediction history with filters',
                    'GET /api/ensemble/:pair/stats - Ensemble statistics and performance',
                    'POST /api/ensemble/:pair/weights - Update ensemble weights',
                    'POST /api/train/:pair/ensemble - Train all models in ensemble',
                    'GET /api/models/:pair/compare - Compare individual model performance',
                    'GET /api/models/:pair/:modelType/predict - Individual model predictions',
                    'GET /api/models/:pair/status - Enhanced model status with ensemble info',
                    'GET /api/features/:pair - Feature extraction with caching',
                    'GET /api/storage/stats - Storage statistics',
                    'POST /api/storage/save - Force save all data',
                    'POST /api/storage/cleanup - Clean up old files'
                ],
                modelTypes: this.enabledModels,
                ensembleStrategies: ['weighted', 'majority', 'average', 'confidence_weighted'],
                currentStrategy: this.ensembleStrategy,
                storage: {
                    enabled: true,
                    features: [
                        'Atomic file writes',
                        'Ensemble configuration persistence',
                        'Individual model metadata',
                        'Performance history tracking',
                        'Prediction history with ensemble info'
                    ]
                },
                timestamp: Date.now()
            });
        });
    }
    
    // Enhanced prediction with ensemble support
    async getEnsemblePrediction(pair, options = {}) {
        const cacheKey = `${pair}_ensemble`;
        
        // Check cache first
        if (this.predictions[cacheKey] && 
            (Date.now() - this.predictions[cacheKey].timestamp) < 60000) {
            return this.predictions[cacheKey];
        }
        
        try {
            // Get or create ensemble
            let ensemble = this.ensembles[pair];
            if (!ensemble) {
                ensemble = await this.getOrCreateEnsemble(pair);
            }
            
            // Get data and extract features
            const pairData = await this.dataClient.getPairData(pair);
            const features = this.featureExtractor.extractFeatures(pairData);
            
            // Prepare input for prediction
            const inputData = await this.prepareRealTimeInput(features.features);
            
            // Make ensemble prediction
            const prediction = await ensemble.predict(inputData, options);
            
            // Cache result
            this.predictions[cacheKey] = {
                ...prediction,
                timestamp: Date.now(),
                type: 'ensemble'
            };
            
            return this.predictions[cacheKey];
            
        } catch (error) {
            Logger.error(`Ensemble prediction failed for ${pair}`, { error: error.message });
            throw error;
        }
    }
    
    // Get single model prediction
    async getSingleModelPrediction(pair, modelType) {
        const cacheKey = `${pair}_${modelType}`;
        
        // Check cache first
        if (this.predictions[cacheKey] && 
            (Date.now() - this.predictions[cacheKey].timestamp) < 60000) {
            return this.predictions[cacheKey];
        }
        
        try {
            // Get or create model
            let model = this.models[pair] && this.models[pair][modelType];
            if (!model) {
                const pairData = await this.dataClient.getPairData(pair);
                const features = this.featureExtractor.extractFeatures(pairData);
                model = await this.getOrCreateModel(pair, modelType, features.features.length);
            }
            
            // Get data and extract features
            const pairData = await this.dataClient.getPairData(pair);
            const features = this.featureExtractor.extractFeatures(pairData);
            
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
                individual: {
                    prediction: prediction,
                    confidence: Math.abs(prediction - 0.5) * 2
                },
                metadata: {
                    timestamp: Date.now(),
                    version: '2.0.0',
                    type: 'individual_prediction'
                }
            };
            
            // Cache result
            this.predictions[cacheKey] = {
                ...result,
                timestamp: Date.now(),
                type: 'individual'
            };
            
            return this.predictions[cacheKey];
            
        } catch (error) {
            Logger.error(`Individual prediction failed for ${pair}:${modelType}`, { error: error.message });
            throw error;
        }
    }
    
    // Create or get ensemble for a pair
    async getOrCreateEnsemble(pair) {
        Logger.info(`Creating ensemble for ${pair}`);
        
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
        
        // Get feature count from data
        const pairData = await this.dataClient.getPairData(pair);
        const features = this.featureExtractor.extractFeatures(pairData);
        const featureCount = features.features.length;
        
        // Create and add individual models to ensemble
        for (const modelType of this.enabledModels) {
            try {
                const model = await this.getOrCreateModel(pair, modelType, featureCount);
                const weight = ensembleConfig.weights[modelType] || 1.0;
                
                ensemble.addModel(modelType, model, weight, {
                    pair: pair,
                    featureCount: featureCount,
                    created: Date.now()
                });
                
                Logger.info(`Added ${modelType} model to ${pair} ensemble`, { weight });
                
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
            modelTypes: this.enabledModels
        });
        
        return ensemble;
    }
    
    // Create individual model
    async getOrCreateModel(pair, modelType, featureCount) {
        if (!this.models[pair]) {
            this.models[pair] = {};
        }
        
        if (this.models[pair][modelType]) {
            return this.models[pair][modelType];
        }
        
        Logger.info(`Creating ${modelType} model for ${pair}`, { featureCount });
        
        const baseConfig = {
            sequenceLength: this.preprocessor.sequenceLength,
            features: featureCount
        };
        
        let model;
        switch (modelType) {
            case 'lstm':
                model = new LSTMModel({ ...baseConfig, ...config.get('ml.models.lstm') });
                break;
            case 'gru':
                model = new GRUModel({ ...baseConfig, ...config.get('ml.models.gru') });
                break;
            case 'cnn':
                model = new CNNModel({ ...baseConfig, ...config.get('ml.models.cnn') });
                break;
            case 'transformer':
                model = new TransformerModel({ ...baseConfig, ...config.get('ml.models.transformer') });
                break;
            default:
                throw new Error(`Unknown model type: ${modelType}`);
        }
        
        model.buildModel();
        model.compileModel();
        
        this.models[pair][modelType] = model;
        
        // Save model metadata
        await this.mlStorage.saveModelMetadata(`${pair}_${modelType}`, {
            config: baseConfig,
            modelType: modelType,
            created: Date.now(),
            featureCount,
            status: 'created'
        });
        
        return model;
    }
    
    // Prepare real-time input from features
    async prepareRealTimeInput(features) {
        // This is a simplified version - in practice, you'd use the preprocessor
        // to create proper sequences from historical feature data
        
        const sequenceLength = this.preprocessor.sequenceLength;
        
        // Create a mock sequence by repeating the current features
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
            timestamp: Date.now()
        };
        
        // Get sample data for comparison
        try {
            const pairData = await this.dataClient.getPairData(pair);
            const features = this.featureExtractor.extractFeatures(pairData);
            const inputData = await this.prepareRealTimeInput(features.features);
            
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
                            available: true
                        };
                        
                    } catch (error) {
                        comparison.models[modelType] = {
                            error: error.message,
                            available: false
                        };
                    }
                } else {
                    comparison.models[modelType] = {
                        available: false,
                        message: 'Model not created'
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
    
    // Train ensemble models
    async trainEnsembleModels(pair, config = {}) {
        Logger.info(`Starting ensemble training for ${pair}`, config);
        
        try {
            // Get historical data
            const pairData = await this.dataClient.getPairData(pair);
            const features = this.featureExtractor.extractFeatures(pairData);
            
            // Create targets for training
            const targets = this.featureExtractor.createTargets(pairData.history);
            const binaryTargets = targets[`direction_${config.targetPeriods || 5}`] || targets['direction_5'];
            
            if (!binaryTargets || binaryTargets.length === 0) {
                throw new Error('No training targets available');
            }
            
            // Prepare training data (simplified - in practice, use preprocessor)
            const featuresArray = Array(binaryTargets.length).fill().map(() => features.features);
            const processedData = await this.preprocessor.prepareTrainingData(featuresArray, binaryTargets);
            
            const trainingResults = {
                pair: pair,
                startTime: Date.now(),
                models: {},
                ensemble: null,
                config: config
            };
            
            // Train individual models
            for (const modelType of this.enabledModels) {
                try {
                    Logger.info(`Training ${modelType} model for ${pair}`);
                    
                    const model = await this.getOrCreateModel(pair, modelType, features.features.length);
                    
                    const modelTrainingConfig = {
                        epochs: config.epochs || 50,
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
                    
                    trainingResults.models[modelType] = {
                        status: 'completed',
                        finalMetrics: history.finalMetrics,
                        epochsCompleted: history.epochsCompleted || history.finalMetrics.epochsCompleted,
                        modelType: modelType
                    };
                    
                    // Save individual model training history
                    await this.mlStorage.saveTrainingHistory(`${pair}_${modelType}`, {
                        ...trainingResults.models[modelType],
                        pair: pair,
                        modelType: modelType,
                        timestamp: Date.now()
                    });
                    
                    Logger.info(`${modelType} training completed for ${pair}`, 
                        trainingResults.models[modelType].finalMetrics);
                    
                } catch (error) {
                    Logger.error(`${modelType} training failed for ${pair}`, { error: error.message });
                    trainingResults.models[modelType] = {
                        status: 'failed',
                        error: error.message,
                        modelType: modelType
                    };
                }
            }
            
            // Update ensemble weights based on training performance
            const ensemble = this.ensembles[pair];
            if (ensemble) {
                const performanceWeights = {};
                
                for (const [modelType, result] of Object.entries(trainingResults.models)) {
                    if (result.status === 'completed' && result.finalMetrics) {
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
                    performanceWeights: performanceWeights
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
            
            Logger.info(`Ensemble training completed for ${pair}`, {
                totalTime: trainingResults.totalTime,
                modelsCompleted: Object.values(trainingResults.models).filter(m => m.status === 'completed').length,
                modelsFailed: Object.values(trainingResults.models).filter(m => m.status === 'failed').length
            });
            
        } catch (error) {
            Logger.error(`Ensemble training failed for ${pair}`, { error: error.message });
            
            // Save failed training attempt
            await this.mlStorage.saveTrainingHistory(`${pair}_ensemble`, {
                pair: pair,
                status: 'failed',
                error: error.message,
                startTime: Date.now(),
                endTime: Date.now()
            });
            
            throw error;
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
            Logger.info('Starting Enhanced ML Server with Model Ensemble...');
            
            // Wait for core service to be ready
            await this.dataClient.waitForCoreService();
            
            // Start HTTP server
            this.server = this.app.listen(this.port, () => {
                Logger.info(`Enhanced ML Server running at http://localhost:${this.port}`);
                console.log(`🤖 Enhanced ML API available at: http://localhost:${this.port}/api`);
                console.log(`📊 Health check: http://localhost:${this.port}/api/health`);
                console.log(`🔮 Ensemble predictions: http://localhost:${this.port}/api/predictions/RVN`);
                console.log(`🏆 Model comparison: http://localhost:${this.port}/api/models/RVN/compare`);
                console.log(`⚖️ Ensemble stats: http://localhost:${this.port}/api/ensemble/RVN/stats`);
                console.log(`💾 Storage stats: http://localhost:${this.port}/api/storage/stats`);
                console.log('');
                console.log('🚀 Advanced Features Available:');
                console.log(`   • Model Types: ${this.enabledModels.join(', ')}`);
                console.log(`   • Ensemble Strategy: ${this.ensembleStrategy}`);
                console.log(`   • Individual Model Predictions`);
                console.log(`   • Performance Comparison`);
                console.log(`   • Dynamic Weight Updates`);
            });
            
        } catch (error) {
            Logger.error('Failed to start Enhanced ML server', { error: error.message });
            process.exit(1);
        }
    }
    
    async stop() {
        Logger.info('Stopping Enhanced ML Server...');
        
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
        
        Logger.info('Enhanced ML Server stopped');
    }
}

module.exports = MLServer;