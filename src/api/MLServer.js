const express = require('express');
const config = require('config');
const DataClient = require('../data/DataClient');
const FeatureExtractor = require('../data/FeatureExtractor');
const DataPreprocessor = require('../data/DataPreprocessor');
const LSTMModel = require('../models/LSTMModel');
const { Logger, MLStorage } = require('../utils');

class MLServer {
    constructor() {
        this.app = express();
        this.port = config.get('server.port') || 3001;
        this.startTime = Date.now();
        
        this.dataClient = null;
        this.featureExtractor = null;
        this.preprocessor = null;
        this.models = {}; // Store models for each pair
        this.predictions = {}; // Cache recent predictions
        this.mlStorage = null; // Advanced persistence
        
        this.initializeServices();
        this.setupRoutes();
        this.setupMiddleware();
    }
    
    initializeServices() {
        Logger.info('Initializing ML services...');
        
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
        
        Logger.info('ML services initialized successfully');
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
        // Health check with storage information
        this.app.get('/api/health', async (req, res) => {
            try {
                const coreHealth = await this.dataClient.checkCoreHealth();
                const storageStats = this.mlStorage.getStorageStats();
                
                res.json({
                    status: 'healthy',
                    service: 'trading-bot-ml',
                    timestamp: Date.now(),
                    uptime: this.getUptime(),
                    core: coreHealth,
                    models: {
                        loaded: Object.keys(this.models).length,
                        pairs: Object.keys(this.models)
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
                    service: 'trading-bot-ml',
                    error: error.message,
                    timestamp: Date.now()
                });
            }
        });
        
        // Storage management endpoints
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
        
        // Get ML predictions for a specific pair
        this.app.get('/api/predictions/:pair', async (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                const prediction = await this.getPrediction(pair);
                
                // Save prediction to persistent storage
                await this.mlStorage.savePredictionHistory(pair, {
                    ...prediction,
                    timestamp: Date.now(),
                    requestId: `${pair}_${Date.now()}`
                });
                
                res.json({
                    pair,
                    prediction,
                    timestamp: Date.now(),
                    cached: this.predictions[pair] && 
                           (Date.now() - this.predictions[pair].timestamp) < 60000 // 1 minute cache
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
        
        // Get predictions for all pairs
        this.app.get('/api/predictions', async (req, res) => {
            try {
                const allData = await this.dataClient.getAllData();
                const predictions = {};
                
                for (const pair of allData.pairs) {
                    try {
                        predictions[pair] = await this.getPrediction(pair);
                        
                        // Save to persistent storage
                        await this.mlStorage.savePredictionHistory(pair, {
                            ...predictions[pair],
                            timestamp: Date.now(),
                            requestId: `${pair}_batch_${Date.now()}`
                        });
                    } catch (error) {
                        Logger.warn(`Failed to get prediction for ${pair}`, { 
                            error: error.message 
                        });
                        predictions[pair] = {
                            error: error.message,
                            timestamp: Date.now()
                        };
                    }
                }
                
                res.json({
                    predictions,
                    timestamp: Date.now(),
                    pairs: allData.pairs
                });
                
            } catch (error) {
                Logger.error('Failed to get all predictions', { error: error.message });
                res.status(500).json({
                    error: 'Failed to get predictions',
                    message: error.message
                });
            }
        });
        
        // Get extracted features for a pair with caching
        this.app.get('/api/features/:pair', async (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                
                // Try to load from cache first
                const cachedFeatures = this.mlStorage.loadFeatureCache(pair);
                if (cachedFeatures && (Date.now() - cachedFeatures.timestamp) < 300000) { // 5 minute cache
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
                    values: features.features.slice(0, 10), // First 10 for preview
                    metadata: features.metadata
                });
                
                res.json({
                    pair,
                    features: {
                        count: features.features.length,
                        names: features.featureNames,
                        values: features.features.slice(0, 10), // First 10 for preview
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
        
        // Train model for a specific pair with history tracking
        this.app.post('/api/train/:pair', async (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                const trainingConfig = req.body || {};
                
                // Start training in background
                this.trainModelWithHistory(pair, trainingConfig).catch(error => {
                    Logger.error(`Background training failed for ${pair}`, { 
                        error: error.message 
                    });
                });
                
                res.json({
                    message: `Training started for ${pair}`,
                    pair,
                    config: trainingConfig,
                    timestamp: Date.now()
                });
                
            } catch (error) {
                Logger.error(`Training start failed for ${req.params.pair}`, { 
                    error: error.message 
                });
                res.status(500).json({
                    error: 'Training start failed',
                    message: error.message,
                    pair: req.params.pair.toUpperCase()
                });
            }
        });
        
        // Get model status with persistent data
        this.app.get('/api/models/:pair/status', async (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                const model = this.models[pair];
                
                // Load persistent model metadata
                const modelMetadata = this.mlStorage.loadModelMetadata(pair);
                const trainingHistory = this.mlStorage.loadTrainingHistory(pair);
                
                res.json({
                    pair,
                    hasModel: !!model,
                    modelInfo: model ? model.getModelSummary() : null,
                    persistent: {
                        metadata: modelMetadata,
                        trainingHistory: trainingHistory,
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
        
        // Get prediction history for a pair
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
                
                if (since) {
                    predictions = predictions.filter(p => p.timestamp >= since);
                }
                
                predictions = predictions.slice(-limit); // Get most recent
                
                res.json({
                    pair,
                    predictions,
                    count: predictions.length,
                    totalCount: history.count,
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
        
        // List all available endpoints
        this.app.get('/api', (req, res) => {
            res.json({
                service: 'trading-bot-ml',
                version: '1.0.0',
                endpoints: [
                    'GET /api/health - Service health check with storage info',
                    'GET /api/predictions - Get all predictions',
                    'GET /api/predictions/:pair - Get prediction for specific pair',
                    'GET /api/predictions/:pair/history - Get prediction history for pair',
                    'GET /api/features/:pair - Get extracted features for pair',
                    'POST /api/train/:pair - Start training model for pair',
                    'GET /api/models/:pair/status - Get model status with history',
                    'GET /api/storage/stats - Get storage statistics',
                    'POST /api/storage/save - Force save all data',
                    'POST /api/storage/cleanup - Clean up old files'
                ],
                storage: {
                    enabled: true,
                    features: [
                        'Atomic file writes',
                        'Prediction history tracking',
                        'Model metadata persistence',
                        'Feature caching',
                        'Training history storage',
                        'Automatic cleanup'
                    ]
                },
                timestamp: Date.now()
            });
        });
    }
    
    async getPrediction(pair) {
        // Check cache first
        if (this.predictions[pair] && 
            (Date.now() - this.predictions[pair].timestamp) < 60000) { // 1 minute cache
            return this.predictions[pair];
        }
        
        try {
            // Get data from core
            const pairData = await this.dataClient.getPairData(pair);
            
            // Extract features
            const features = this.featureExtractor.extractFeatures(pairData);
            
            // Get or create model
            let model = this.models[pair];
            if (!model) {
                model = await this.getOrCreateModel(pair, features.features.length);
            }
            
            // Make prediction
            const prediction = await this.makePrediction(model, features.features);
            
            // Cache result
            this.predictions[pair] = {
                ...prediction,
                timestamp: Date.now()
            };
            
            return this.predictions[pair];
            
        } catch (error) {
            Logger.error(`Prediction failed for ${pair}`, { error: error.message });
            throw error;
        }
    }
    
    async getOrCreateModel(pair, featureCount) {
        Logger.info(`Creating new model for ${pair}`, { featureCount });
        
        // Check if we have persistent model metadata
        const modelMetadata = this.mlStorage.loadModelMetadata(pair);
        
        const modelConfig = {
            ...config.get('ml.models.lstm'),
            features: featureCount
        };
        
        const model = new LSTMModel(modelConfig);
        model.buildModel();
        model.compileModel();
        
        this.models[pair] = model;
        
        // Save model metadata
        await this.mlStorage.saveModelMetadata(pair, {
            config: modelConfig,
            created: Date.now(),
            featureCount,
            status: 'created'
        });
        
        return model;
    }
    
    async makePrediction(model, features) {
        try {
            // For now, return a mock prediction
            // In real implementation, we would:
            // 1. Prepare features for real-time prediction
            // 2. Use trained model to predict
            // 3. Return formatted prediction
            
            const mockPrediction = Math.random();
            
            return {
                direction: mockPrediction > 0.5 ? 'up' : 'down',
                confidence: Math.abs(mockPrediction - 0.5) * 2,
                probability: mockPrediction,
                signal: mockPrediction > 0.7 ? 'BUY' : 
                        mockPrediction < 0.3 ? 'SELL' : 'HOLD',
                features: {
                    count: features.length,
                    sample: features.slice(0, 5) // First 5 features for debugging
                },
                model: 'LSTM',
                version: '1.0.0'
            };
            
        } catch (error) {
            Logger.error('Prediction failed', { error: error.message });
            throw error;
        }
    }
    
    async trainModelWithHistory(pair, config = {}) {
        Logger.info(`Starting training for ${pair}`, config);
        
        try {
            // Get historical data
            const pairData = await this.dataClient.getPairData(pair);
            
            const trainingResults = {
                pair,
                config,
                startTime: Date.now(),
                status: 'completed',
                // TODO: Implement actual training results
                epochs: config.epochs || 100,
                finalLoss: Math.random() * 0.1,
                finalAccuracy: 0.6 + Math.random() * 0.3,
                trainingTime: Math.floor(Math.random() * 600000) + 300000 // 5-15 minutes
            };
            
            trainingResults.endTime = trainingResults.startTime + trainingResults.trainingTime;
            
            // Save training history
            await this.mlStorage.saveTrainingHistory(pair, trainingResults);
            
            // Update model metadata
            await this.mlStorage.saveModelMetadata(pair, {
                lastTrained: trainingResults.endTime,
                trainingConfig: config,
                performance: {
                    loss: trainingResults.finalLoss,
                    accuracy: trainingResults.finalAccuracy
                },
                status: 'trained'
            });
            
            Logger.info(`Training completed for ${pair}`, trainingResults);
            
        } catch (error) {
            Logger.error(`Training failed for ${pair}`, { error: error.message });
            
            // Save failed training attempt
            await this.mlStorage.saveTrainingHistory(pair, {
                pair,
                config,
                startTime: Date.now(),
                status: 'failed',
                error: error.message,
                endTime: Date.now()
            });
            
            throw error;
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
            Logger.info('Starting ML Server...');
            
            // Wait for core service to be ready
            await this.dataClient.waitForCoreService();
            
            // Start HTTP server
            this.server = this.app.listen(this.port, () => {
                Logger.info(`ML Server running at http://localhost:${this.port}`);
                console.log(`🤖 ML API available at: http://localhost:${this.port}/api`);
                console.log(`📊 Health check: http://localhost:${this.port}/api/health`);
                console.log(`🔮 Predictions: http://localhost:${this.port}/api/predictions`);
                console.log(`💾 Storage stats: http://localhost:${this.port}/api/storage/stats`);
            });
            
        } catch (error) {
            Logger.error('Failed to start ML server', { error: error.message });
            process.exit(1);
        }
    }
    
    async stop() {
        Logger.info('Stopping ML Server...');
        
        // Shutdown storage system gracefully
        if (this.mlStorage) {
            await this.mlStorage.shutdown();
        }
        
        // Dispose of all models
        Object.values(this.models).forEach(model => {
            if (model && typeof model.dispose === 'function') {
                model.dispose();
            }
        });
        
        if (this.preprocessor) {
            this.preprocessor.dispose();
        }
        
        if (this.server) {
            this.server.close();
        }
        
        Logger.info('ML Server stopped');
    }
}

module.exports = MLServer;