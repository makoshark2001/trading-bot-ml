const express = require('express');
const config = require('config');
const DataClient = require('../data/DataClient');
const FeatureExtractor = require('../data/FeatureExtractor');
const DataPreprocessor = require('../data/DataPreprocessor');
const LSTMModel = require('../models/LSTMModel');
const { Logger } = require('../utils');

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
        // Health check
        this.app.get('/api/health', async (req, res) => {
            try {
                const coreHealth = await this.dataClient.checkCoreHealth();
                
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
        
        // Get ML predictions for a specific pair
        this.app.get('/api/predictions/:pair', async (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                const prediction = await this.getPrediction(pair);
                
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
        
        // Get extracted features for a pair (for debugging)
        this.app.get('/api/features/:pair', async (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                const pairData = await this.dataClient.getPairData(pair);
                const features = this.featureExtractor.extractFeatures(pairData);
                
                res.json({
                    pair,
                    features: {
                        count: features.features.length,
                        names: features.featureNames,
                        values: features.features.slice(0, 10), // First 10 for preview
                        metadata: features.metadata
                    },
                    timestamp: Date.now()
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
        
        // Train model for a specific pair
        this.app.post('/api/train/:pair', async (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                const trainingConfig = req.body || {};
                
                // Start training in background
                this.trainModel(pair, trainingConfig).catch(error => {
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
        
        // Get model status
        this.app.get('/api/models/:pair/status', (req, res) => {
            const pair = req.params.pair.toUpperCase();
            const model = this.models[pair];
            
            res.json({
                pair,
                hasModel: !!model,
                modelInfo: model ? model.getModelSummary() : null,
                timestamp: Date.now()
            });
        });
        
        // List all available endpoints
        this.app.get('/api', (req, res) => {
            res.json({
                service: 'trading-bot-ml',
                version: '1.0.0',
                endpoints: [
                    'GET /api/health - Service health check',
                    'GET /api/predictions - Get all predictions',
                    'GET /api/predictions/:pair - Get prediction for specific pair',
                    'GET /api/features/:pair - Get extracted features for pair',
                    'POST /api/train/:pair - Start training model for pair',
                    'GET /api/models/:pair/status - Get model status'
                ],
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
        
        const modelConfig = {
            ...config.get('ml.models.lstm'),
            features: featureCount
        };
        
        const model = new LSTMModel(modelConfig);
        model.buildModel();
        model.compileModel();
        
        this.models[pair] = model;
        
        // TODO: Load pre-trained model if exists
        // TODO: Train model with historical data
        
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
    
    async trainModel(pair, config = {}) {
        Logger.info(`Starting training for ${pair}`, config);
        
        try {
            // Get historical data
            const pairData = await this.dataClient.getPairData(pair);
            
            // TODO: Implement full training pipeline
            // 1. Extract features for multiple time points
            // 2. Create training dataset
            // 3. Train model
            // 4. Save model
            
            Logger.info(`Training completed for ${pair}`);
            
        } catch (error) {
            Logger.error(`Training failed for ${pair}`, { error: error.message });
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
            });
            
        } catch (error) {
            Logger.error('Failed to start ML server', { error: error.message });
            process.exit(1);
        }
    }
    
    async stop() {
        Logger.info('Stopping ML Server...');
        
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