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
const { Logger, MLStorage, TrainingQueueManager } = require('../utils');

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
        
        // Enhanced caching for better performance
        this.predictionCache = new Map(); // Fast prediction cache
        this.featureCache = new Map(); // Feature cache
        this.modelStatusCache = new Map(); // Model status cache
        this.lastHealthCheck = null; // Health check cache
        
        // Model configuration
        this.modelTypes = ['lstm', 'gru', 'cnn', 'transformer'];
        this.enabledModels = config.get('ml.ensemble.enabledModels') || ['lstm']; // Start with just LSTM for speed
        this.ensembleStrategy = config.get('ml.ensemble.strategy') || 'weighted';
        
        // Performance optimization flags
        this.quickMode = process.env.ML_QUICK_MODE === 'true' || true; // Enable quick mode by default
        this.cacheTimeout = 30000; // 30 second cache timeout
        
        // Initialize services first
        this.initializeServices();
        
        // Create training queue AFTER everything is initialized
        this.initializeTrainingQueue();
        
        this.setupRoutes();
        this.setupMiddleware();
    }
    
    initializeServices() {
        Logger.info('Initializing ML services with training queue management...');
        
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
        
        Logger.info('ML services initialized successfully', {
            enabledModels: this.enabledModels,
            ensembleStrategy: this.ensembleStrategy,
            quickMode: this.quickMode,
            cacheTimeout: this.cacheTimeout
        });
    }
    
    initializeTrainingQueue() {
        try {
            // Add a small delay to ensure Logger is fully ready
            setTimeout(() => {
                this.trainingQueue = new TrainingQueueManager({
                    maxConcurrentTraining: 1, // Only 1 training at a time
                    trainingCooldown: 1800000, // 30 minutes between training sessions
                    processingInterval: 5000 // Check queue every 5 seconds
                });
                
                Logger.info('Training queue manager initialized successfully', {
                    maxConcurrentTraining: 1,
                    cooldownMinutes: 30
                });
            }, 100); // 100ms delay
            
        } catch (error) {
            console.error('Failed to initialize training queue:', error.message);
            // Create a dummy training queue to prevent errors
            this.trainingQueue = {
                getQueueStatus: () => ({ active: { count: 0, jobs: [] }, queued: { count: 0, jobs: [] } }),
                addTrainingJob: () => Promise.reject(new Error('Training queue not available')),
                canTrain: () => ({ allowed: false, reason: 'Training queue not initialized' }),
                cancelTraining: () => false,
                emergencyStop: () => ({ queuedJobsCancelled: 0, activeJobsMarked: 0 }),
                clearCooldown: () => false,
                clearAllCooldowns: () => 0,
                maxConcurrentTraining: 1,
                trainingCooldown: 1800000,
                shutdown: () => Promise.resolve()
            };
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
        
        // Request timing middleware
        this.app.use((req, res, next) => {
            req.startTime = Date.now();
            res.on('finish', () => {
                const duration = Date.now() - req.startTime;
                if (duration > 5000) { // Log slow requests
                    Logger.warn('Slow request detected', {
                        method: req.method,
                        url: req.url,
                        duration: `${duration}ms`,
                        ip: req.ip
                    });
                }
            });
            next();
        });
    }
    
    setupRoutes() {
        // Fast health check with training queue status
        this.app.get('/api/health', async (req, res) => {
            try {
                // Use cached health check if recent
                if (this.lastHealthCheck && (Date.now() - this.lastHealthCheck.timestamp) < 10000) {
                    return res.json(this.lastHealthCheck);
                }
                
                // Get training queue status
                const queueStatus = this.trainingQueue.getQueueStatus();
                
                // Quick health check
                const quickHealth = {
                    status: 'healthy',
                    service: 'trading-bot-ml-optimized',
                    timestamp: Date.now(),
                    uptime: this.getUptime(),
                    quickMode: this.quickMode,
                    models: {
                        individual: {
                            loaded: Object.keys(this.models).length,
                            pairs: Object.keys(this.models)
                        },
                        ensembles: {
                            loaded: Object.keys(this.ensembles).length,
                            pairs: Object.keys(this.ensembles)
                        },
                        enabledTypes: this.enabledModels,
                        strategy: this.ensembleStrategy,
                        featureCounts: this.featureCounts
                    },
                    predictions: {
                        cached: this.predictionCache.size,
                        lastUpdate: this.getLastPredictionTime()
                    },
                    training: {
                        queue: queueStatus,
                        maxConcurrent: this.trainingQueue.maxConcurrentTraining,
                        cooldownMinutes: this.trainingQueue.trainingCooldown / 1000 / 60
                    },
                    performance: {
                        cacheHits: this.getCacheHitRate(),
                        avgResponseTime: this.getAverageResponseTime()
                    }
                };
                
                // Try core health check in background (non-blocking)
                this.checkCoreHealthBackground().catch(err => {
                    Logger.warn('Background core health check failed', { error: err.message });
                });
                
                this.lastHealthCheck = quickHealth;
                res.json(quickHealth);
                
            } catch (error) {
                Logger.error('Health check failed', { error: error.message });
                res.status(500).json({
                    status: 'unhealthy',
                    service: 'trading-bot-ml-optimized',
                    error: error.message,
                    timestamp: Date.now()
                });
            }
        });
        
        // Fast prediction endpoint with aggressive caching
        this.app.get('/api/predictions/:pair', async (req, res) => {
            const requestStart = Date.now();
            
            try {
                const pair = req.params.pair.toUpperCase();
                const useEnsemble = req.query.ensemble !== 'false';
                const strategy = req.query.strategy || this.ensembleStrategy;
                
                // Create cache key
                const cacheKey = `${pair}_${useEnsemble ? 'ensemble' : 'single'}_${strategy}`;
                
                // Check cache first (aggressive caching)
                const cached = this.predictionCache.get(cacheKey);
                if (cached && (Date.now() - cached.timestamp) < this.cacheTimeout) {
                    Logger.debug(`Cache hit for prediction: ${pair}`, {
                        cacheAge: Date.now() - cached.timestamp,
                        responseTime: Date.now() - requestStart
                    });
                    
                    return res.json({
                        ...cached.data,
                        cached: true,
                        cacheAge: Date.now() - cached.timestamp,
                        responseTime: Date.now() - requestStart
                    });
                }
                
                let prediction;
                if (useEnsemble && !this.quickMode) {
                    prediction = await this.getEnsemblePrediction(pair, { strategy });
                } else {
                    // Use single fastest model in quick mode
                    const fastModel = this.quickMode ? 'lstm' : (req.query.model || 'lstm');
                    prediction = await this.getSingleModelPrediction(pair, fastModel);
                }
                
                // Cache the result
                this.predictionCache.set(cacheKey, {
                    data: {
                        pair,
                        prediction,
                        ensemble: useEnsemble,
                        strategy: useEnsemble ? strategy : null,
                        timestamp: Date.now(),
                        responseTime: Date.now() - requestStart
                    },
                    timestamp: Date.now()
                });
                
                // Clean old cache entries
                this.cleanOldCacheEntries();
                
                res.json({
                    pair,
                    prediction,
                    ensemble: useEnsemble,
                    strategy: useEnsemble ? strategy : null,
                    timestamp: Date.now(),
                    cached: false,
                    responseTime: Date.now() - requestStart
                });
                
                // Save prediction to persistent storage (async, non-blocking)
                this.mlStorage.savePredictionHistory(pair, {
                    ...prediction,
                    timestamp: Date.now(),
                    requestId: `${pair}_${Date.now()}`,
                    useEnsemble: useEnsemble,
                    strategy: useEnsemble ? strategy : null
                }).catch(error => {
                    Logger.warn('Failed to save prediction history', { error: error.message });
                });
                
            } catch (error) {
                const responseTime = Date.now() - requestStart;
                Logger.error(`Prediction failed for ${req.params.pair}`, { 
                    error: error.message,
                    responseTime
                });
                res.status(500).json({
                    error: 'Prediction failed',
                    message: error.message,
                    pair: req.params.pair.toUpperCase(),
                    responseTime
                });
            }
        });
        
        // Continue with remaining routes...
        this.setupTrainingRoutes();
        this.setupModelRoutes();
        this.setupUtilityRoutes();
    }
    
    // Training-related routes
    setupTrainingRoutes() {
        // Training queue status
        this.app.get('/api/training/queue', (req, res) => {
            try {
                const queueStatus = this.trainingQueue.getQueueStatus();
                res.json({
                    queue: queueStatus,
                    timestamp: Date.now()
                });
            } catch (error) {
                Logger.error('Failed to get training queue status', { error: error.message });
                res.status(500).json({
                    error: 'Failed to get training queue status',
                    message: error.message
                });
            }
        });
        
        // Queue-managed training endpoint
        this.app.post('/api/train/:pair/:modelType', async (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                const modelType = req.params.modelType.toLowerCase();
                const trainingConfig = req.body || {};
                
                // Check if training is allowed
                const canTrain = this.trainingQueue.canTrain(pair, modelType);
                if (!canTrain.allowed) {
                    return res.status(429).json({
                        error: 'Training not allowed',
                        reason: canTrain.reason,
                        details: canTrain,
                        pair,
                        modelType
                    });
                }
                
                // Add training job to queue
                const jobId = await this.trainingQueue.addTrainingJob(
                    pair,
                    modelType,
                    this.performModelTraining.bind(this), // Bind training function
                    {
                        ...trainingConfig,
                        priority: trainingConfig.priority || 5,
                        maxAttempts: trainingConfig.maxAttempts || 2
                    }
                );
                
                res.json({
                    message: `Training job queued for ${pair}:${modelType}`,
                    jobId,
                    pair,
                    modelType,
                    canTrain,
                    queueStatus: this.trainingQueue.getQueueStatus(),
                    timestamp: Date.now()
                });
                
            } catch (error) {
                Logger.error(`Failed to queue training for ${req.params.pair}:${req.params.modelType}`, { 
                    error: error.message 
                });
                res.status(500).json({
                    error: 'Failed to queue training',
                    message: error.message,
                    pair: req.params.pair.toUpperCase(),
                    modelType: req.params.modelType.toLowerCase()
                });
            }
        });
        
        // Cancel training job
        this.app.delete('/api/training/job/:jobId', async (req, res) => {
            try {
                const jobId = req.params.jobId;
                const reason = req.body.reason || 'User requested cancellation';
                
                const cancelled = await this.trainingQueue.cancelTraining(jobId, reason);
                
                if (cancelled) {
                    res.json({
                        message: 'Training job cancelled',
                        jobId,
                        reason,
                        timestamp: Date.now()
                    });
                } else {
                    res.status(404).json({
                        error: 'Training job not found',
                        jobId
                    });
                }
                
            } catch (error) {
                Logger.error(`Failed to cancel training job ${req.params.jobId}`, { 
                    error: error.message 
                });
                res.status(500).json({
                    error: 'Failed to cancel training job',
                    message: error.message,
                    jobId: req.params.jobId
                });
            }
        });
        
        // Emergency stop all training
        this.app.post('/api/training/emergency-stop', (req, res) => {
            try {
                const result = this.trainingQueue.emergencyStop();
                
                Logger.warn('Emergency stop activated via API', result);
                
                res.json({
                    message: 'Emergency stop activated',
                    result,
                    timestamp: Date.now()
                });
                
            } catch (error) {
                Logger.error('Failed to execute emergency stop', { error: error.message });
                res.status(500).json({
                    error: 'Failed to execute emergency stop',
                    message: error.message
                });
            }
        });
        
        // Clear training cooldowns (admin function)
        this.app.post('/api/training/clear-cooldowns', (req, res) => {
            try {
                const pair = req.body.pair;
                const modelType = req.body.modelType;
                
                let result;
                if (pair && modelType) {
                    result = this.trainingQueue.clearCooldown(pair, modelType);
                    res.json({
                        message: `Cooldown cleared for ${pair}:${modelType}`,
                        cleared: result,
                        timestamp: Date.now()
                    });
                } else {
                    result = this.trainingQueue.clearAllCooldowns();
                    res.json({
                        message: 'All cooldowns cleared',
                        count: result,
                        timestamp: Date.now()
                    });
                }
                
            } catch (error) {
                Logger.error('Failed to clear cooldowns', { error: error.message });
                res.status(500).json({
                    error: 'Failed to clear cooldowns',
                    message: error.message
                });
            }
        });
    }
    // Model-related routes
    setupModelRoutes() {
        // Fast model status endpoint
        this.app.get('/api/models/:pair/status', async (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                
                // Check cache first
                const cacheKey = `status_${pair}`;
                const cached = this.modelStatusCache.get(cacheKey);
                if (cached && (Date.now() - cached.timestamp) < 60000) { // 1 minute cache
                    return res.json({
                        ...cached.data,
                        cached: true,
                        cacheAge: Date.now() - cached.timestamp
                    });
                }
                
                const pairModels = this.models[pair] || {};
                const ensemble = this.ensembles[pair];
                
                // Get training status for this pair
                const trainingStatus = {};
                for (const modelType of this.enabledModels) {
                    const canTrain = this.trainingQueue.canTrain(pair, modelType);
                    trainingStatus[modelType] = canTrain;
                }
                
                const individualModels = {};
                for (const modelType of this.enabledModels) {
                    const model = pairModels[modelType];
                    individualModels[modelType] = {
                        hasModel: !!model,
                        modelInfo: model ? {
                            totalParams: model.model?.countParams?.() || 0,
                            layers: model.model?.layers?.length || 0,
                            isCompiled: model.isCompiled || false,
                            isTraining: model.isTraining || false
                        } : null,
                        training: trainingStatus[modelType]
                    };
                }
                
                const statusData = {
                    pair,
                    featureCount: this.featureCounts[pair] || 'unknown',
                    individual: individualModels,
                    ensemble: {
                        hasEnsemble: !!ensemble,
                        stats: ensemble ? {
                            modelCount: ensemble.models.size,
                            votingStrategy: ensemble.votingStrategy,
                            performanceHistorySize: ensemble.performanceHistory.size
                        } : null,
                        strategy: this.ensembleStrategy,
                        enabledModels: this.enabledModels
                    },
                    trainingQueue: this.trainingQueue.getQueueStatus(),
                    quickMode: this.quickMode,
                    timestamp: Date.now()
                };
                
                // Cache the result
                this.modelStatusCache.set(cacheKey, {
                    data: statusData,
                    timestamp: Date.now()
                });
                
                res.json({
                    ...statusData,
                    cached: false
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
    }
    
    // Utility routes
    setupUtilityRoutes() {
        // Fast features endpoint with caching
        this.app.get('/api/features/:pair', async (req, res) => {
            try {
                const pair = req.params.pair.toUpperCase();
                
                // Check cache first
                const cacheKey = `features_${pair}`;
                const cached = this.featureCache.get(cacheKey);
                if (cached && (Date.now() - cached.timestamp) < 300000) { // 5 minute cache
                    return res.json({
                        ...cached.data,
                        cached: true,
                        cacheAge: Date.now() - cached.timestamp
                    });
                }
                
                // Extract features (this is the slow part)
                const pairData = await this.dataClient.getPairData(pair);
                const features = this.featureExtractor.extractFeatures(pairData);
                
                // Update feature count tracking
                this.featureCounts[pair] = features.features.length;
                
                const featureData = {
                    pair,
                    features: {
                        count: features.features.length,
                        names: features.featureNames,
                        values: features.features.slice(0, 10), // Only return first 10 values
                        metadata: features.metadata
                    },
                    timestamp: Date.now(),
                    cached: false
                };
                
                // Cache the result
                this.featureCache.set(cacheKey, {
                    data: featureData,
                    timestamp: Date.now()
                });
                
                res.json(featureData);
                
                // Save to persistent cache (async, non-blocking)
                this.mlStorage.saveFeatureCache(pair, {
                    count: features.features.length,
                    names: features.featureNames,
                    values: features.features.slice(0, 10),
                    metadata: features.metadata
                }).catch(error => {
                    Logger.warn('Failed to save feature cache', { error: error.message });
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
        
        // Storage stats endpoint
        this.app.get('/api/storage/stats', (req, res) => {
            try {
                const stats = this.mlStorage.getStorageStats();
                res.json({
                    storage: stats,
                    performance: {
                        predictionCacheSize: this.predictionCache.size,
                        featureCacheSize: this.featureCache.size,
                        modelStatusCacheSize: this.modelStatusCache.size
                    },
                    trainingQueue: this.trainingQueue.getQueueStatus(),
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
        
        // Quick API info endpoint
        this.app.get('/api', (req, res) => {
            res.json({
                service: 'trading-bot-ml-optimized',
                version: '2.0.0-performance-queue',
                quickMode: this.quickMode,
                features: [
                    'Optimized Response Times',
                    'Aggressive Caching',
                    'Training Queue Management',
                    'Concurrent Training Prevention',
                    'Background Processing',
                    'Performance Monitoring'
                ],
                endpoints: [
                    'GET /api/health - Fast health check with training queue status',
                    'GET /api/predictions/:pair - Fast predictions with 30s cache',
                    'GET /api/features/:pair - Feature extraction with 5min cache',
                    'GET /api/models/:pair/status - Model status with training info',
                    'GET /api/training/queue - Training queue status and history',
                    'POST /api/train/:pair/:modelType - Queue-managed training',
                    'DELETE /api/training/job/:jobId - Cancel training job',
                    'POST /api/training/emergency-stop - Emergency stop all training',
                    'POST /api/training/clear-cooldowns - Clear training cooldowns (admin)',
                    'GET /api/storage/stats - Storage and performance statistics'
                ],
                performance: {
                    cacheTimeout: this.cacheTimeout,
                    enabledModels: this.enabledModels,
                    currentCacheSize: {
                        predictions: this.predictionCache.size,
                        features: this.featureCache.size,
                        modelStatus: this.modelStatusCache.size
                    }
                },
                trainingQueue: {
                    maxConcurrentTraining: this.trainingQueue.maxConcurrentTraining,
                    cooldownMinutes: this.trainingQueue.trainingCooldown / 1000 / 60,
                    currentStatus: this.trainingQueue.getQueueStatus()
                },
                timestamp: Date.now()
            });
        });
    }
    
    // Training function that will be called by the queue manager
    async performModelTraining(pair, modelType, config) {
        Logger.info(`Performing queued training for ${pair}:${modelType}`, config);
        
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
            
            // Get or create model
            const model = await this.getOrCreateModel(pair, modelType, currentFeatureCount);
            
            const modelTrainingConfig = {
                epochs: config.epochs || 25, // Reduced for queue management
                batchSize: config.batchSize || 32,
                verbose: 0, // Silent training
                ...config
            };
            
            // Perform training
            const history = await model.train(
                processedData.trainX,
                processedData.trainY,
                processedData.validationX,
                processedData.validationY,
                modelTrainingConfig
            );
            
            // Save model weights if training was successful
            if (history.finalMetrics && parseFloat(history.finalMetrics.finalAccuracy) > 0.5) {
                try {
                    await this.mlStorage.saveModelWeights(pair, modelType, model);
                    Logger.info(`Model weights saved for ${pair}:${modelType}`);
                } catch (saveError) {
                    Logger.warn(`Failed to save model weights for ${pair}:${modelType}`, { 
                        error: saveError.message 
                    });
                }
            }
            
            const trainingResults = {
                pair: pair,
                modelType: modelType,
                status: 'completed',
                finalMetrics: history.finalMetrics,
                epochsCompleted: history.epochsCompleted || history.finalMetrics.epochsCompleted,
                featureCount: currentFeatureCount,
                timestamp: Date.now()
            };
            
            // Clean up tensors
            processedData.trainX.dispose();
            processedData.trainY.dispose();
            processedData.validationX.dispose();
            processedData.validationY.dispose();
            processedData.testX.dispose();
            processedData.testY.dispose();
            
            // Save training history
            await this.mlStorage.saveTrainingHistory(`${pair}_${modelType}`, trainingResults);
            
            Logger.info(`Queued training completed for ${pair}:${modelType}`, trainingResults.finalMetrics);
            
            return trainingResults;
            
        } catch (error) {
            Logger.error(`Queued training failed for ${pair}:${modelType}`, { error: error.message });
            throw error;
        }
    }
    
    // Optimized single model prediction (fastest path)
    async getSingleModelPrediction(pair, modelType = 'lstm') {
        const cacheKey = `${pair}_${modelType}`;
        
        // Check cache first
        if (this.predictions[cacheKey] && 
            (Date.now() - this.predictions[cacheKey].timestamp) < this.cacheTimeout) {
            return this.predictions[cacheKey];
        }
        
        try {
            // Get data and extract features first to check feature count
            const pairData = await this.dataClient.getPairData(pair);
            const features = this.featureExtractor.extractFeatures(pairData);
            const currentFeatureCount = features.features.length;
            
            // Update feature count tracking
            this.featureCounts[pair] = currentFeatureCount;
            
            // Get or create model with correct feature count
            let model = this.models[pair] && this.models[pair][modelType];
            if (!model) {
                model = await this.getOrCreateModel(pair, modelType, currentFeatureCount);
            }
            
            // Prepare input for prediction
            const inputData = await this.prepareRealTimeInput(features.features);
            
            // Make prediction
            const predictions = await model.predict(inputData);
            const prediction = Array.isArray(predictions) ? predictions[0] : predictions;
            
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
                    version: '2.0.0-performance-queue',
                    type: 'individual_prediction',
                    featureCount: currentFeatureCount,
                    quickMode: this.quickMode
                }
            };
            
            // Cache result
            this.predictions[cacheKey] = {
                ...result,
                timestamp: Date.now(),
                type: 'individual'
            };
            
            // Clean up input tensor
            if (inputData && typeof inputData.dispose === 'function') {
                inputData.dispose();
            }
            
            return this.predictions[cacheKey];
            
        } catch (error) {
            Logger.error(`Individual prediction failed for ${pair}:${modelType}`, { error: error.message });
            
            // Return a fallback prediction
            return {
                prediction: 0.5,
                confidence: 0.1,
                direction: 'neutral',
                signal: 'HOLD',
                modelType: modelType,
                individual: {
                    prediction: 0.5,
                    confidence: 0.1
                },
                metadata: {
                    timestamp: Date.now(),
                    version: '2.0.0-performance-queue',
                    type: 'fallback_prediction',
                    error: error.message,
                    quickMode: this.quickMode
                }
            };
        }
    }
    
    // Simplified ensemble prediction (only if not in quick mode)
    async getEnsemblePrediction(pair, options = {}) {
        if (this.quickMode) {
            // In quick mode, just use the fastest single model
            return this.getSingleModelPrediction(pair, 'lstm');
        }
        
        // Full ensemble logic would go here
        // For now, fallback to single model
        return this.getSingleModelPrediction(pair, 'lstm');
    }
    
    // Create individual model - optimized for speed with weight loading
    async getOrCreateModel(pair, modelType, featureCount) {
        if (!this.models[pair]) {
            this.models[pair] = {};
        }
        
        // Check if model exists and has correct feature count
        if (this.models[pair][modelType]) {
            const existingModel = this.models[pair][modelType];
            if (existingModel.features === featureCount) {
                return existingModel; // Model is correct, return it
            } else {
                // Feature count mismatch, dispose and recreate
                if (existingModel.dispose) {
                    existingModel.dispose();
                }
                delete this.models[pair][modelType];
            }
        }
        
        Logger.info(`Creating optimized ${modelType} model for ${pair}`, { featureCount });
        
        // Create optimized config for fast training/inference
        const baseConfig = {
            sequenceLength: 30, // Reduced from 60 for speed
            features: featureCount
        };
        
        // Get model-specific config with optimizations
        const modelSpecificConfig = config.get(`ml.models.${modelType}`) || {};
        const finalConfig = {
            ...modelSpecificConfig,
            ...baseConfig,
            // Performance optimizations
            units: Math.min(modelSpecificConfig.units || 50, 32), // Smaller networks
            layers: Math.min(modelSpecificConfig.layers || 2, 1), // Fewer layers
            epochs: Math.min(modelSpecificConfig.epochs || 100, 10), // Fewer epochs for quick mode
            dropout: 0.1 // Reduced dropout
        };
        
        // Try to load pre-trained weights first
        let model;
        const ModelClass = this.getModelClass(modelType);
        
        if (this.mlStorage.hasTrainedWeights(pair, modelType)) {
            try {
                model = await this.mlStorage.loadModelWeights(pair, modelType, ModelClass, finalConfig);
                if (model) {
                    Logger.info(`Loaded pre-trained ${modelType} model for ${pair}`, {
                        featureCount,
                        params: model.model?.countParams?.() || 0
                    });
                    model.features = featureCount;
                    this.models[pair][modelType] = model;
                    return model;
                }
            } catch (loadError) {
                Logger.warn(`Failed to load pre-trained weights for ${pair}:${modelType}`, {
                    error: loadError.message
                });
            }
        }
        
        // Create new model if loading failed or no weights exist
        model = new ModelClass(finalConfig);
        model.buildModel();
        model.compileModel();
        model.features = featureCount; // Store feature count for quick access
        
        this.models[pair][modelType] = model;
        
        Logger.info(`New ${modelType} model created for ${pair}`, {
            featureCount,
            params: model.model?.countParams?.() || 0,
            hasPretrainedWeights: false
        });
        
        return model;
    }
    
    // Get model class by type
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
    
    // Optimized input preparation
    async prepareRealTimeInput(features) {
        const tf = require('@tensorflow/tfjs');
        const sequenceLength = 30; // Reduced for performance
        
        // Create a mock sequence by repeating the current features
        const sequence = Array(sequenceLength).fill(features);
        
        // Convert to tensor
        const inputTensor = tf.tensor3d([sequence]); // Shape: [1, sequenceLength, features]
        
        return inputTensor;
    }
    
    // Background core health check (non-blocking)
    async checkCoreHealthBackground() {
        try {
            await this.dataClient.checkCoreHealth();
        } catch (error) {
            // Don't throw, just log
            Logger.warn('Background core health check failed', { error: error.message });
        }
    }
    
    // Clean old cache entries
    cleanOldCacheEntries() {
        const now = Date.now();
        const maxAge = this.cacheTimeout * 2; // Clean entries older than 2x cache timeout
        
        // Clean prediction cache
        for (const [key, value] of this.predictionCache.entries()) {
            if (now - value.timestamp > maxAge) {
                this.predictionCache.delete(key);
            }
        }
        
        // Clean feature cache
        for (const [key, value] of this.featureCache.entries()) {
            if (now - value.timestamp > maxAge) {
                this.featureCache.delete(key);
            }
        }
        
        // Clean model status cache
        for (const [key, value] of this.modelStatusCache.entries()) {
            if (now - value.timestamp > maxAge) {
                this.modelStatusCache.delete(key);
            }
        }
    }
    
    // Performance monitoring helpers
    getCacheHitRate() {
        return {
            predictions: this.predictionCache.size,
            features: this.featureCache.size,
            modelStatus: this.modelStatusCache.size
        };
    }
    
    getAverageResponseTime() {
        return 'Not implemented';
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
        const times = Array.from(this.predictionCache.values()).map(p => p.timestamp);
        return times.length > 0 ? Math.max(...times) : null;
    }
    
    async start() {
        try {
            Logger.info('Starting Optimized ML Server with Training Queue Management...');
            
            // Don't wait for core service in quick mode - start immediately
            if (!this.quickMode) {
                await this.dataClient.waitForCoreService();
            } else {
                Logger.info('Quick mode enabled - starting without waiting for core service');
            }
            
            // Start HTTP server
            this.server = this.app.listen(this.port, () => {
                Logger.info(`Optimized ML Server running at http://localhost:${this.port}`);
                console.log(`🚀 Optimized ML API available at: http://localhost:${this.port}/api`);
                console.log(`⚡ Health check: http://localhost:${this.port}/api/health`);
                console.log(`🎯 Fast predictions: http://localhost:${this.port}/api/predictions/BTC`);
                console.log(`📊 Performance stats: http://localhost:${this.port}/api/storage/stats`);
                console.log(`🔄 Training queue: http://localhost:${this.port}/api/training/queue`);
                console.log('');
                console.log('⚡ Performance Optimizations Active:');
                console.log(`   • Quick Mode: ${this.quickMode ? 'ENABLED' : 'DISABLED'}`);
                console.log(`   • Cache Timeout: ${this.cacheTimeout}ms`);
                console.log(`   • Enabled Models: ${this.enabledModels.join(', ')}`);
                console.log(`   • Max Concurrent Training: ${this.trainingQueue?.maxConcurrentTraining || 'Not Ready'}`);
                console.log(`   • Training Cooldown: ${this.trainingQueue?.trainingCooldown ? (this.trainingQueue.trainingCooldown / 1000 / 60) + ' minutes' : 'Not Ready'}`);
                console.log(`   • Aggressive Caching: ENABLED`);
                console.log(`   • Background Processing: ENABLED`);
                console.log(`   • Training Queue: ${this.trainingQueue ? 'ACTIVE' : 'INITIALIZING'}`);
            });
            
        } catch (error) {
            Logger.error('Failed to start Optimized ML server', { error: error.message });
            process.exit(1);
        }
    }
    
    async stop() {
        Logger.info('Stopping Optimized ML Server with Training Queue...');
        
        // Stop training queue first
        if (this.trainingQueue) {
            await this.trainingQueue.shutdown();
        }
        
        // Clear all caches
        this.predictionCache.clear();
        this.featureCache.clear();
        this.modelStatusCache.clear();
        
        // Shutdown storage system gracefully
        if (this.mlStorage) {
            await this.mlStorage.shutdown();
        }
        
        // Dispose of all models
        Object.values(this.models).forEach(pairModels => {
            Object.values(pairModels).forEach(model => {
                if (model && typeof model.dispose === 'function') {
                    model.dispose();
                }
            });
        });
        
        if (this.preprocessor) {
            this.preprocessor.dispose();
        }
        
        if (this.server) {
            this.server.close();
        }
        
        Logger.info('Optimized ML Server stopped');
    }
}

module.exports = MLServer;
                