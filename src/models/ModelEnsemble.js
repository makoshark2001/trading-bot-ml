const tf = require('@tensorflow/tfjs');
const { Logger } = require('../utils');

class ModelEnsemble {
    constructor(config = {}) {
        this.models = new Map(); // Store different model types
        this.weights = config.weights || {}; // Model weights for ensemble
        this.modelTypes = config.modelTypes || ['lstm', 'gru', 'cnn', 'transformer'];
        this.votingStrategy = config.votingStrategy || 'weighted'; // 'weighted', 'majority', 'average'
        this.performanceHistory = new Map(); // Track model performance
        this.isInitialized = false;
        
        Logger.info('ModelEnsemble initialized', {
            modelTypes: this.modelTypes,
            votingStrategy: this.votingStrategy
        });
    }
    
    // Add a model to the ensemble
    addModel(modelType, model, weight = 1.0, metadata = {}) {
        this.models.set(modelType, {
            model: model,
            weight: weight,
            metadata: {
                ...metadata,
                addedAt: Date.now(),
                predictions: 0,
                totalConfidence: 0,
                accuracy: 0
            }
        });
        
        this.weights[modelType] = weight;
        
        Logger.info(`Model added to ensemble: ${modelType}`, {
            weight: weight,
            totalModels: this.models.size
        });
    }
    
    // Remove a model from the ensemble
    removeModel(modelType) {
        if (this.models.has(modelType)) {
            const modelInfo = this.models.get(modelType);
            if (modelInfo.model && typeof modelInfo.model.dispose === 'function') {
                modelInfo.model.dispose();
            }
            this.models.delete(modelType);
            delete this.weights[modelType];
            
            Logger.info(`Model removed from ensemble: ${modelType}`);
        }
    }
    
    // Update model weights based on performance
    updateWeights(performanceMetrics) {
        const totalPerformance = Object.values(performanceMetrics).reduce((sum, perf) => sum + perf, 0);
        
        if (totalPerformance > 0) {
            Object.keys(performanceMetrics).forEach(modelType => {
                if (this.models.has(modelType)) {
                    const normalizedWeight = performanceMetrics[modelType] / totalPerformance;
                    this.weights[modelType] = normalizedWeight;
                    
                    const modelInfo = this.models.get(modelType);
                    modelInfo.weight = normalizedWeight;
                    this.models.set(modelType, modelInfo);
                }
            });
            
            Logger.info('Model weights updated based on performance', this.weights);
        }
    }
    
    // Make ensemble prediction
    async predict(inputX, options = {}) {
        if (this.models.size === 0) {
            throw new Error('No models in ensemble');
        }
        
        const predictions = new Map();
        const confidences = new Map();
        const errors = new Map();
        
        // Get predictions from all models
        for (const [modelType, modelInfo] of this.models.entries()) {
            try {
                const startTime = Date.now();
                const prediction = await modelInfo.model.predict(inputX);
                const predictionTime = Date.now() - startTime;
                
                // Convert tensor to array if needed
                let predictionArray;
                if (prediction.data) {
                    predictionArray = await prediction.data();
                    prediction.dispose();
                } else {
                    predictionArray = Array.isArray(prediction) ? prediction : [prediction];
                }
                
                predictions.set(modelType, predictionArray[0]);
                
                // Calculate confidence based on distance from 0.5
                const confidence = Math.abs(predictionArray[0] - 0.5) * 2;
                confidences.set(modelType, confidence);
                
                // Update model statistics
                modelInfo.metadata.predictions++;
                modelInfo.metadata.totalConfidence += confidence;
                modelInfo.metadata.lastPrediction = Date.now();
                modelInfo.metadata.lastPredictionTime = predictionTime;
                
                Logger.debug(`${modelType} prediction: ${predictionArray[0].toFixed(4)}, confidence: ${confidence.toFixed(4)}`);
                
            } catch (error) {
                Logger.error(`Prediction failed for ${modelType}`, { error: error.message });
                errors.set(modelType, error.message);
            }
        }
        
        if (predictions.size === 0) {
            throw new Error('All models failed to make predictions');
        }
        
        // Combine predictions using voting strategy
        const ensemblePrediction = this.combinePredicti

ons(predictions, confidences, options);
        
        // Update performance tracking
        this.updatePerformanceTracking(predictions, confidences, ensemblePrediction);
        
        return {
            prediction: ensemblePrediction.value,
            confidence: ensemblePrediction.confidence,
            direction: ensemblePrediction.value > 0.5 ? 'up' : 'down',
            signal: this.getTradeSignal(ensemblePrediction.value, ensemblePrediction.confidence),
            ensemble: {
                strategy: this.votingStrategy,
                modelCount: predictions.size,
                individualPredictions: Object.fromEntries(predictions),
                individualConfidences: Object.fromEntries(confidences),
                weights: this.getActiveWeights(Array.from(predictions.keys())),
                errors: errors.size > 0 ? Object.fromEntries(errors) : undefined
            },
            metadata: {
                timestamp: Date.now(),
                version: '1.0.0',
                type: 'ensemble_prediction'
            }
        };
    }
    
    // Combine predictions using different strategies
    combinePredicti

ons(predictions, confidences, options = {}) {
        const strategy = options.strategy || this.votingStrategy;
        
        switch (strategy) {
            case 'weighted':
                return this.weightedVoting(predictions, confidences);
            case 'majority':
                return this.majorityVoting(predictions, confidences);
            case 'average':
                return this.averageVoting(predictions, confidences);
            case 'confidence_weighted':
                return this.confidenceWeightedVoting(predictions, confidences);
            default:
                return this.weightedVoting(predictions, confidences);
        }
    }
    
    // Weighted voting based on model weights
    weightedVoting(predictions, confidences) {
        let weightedSum = 0;
        let totalWeight = 0;
        let weightedConfidence = 0;
        
        for (const [modelType, prediction] of predictions.entries()) {
            const weight = this.weights[modelType] || 1.0;
            const confidence = confidences.get(modelType) || 0.5;
            
            weightedSum += prediction * weight;
            weightedConfidence += confidence * weight;
            totalWeight += weight;
        }
        
        return {
            value: totalWeight > 0 ? weightedSum / totalWeight : 0.5,
            confidence: totalWeight > 0 ? weightedConfidence / totalWeight : 0.5,
            strategy: 'weighted'
        };
    }
    
    // Majority voting (>0.5 = up, <0.5 = down)
    majorityVoting(predictions, confidences) {
        let upVotes = 0;
        let downVotes = 0;
        let totalConfidence = 0;
        
        for (const [modelType, prediction] of predictions.entries()) {
            if (prediction > 0.5) {
                upVotes++;
            } else {
                downVotes++;
            }
            totalConfidence += confidences.get(modelType) || 0.5;
        }
        
        const totalVotes = upVotes + downVotes;
        const avgConfidence = totalVotes > 0 ? totalConfidence / totalVotes : 0.5;
        
        return {
            value: upVotes > downVotes ? 0.7 : 0.3, // Strong signal for majority
            confidence: avgConfidence * (Math.max(upVotes, downVotes) / totalVotes), // Confidence based on majority strength
            strategy: 'majority',
            votes: { up: upVotes, down: downVotes }
        };
    }
    
    // Simple average of all predictions
    averageVoting(predictions, confidences) {
        let sum = 0;
        let confSum = 0;
        const count = predictions.size;
        
        for (const [modelType, prediction] of predictions.entries()) {
            sum += prediction;
            confSum += confidences.get(modelType) || 0.5;
        }
        
        return {
            value: count > 0 ? sum / count : 0.5,
            confidence: count > 0 ? confSum / count : 0.5,
            strategy: 'average'
        };
    }
    
    // Confidence-weighted voting (higher confidence = more weight)
    confidenceWeightedVoting(predictions, confidences) {
        let weightedSum = 0;
        let totalWeight = 0;
        
        for (const [modelType, prediction] of predictions.entries()) {
            const confidence = confidences.get(modelType) || 0.5;
            
            weightedSum += prediction * confidence;
            totalWeight += confidence;
        }
        
        const finalConfidence = totalWeight / predictions.size; // Average confidence
        
        return {
            value: totalWeight > 0 ? weightedSum / totalWeight : 0.5,
            confidence: finalConfidence,
            strategy: 'confidence_weighted'
        };
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
    
    // Get active model weights
    getActiveWeights(modelTypes) {
        const activeWeights = {};
        modelTypes.forEach(type => {
            activeWeights[type] = this.weights[type] || 1.0;
        });
        return activeWeights;
    }
    
    // Update performance tracking
    updatePerformanceTracking(predictions, confidences, ensemblePrediction) {
        const timestamp = Date.now();
        
        // Store prediction for later accuracy calculation
        if (!this.performanceHistory.has(timestamp)) {
            this.performanceHistory.set(timestamp, {
                individualPredictions: Object.fromEntries(predictions),
                individualConfidences: Object.fromEntries(confidences),
                ensemblePrediction: ensemblePrediction,
                timestamp: timestamp
            });
        }
        
        // Keep only recent performance data (last 1000 predictions)
        if (this.performanceHistory.size > 1000) {
            const oldestKey = Math.min(...this.performanceHistory.keys());
            this.performanceHistory.delete(oldestKey);
        }
    }
    
    // Evaluate actual vs predicted performance (requires actual outcomes)
    evaluatePerformance(actualOutcomes) {
        const performance = {
            ensemble: { correct: 0, total: 0, accuracy: 0 },
            individual: {}
        };
        
        // Initialize individual model performance
        for (const modelType of this.models.keys()) {
            performance.individual[modelType] = { correct: 0, total: 0, accuracy: 0 };
        }
        
        let matchedPredictions = 0;
        
        for (const [timestamp, outcome] of Object.entries(actualOutcomes)) {
            const predictionData = this.performanceHistory.get(parseInt(timestamp));
            
            if (predictionData) {
                matchedPredictions++;
                
                // Evaluate ensemble
                const ensembleCorrect = (predictionData.ensemblePrediction.value > 0.5) === (outcome > 0);
                if (ensembleCorrect) performance.ensemble.correct++;
                performance.ensemble.total++;
                
                // Evaluate individual models
                for (const [modelType, prediction] of Object.entries(predictionData.individualPredictions)) {
                    const modelCorrect = (prediction > 0.5) === (outcome > 0);
                    if (modelCorrect) performance.individual[modelType].correct++;
                    performance.individual[modelType].total++;
                }
            }
        }
        
        // Calculate accuracies
        performance.ensemble.accuracy = performance.ensemble.total > 0 ? 
            performance.ensemble.correct / performance.ensemble.total : 0;
        
        for (const modelType of Object.keys(performance.individual)) {
            const modelPerf = performance.individual[modelType];
            modelPerf.accuracy = modelPerf.total > 0 ? modelPerf.correct / modelPerf.total : 0;
        }
        
        Logger.info('Performance evaluation completed', {
            matchedPredictions,
            ensembleAccuracy: performance.ensemble.accuracy.toFixed(4),
            individualAccuracies: Object.fromEntries(
                Object.entries(performance.individual).map(([type, perf]) => [type, perf.accuracy.toFixed(4)])
            )
        });
        
        // Auto-update weights based on performance
        if (matchedPredictions > 10) { // Only update with sufficient data
            const performanceWeights = {};
            for (const [modelType, perf] of Object.entries(performance.individual)) {
                performanceWeights[modelType] = Math.max(0.1, perf.accuracy); // Minimum weight of 0.1
            }
            this.updateWeights(performanceWeights);
        }
        
        return performance;
    }
    
    // Get ensemble statistics
    getEnsembleStats() {
        const stats = {
            modelCount: this.models.size,
            models: {},
            weights: this.weights,
            votingStrategy: this.votingStrategy,
            performanceHistorySize: this.performanceHistory.size,
            isInitialized: this.isInitialized
        };
        
        for (const [modelType, modelInfo] of this.models.entries()) {
            stats.models[modelType] = {
                weight: modelInfo.weight,
                predictions: modelInfo.metadata.predictions,
                avgConfidence: modelInfo.metadata.predictions > 0 ? 
                    modelInfo.metadata.totalConfidence / modelInfo.metadata.predictions : 0,
                lastPrediction: modelInfo.metadata.lastPrediction,
                lastPredictionTime: modelInfo.metadata.lastPredictionTime
            };
        }
        
        return stats;
    }
    
    // Dispose of all models in ensemble
    dispose() {
        for (const [modelType, modelInfo] of this.models.entries()) {
            if (modelInfo.model && typeof modelInfo.model.dispose === 'function') {
                modelInfo.model.dispose();
            }
        }
        
        this.models.clear();
        this.performanceHistory.clear();
        this.weights = {};
        
        Logger.info('ModelEnsemble disposed');
    }
    
    // Save ensemble configuration and performance
    toJSON() {
        return {
            modelTypes: this.modelTypes,
            weights: this.weights,
            votingStrategy: this.votingStrategy,
            stats: this.getEnsembleStats(),
            timestamp: Date.now(),
            version: '1.0.0'
        };
    }
    
    // Load ensemble configuration
    fromJSON(data) {
        this.modelTypes = data.modelTypes || this.modelTypes;
        this.weights = data.weights || {};
        this.votingStrategy = data.votingStrategy || 'weighted';
        
        Logger.info('ModelEnsemble configuration loaded', {
            modelTypes: this.modelTypes,
            votingStrategy: this.votingStrategy
        });
    }
}

module.exports = ModelEnsemble;