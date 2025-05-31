const tf = require('@tensorflow/tfjs');
const { Logger } = require('../utils');

// Rest of the file remains the same...

class DataPreprocessor {
    constructor(config = {}) {
        this.sequenceLength = config.sequenceLength || 60;
        this.testSplit = config.testSplit || 0.2;
        this.validationSplit = config.validationSplit || 0.2;
        
        this.scaler = {
            mean: null,
            std: null,
            min: null,
            max: null
        };
        
        Logger.info('DataPreprocessor initialized', {
            sequenceLength: this.sequenceLength,
            testSplit: this.testSplit,
            validationSplit: this.validationSplit
        });
    }
    
    async prepareTrainingData(featuresArray, targets) {
        try {
            Logger.info('Preparing training data', {
                samples: featuresArray.length,
                features: featuresArray[0]?.length || 0
            });
            
            // Normalize features
            const normalizedFeatures = this.normalizeFeatures(featuresArray);
            
            // Create sequences
            const sequences = this.createSequences(normalizedFeatures, targets);
            
            // Split data
            const splits = this.splitData(sequences.X, sequences.y);
            
            Logger.info('Training data prepared', {
                trainSamples: splits.trainX.shape[0],
                validationSamples: splits.validationX.shape[0],
                testSamples: splits.testX.shape[0],
                sequenceLength: splits.trainX.shape[1],
                features: splits.trainX.shape[2]
            });
            
            return splits;
            
        } catch (error) {
            Logger.error('Failed to prepare training data', { error: error.message });
            throw error;
        }
    }
    
    normalizeFeatures(featuresArray) {
        Logger.debug('Normalizing features');
        
        // Convert to tensor for easier computation
        const featuresTensor = tf.tensor2d(featuresArray);
        
        // Calculate statistics
        const mean = featuresTensor.mean(0);
        const std = featuresTensor.sub(mean).square().mean(0).sqrt();
        
        // Store for later use in prediction
        this.scaler.mean = mean;
        this.scaler.std = std;
        
        // Normalize: (x - mean) / std
        const normalizedTensor = featuresTensor.sub(mean).div(std.add(1e-7)); // Add small epsilon to avoid division by zero
        
        // Convert back to array
        const normalizedArray = normalizedTensor.arraySync();
        
        // Clean up tensors
        featuresTensor.dispose();
        normalizedTensor.dispose();
        
        Logger.debug('Feature normalization completed');
        
        return normalizedArray;
    }
    
    createSequences(features, targets) {
        Logger.debug('Creating sequences for LSTM');
        
        const sequencesX = [];
        const sequencesY = [];
        
        // Create overlapping sequences
        for (let i = 0; i < features.length - this.sequenceLength; i++) {
            const sequence = features.slice(i, i + this.sequenceLength);
            const target = targets[i + this.sequenceLength - 1]; // Predict next value
            
            sequencesX.push(sequence);
            sequencesY.push(target);
        }
        
        Logger.debug('Sequences created', {
            sequences: sequencesX.length,
            sequenceLength: this.sequenceLength,
            features: sequencesX[0][0].length
        });
        
        return {
            X: tf.tensor3d(sequencesX),
            y: tf.tensor1d(sequencesY)
        };
    }
    
    splitData(X, y) {
        Logger.debug('Splitting data into train/validation/test sets');
        
        const totalSamples = X.shape[0];
        const testSize = Math.floor(totalSamples * this.testSplit);
        const validationSize = Math.floor((totalSamples - testSize) * this.validationSplit);
        const trainSize = totalSamples - testSize - validationSize;
        
        // Split the data
        const trainX = X.slice([0, 0, 0], [trainSize, -1, -1]);
        const trainY = y.slice([0], [trainSize]);
        
        const validationX = X.slice([trainSize, 0, 0], [validationSize, -1, -1]);
        const validationY = y.slice([trainSize], [validationSize]);
        
        const testX = X.slice([trainSize + validationSize, 0, 0], [testSize, -1, -1]);
        const testY = y.slice([trainSize + validationSize], [testSize]);
        
        Logger.debug('Data split completed', {
            train: trainSize,
            validation: validationSize,
            test: testSize
        });
        
        return {
            trainX,
            trainY,
            validationX,
            validationY,
            testX,
            testY
        };
    }
    
    async prepareRealTimeData(featuresArray) {
        if (!this.scaler.mean || !this.scaler.std) {
            throw new Error('Scaler not fitted. Must call prepareTrainingData first.');
        }
        
        try {
            Logger.debug('Preparing real-time data for prediction');
            
            // Take the last sequenceLength samples
            const recentFeatures = featuresArray.slice(-this.sequenceLength);
            
            if (recentFeatures.length < this.sequenceLength) {
                throw new Error(`Insufficient data. Need ${this.sequenceLength} samples, got ${recentFeatures.length}`);
            }
            
            // Normalize using stored statistics
            const featuresTensor = tf.tensor2d(recentFeatures);
            const normalizedTensor = featuresTensor.sub(this.scaler.mean).div(this.scaler.std.add(1e-7));
            
            // Reshape for LSTM input [1, sequenceLength, features]
            const inputTensor = normalizedTensor.expandDims(0);
            
            // Clean up intermediate tensors
            featuresTensor.dispose();
            normalizedTensor.dispose();
            
            return inputTensor;
            
        } catch (error) {
            Logger.error('Failed to prepare real-time data', { error: error.message });
            throw error;
        }
    }
    
    getScalerStats() {
        if (!this.scaler.mean || !this.scaler.std) {
            return null;
        }
        
        return {
            mean: this.scaler.mean.arraySync(),
            std: this.scaler.std.arraySync(),
            hasScaler: true
        };
    }
    
    dispose() {
        if (this.scaler.mean) {
            this.scaler.mean.dispose();
            this.scaler.mean = null;
        }
        if (this.scaler.std) {
            this.scaler.std.dispose();
            this.scaler.std = null;
        }
        Logger.debug('DataPreprocessor disposed');
    }
}

module.exports = DataPreprocessor;