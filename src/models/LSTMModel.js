const tf = require('@tensorflow/tfjs');
const { Logger, GPUManager } = require('../utils');

class LSTMModel {
    constructor(config = {}) {
        this.sequenceLength = config.sequenceLength || 60;
        this.features = config.features || 50; // Number of features per timestep
        this.units = config.units || 50;
        this.layers = config.layers || 2;
        this.dropout = config.dropout || 0.2;
        this.learningRate = config.learningRate || 0.001;
        
        this.model = null;
        this.isCompiled = false;
        this.isTraining = false;
        
        // GPU Manager for acceleration
        this.gpuManager = new GPUManager();
        this.gpuInitialized = false;
        
        Logger.info('LSTMModel initialized with GPU support', {
            sequenceLength: this.sequenceLength,
            features: this.features,
            units: this.units,
            layers: this.layers
        });
    }
    
    async initializeTensorFlow() {
        try {
            // Initialize GPU manager first
            if (!this.gpuInitialized) {
                await this.gpuManager.initialize();
                this.gpuInitialized = true;
            }
            
            // Ensure TensorFlow is ready with the selected backend
            await tf.ready();
            
            const gpuStatus = this.gpuManager.getStatus();
            Logger.info('TensorFlow.js initialized for LSTM with GPU support', {
                backend: tf.getBackend(),
                gpuAvailable: gpuStatus.gpuAvailable,
                version: tf.version.tfjs
            });
            
        } catch (error) {
            Logger.error('Failed to initialize TensorFlow.js for LSTM', { error: error.message });
            // Try to fallback to CPU
            try {
                await this.gpuManager.switchToCPU();
                await tf.ready();
                Logger.warn('LSTM initialized with CPU fallback after GPU failure');
            } catch (fallbackError) {
                Logger.error('Complete TensorFlow initialization failure', { 
                    error: fallbackError.message 
                });
                throw fallbackError;
            }
        }
    }
    
    buildModel() {
        try {
            Logger.info('Building LSTM model with GPU acceleration support...');
            
            this.model = tf.sequential();
            
            // First LSTM layer
            this.model.add(tf.layers.lstm({
                units: this.units,
                returnSequences: this.layers > 1,
                inputShape: [this.sequenceLength, this.features],
                dropout: this.dropout,
                recurrentDropout: this.dropout
            }));
            
            // Additional LSTM layers
            for (let i = 1; i < this.layers; i++) {
                this.model.add(tf.layers.lstm({
                    units: this.units,
                    returnSequences: i < this.layers - 1,
                    dropout: this.dropout,
                    recurrentDropout: this.dropout
                }));
            }
            
            // Dense layers for output
            this.model.add(tf.layers.dense({
                units: 32,
                activation: 'relu'
            }));
            
            this.model.add(tf.layers.dropout({
                rate: this.dropout
            }));
            
            // Output layer - predicting price direction (0 or 1)
            this.model.add(tf.layers.dense({
                units: 1,
                activation: 'sigmoid'
            }));
            
            Logger.info('LSTM model built successfully', {
                totalParams: this.model.countParams(),
                layers: this.model.layers.length,
                backend: tf.getBackend()
            });
            
            return this.model;
            
        } catch (error) {
            Logger.error('Failed to build LSTM model', { error: error.message });
            throw error;
        }
    }
    
    compileModel() {
        if (!this.model) {
            throw new Error('Model must be built before compilation');
        }
        
        try {
            Logger.info('Compiling LSTM model...');
            
            this.model.compile({
                optimizer: tf.train.adam(this.learningRate),
                loss: 'binaryCrossentropy',
                metrics: ['accuracy']
            });
            
            this.isCompiled = true;
            Logger.info('LSTM model compiled successfully');
            
        } catch (error) {
            Logger.error('Failed to compile LSTM model', { error: error.message });
            throw error;
        }
    }
    
    async train(trainX, trainY, validationX, validationY, config = {}) {
        if (!this.isCompiled) {
            throw new Error('Model must be compiled before training');
        }
        
        // Ensure TensorFlow is initialized
        await this.initializeTensorFlow();
        
        // Use GPU manager for training with automatic fallback
        return await this.gpuManager.performWithGPUFallback(async () => {
            return await this.performTraining(trainX, trainY, validationX, validationY, config);
        }, 'LSTM training');
    }
    
    async performTraining(trainX, trainY, validationX, validationY, config = {}) {
        try {
            this.isTraining = true;
            
            const epochs = config.epochs || 50;
            const batchSize = config.batchSize || 32;
            const verbose = config.verbose !== undefined ? config.verbose : 1;
            
            const currentBackend = tf.getBackend();
            const gpuStatus = this.gpuManager.getStatus();
            
            Logger.info('Starting LSTM model training with GPU acceleration', {
                epochs,
                batchSize,
                trainSamples: trainX.shape[0],
                validationSamples: validationX ? validationX.shape[0] : 0,
                backend: currentBackend,
                gpuAvailable: gpuStatus.gpuAvailable,
                gpuActive: currentBackend !== 'cpu'
            });
            
            const callbacks = {
                onEpochEnd: (epoch, logs) => {
                    if (epoch % 10 === 0 || epoch === epochs - 1) {
                        Logger.info(`LSTM Epoch ${epoch + 1}/${epochs} [${currentBackend.toUpperCase()}]`, {
                            loss: logs.loss.toFixed(4),
                            accuracy: logs.acc.toFixed(4),
                            valLoss: logs.val_loss?.toFixed(4),
                            valAccuracy: logs.val_acc?.toFixed(4),
                            backend: currentBackend,
                            memoryUsage: tf.memory().numBytes
                        });
                    }
                },
                onTrainEnd: () => {
                    // Clean up metrics periodically
                    this.gpuManager.cleanupMetrics();
                }
            };
            
            const history = await this.model.fit(trainX, trainY, {
                epochs,
                batchSize,
                validationData: validationX && validationY ? [validationX, validationY] : null,
                shuffle: true,
                verbose,
                callbacks
            });
            
            this.isTraining = false;
            
            const finalMetrics = {
                finalLoss: history.history.loss[history.history.loss.length - 1].toFixed(4),
                finalAccuracy: history.history.acc[history.history.acc.length - 1].toFixed(4),
                backend: currentBackend,
                gpuAccelerated: currentBackend !== 'cpu'
            };
            
            if (validationX && validationY) {
                finalMetrics.finalValLoss = history.history.val_loss[history.history.val_loss.length - 1].toFixed(4);
                finalMetrics.finalValAccuracy = history.history.val_acc[history.history.val_acc.length - 1].toFixed(4);
            }
            
            Logger.info('LSTM model training completed', finalMetrics);
            
            // Log performance comparison if available
            const perfComparison = this.gpuManager.getPerformanceComparison();
            if (perfComparison) {
                Logger.info('GPU vs CPU Performance Comparison', perfComparison);
            }
            
            return history;
            
        } catch (error) {
            this.isTraining = false;
            Logger.error('LSTM model training failed', { 
                error: error.message,
                backend: tf.getBackend()
            });
            throw error;
        }
    }
    
    async predict(inputX) {
        if (!this.model) {
            throw new Error('Model must be built and trained before prediction');
        }
        
        try {
            Logger.debug('Making LSTM prediction', {
                inputShape: inputX.shape,
                backend: tf.getBackend()
            });
            
            const prediction = this.model.predict(inputX);
            const result = await prediction.data();
            
            // Clean up tensors
            prediction.dispose();
            
            return result;
            
        } catch (error) {
            Logger.error('LSTM prediction failed', { 
                error: error.message,
                backend: tf.getBackend()
            });
            throw error;
        }
    }
    
    async evaluate(testX, testY) {
        if (!this.model) {
            throw new Error('Model must be built and trained before evaluation');
        }
        
        try {
            Logger.info('Evaluating LSTM model');
            
            const evaluation = this.model.evaluate(testX, testY);
            const [loss, accuracy] = await Promise.all([
                evaluation[0].data(),
                evaluation[1].data()
            ]);
            
            // Clean up tensors
            evaluation[0].dispose();
            evaluation[1].dispose();
            
            const results = {
                loss: loss[0],
                accuracy: accuracy[0],
                backend: tf.getBackend()
            };
            
            Logger.info('LSTM model evaluation completed', results);
            
            return results;
            
        } catch (error) {
            Logger.error('LSTM model evaluation failed', { 
                error: error.message,
                backend: tf.getBackend()
            });
            throw error;
        }
    }
    
    async save(modelPath) {
        if (!this.model) {
            throw new Error('Model must be built before saving');
        }
        
        try {
            Logger.info('Saving LSTM model', { path: modelPath });
            
            await this.model.save(`file://${modelPath}`);
            
            Logger.info('LSTM model saved successfully');
            
        } catch (error) {
            Logger.error('Failed to save LSTM model', { error: error.message });
            throw error;
        }
    }
    
    async load(modelPath) {
        try {
            Logger.info('Loading LSTM model', { path: modelPath });
            
            // Ensure TensorFlow is initialized before loading
            await this.initializeTensorFlow();
            
            this.model = await tf.loadLayersModel(`file://${modelPath}/model.json`);
            this.isCompiled = true;
            
            Logger.info('LSTM model loaded successfully', {
                totalParams: this.model.countParams(),
                layers: this.model.layers.length,
                backend: tf.getBackend()
            });
            
        } catch (error) {
            Logger.error('Failed to load LSTM model', { error: error.message });
            throw error;
        }
    }
    
    getModelSummary() {
        if (!this.model) {
            return 'Model not built yet';
        }
        
        const gpuStatus = this.gpuManager ? this.gpuManager.getStatus() : null;
        
        return {
            layers: this.model.layers.length,
            totalParams: this.model.countParams(),
            trainableParams: this.model.countParams(),
            inputShape: this.model.inputShape,
            outputShape: this.model.outputShape,
            isCompiled: this.isCompiled,
            isTraining: this.isTraining,
            backend: tf.getBackend(),
            gpu: {
                available: gpuStatus?.gpuAvailable || false,
                currentBackend: gpuStatus?.currentBackend || 'unknown',
                memoryUsage: tf.memory()
            }
        };
    }
    
    getGPUStatus() {
        return this.gpuManager ? this.gpuManager.getStatus() : {
            gpuAvailable: false,
            currentBackend: 'cpu',
            error: 'GPU manager not initialized'
        };
    }
    
    getPerformanceMetrics() {
        return this.gpuManager ? this.gpuManager.getPerformanceComparison() : null;
    }
    
    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
            this.isCompiled = false;
        }
        
        if (this.gpuManager) {
            this.gpuManager.dispose();
        }
        
        Logger.info('LSTM model disposed');
    }
}

module.exports = LSTMModel;