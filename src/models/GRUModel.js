const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-backend-cpu');
const { Logger } = require('../utils');

class GRUModel {
    constructor(config = {}) {
        this.sequenceLength = config.sequenceLength || 60;
        this.features = config.features || 50;
        this.units = config.units || 50;
        this.layers = config.layers || 2;
        this.dropout = config.dropout || 0.2;
        this.learningRate = config.learningRate || 0.001;
        this.recurrentDropout = config.recurrentDropout || 0.2;
        
        this.model = null;
        this.isCompiled = false;
        this.isTraining = false;
        
        this.initializeTensorFlow();
        
        Logger.info('GRUModel initialized', {
            sequenceLength: this.sequenceLength,
            features: this.features,
            units: this.units,
            layers: this.layers
        });
    }
    
    async initializeTensorFlow() {
        try {
            await tf.ready();
            Logger.info('TensorFlow.js initialized for GRU', {
                backend: tf.getBackend(),
                version: tf.version.tfjs
            });
        } catch (error) {
            Logger.error('Failed to initialize TensorFlow.js for GRU', { error: error.message });
        }
    }
    
    buildModel() {
        try {
            Logger.info('Building GRU model...');
            
            this.model = tf.sequential();
            
            // First GRU layer
            this.model.add(tf.layers.gru({
                units: this.units,
                returnSequences: this.layers > 1,
                inputShape: [this.sequenceLength, this.features],
                dropout: this.dropout,
                recurrentDropout: this.recurrentDropout,
                kernelInitializer: 'glorotUniform',
                recurrentInitializer: 'orthogonal'
            }));
            
            // Additional GRU layers
            for (let i = 1; i < this.layers; i++) {
                this.model.add(tf.layers.gru({
                    units: this.units,
                    returnSequences: i < this.layers - 1,
                    dropout: this.dropout,
                    recurrentDropout: this.recurrentDropout,
                    kernelInitializer: 'glorotUniform',
                    recurrentInitializer: 'orthogonal'
                }));
            }
            
            // Batch normalization for better training stability
            this.model.add(tf.layers.batchNormalization());
            
            // Dense layers for feature transformation
            this.model.add(tf.layers.dense({
                units: 32,
                activation: 'relu',
                kernelInitializer: 'heNormal'
            }));
            
            this.model.add(tf.layers.dropout({
                rate: this.dropout
            }));
            
            this.model.add(tf.layers.dense({
                units: 16,
                activation: 'relu',
                kernelInitializer: 'heNormal'
            }));
            
            this.model.add(tf.layers.dropout({
                rate: this.dropout / 2
            }));
            
            // Output layer - predicting price direction (0 or 1)
            this.model.add(tf.layers.dense({
                units: 1,
                activation: 'sigmoid',
                kernelInitializer: 'glorotUniform'
            }));
            
            Logger.info('GRU model built successfully', {
                totalParams: this.model.countParams(),
                layers: this.model.layers.length,
                outputShape: this.model.outputShape
            });
            
            return this.model;
            
        } catch (error) {
            Logger.error('Failed to build GRU model', { error: error.message });
            throw error;
        }
    }
    
    compileModel() {
        if (!this.model) {
            throw new Error('Model must be built before compilation');
        }
        
        try {
            Logger.info('Compiling GRU model...');
            
            // Use Adam optimizer with custom learning rate and parameters
            const optimizer = tf.train.adam(this.learningRate, 0.9, 0.999, 1e-8);
            
            this.model.compile({
                optimizer: optimizer,
                loss: 'binaryCrossentropy',
                metrics: ['accuracy']
            });
            
            this.isCompiled = true;
            Logger.info('GRU model compiled successfully with enhanced metrics');
            
        } catch (error) {
            Logger.error('Failed to compile GRU model', { error: error.message });
            throw error;
        }
    }
    
    async train(trainX, trainY, validationX, validationY, config = {}) {
        if (!this.isCompiled) {
            throw new Error('Model must be compiled before training');
        }
        
        try {
            this.isTraining = true;
            
            const epochs = config.epochs || 50;
            const batchSize = config.batchSize || 32;
            const verbose = config.verbose !== undefined ? config.verbose : 1;
            const patience = config.patience || 10;
            
            Logger.info('Starting GRU model training', {
                epochs,
                batchSize,
                trainSamples: trainX.shape[0],
                validationSamples: validationX ? validationX.shape[0] : 0,
                patience
            });
            
            // Enhanced callbacks for better training
            const callbacks = {
                onEpochEnd: (epoch, logs) => {
                    if (epoch % 5 === 0 || epoch === epochs - 1) {
                        Logger.info(`GRU Epoch ${epoch + 1}/${epochs}`, {
                            loss: logs.loss.toFixed(4),
                            accuracy: logs.acc.toFixed(4),
                            valLoss: logs.val_loss?.toFixed(4),
                            valAccuracy: logs.val_acc?.toFixed(4)
                        });
                    }
                },
                onBatchEnd: (batch, logs) => {
                    // Optional: Log batch progress for very verbose training
                    if (config.veryVerbose && batch % 10 === 0) {
                        Logger.debug(`GRU Batch ${batch}`, {
                            batchLoss: logs.loss.toFixed(4),
                            batchAcc: logs.acc.toFixed(4)
                        });
                    }
                }
            };
            
            // Early stopping configuration
            const earlyStopping = validationX && validationY ? {
                monitor: 'val_loss',
                patience: patience,
                verbose: 1,
                restoreBestWeights: true
            } : null;
            
            const trainingConfig = {
                epochs,
                batchSize,
                validationData: validationX && validationY ? [validationX, validationY] : null,
                shuffle: true,
                verbose,
                callbacks: earlyStopping ? [callbacks, earlyStopping] : callbacks
            };
            
            const history = await this.model.fit(trainX, trainY, trainingConfig);
            
            this.isTraining = false;
            
            const finalMetrics = {
                finalLoss: history.history.loss[history.history.loss.length - 1].toFixed(4),
                finalAccuracy: history.history.acc[history.history.acc.length - 1].toFixed(4),
                epochsCompleted: history.epoch.length
            };
            
            if (validationX && validationY) {
                finalMetrics.finalValLoss = history.history.val_loss[history.history.val_loss.length - 1].toFixed(4);
                finalMetrics.finalValAccuracy = history.history.val_acc[history.history.val_acc.length - 1].toFixed(4);
            }
            
            Logger.info('GRU model training completed', finalMetrics);
            
            return {
                ...history,
                finalMetrics,
                modelType: 'GRU'
            };
            
        } catch (error) {
            this.isTraining = false;
            Logger.error('GRU model training failed', { error: error.message });
            throw error;
        }
    }
    
    async predict(inputX) {
        if (!this.model) {
            throw new Error('Model must be built and trained before prediction');
        }
        
        try {
            Logger.debug('Making GRU prediction', {
                inputShape: inputX.shape,
                modelType: 'GRU'
            });
            
            const prediction = this.model.predict(inputX);
            const result = await prediction.data();
            
            // Clean up tensors
            prediction.dispose();
            
            Logger.debug('GRU prediction completed', {
                predictions: result.length,
                samplePrediction: result[0]?.toFixed(4)
            });
            
            return result;
            
        } catch (error) {
            Logger.error('GRU prediction failed', { error: error.message });
            throw error;
        }
    }
    
    async evaluate(testX, testY) {
        if (!this.model) {
            throw new Error('Model must be built and trained before evaluation');
        }
        
        try {
            Logger.info('Evaluating GRU model');
            
            const evaluation = this.model.evaluate(testX, testY, { verbose: 0 });
            
            let results;
            if (Array.isArray(evaluation)) {
                // Multiple metrics
                const [loss, accuracy] = await Promise.all(
                    evaluation.map(tensor => tensor.data())
                );
                
                results = {
                    loss: loss[0],
                    accuracy: accuracy[0]
                };
                
                // Clean up tensors
                evaluation.forEach(tensor => tensor.dispose());
            } else {
                // Single metric (loss only)
                const lossData = await evaluation.data();
                results = { loss: lossData[0] };
                evaluation.dispose();
            }
            
            Logger.info('GRU model evaluation completed', {
                ...results,
                modelType: 'GRU'
            });
            
            return results;
            
        } catch (error) {
            Logger.error('GRU model evaluation failed', { error: error.message });
            throw error;
        }
    }
    
    async save(modelPath) {
        if (!this.model) {
            throw new Error('Model must be built before saving');
        }
        
        try {
            Logger.info('Saving GRU model', { path: modelPath });
            
            await this.model.save(`file://${modelPath}`);
            
            // Save additional GRU-specific metadata
            const metadata = {
                modelType: 'GRU',
                config: {
                    sequenceLength: this.sequenceLength,
                    features: this.features,
                    units: this.units,
                    layers: this.layers,
                    dropout: this.dropout,
                    recurrentDropout: this.recurrentDropout,
                    learningRate: this.learningRate
                },
                architecture: this.getModelSummary(),
                savedAt: Date.now(),
                version: '1.0.0'
            };
            
            // Save metadata alongside model
            const fs = require('fs');
            const path = require('path');
            const metadataPath = path.join(modelPath, 'gru_metadata.json');
            fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));
            
            Logger.info('GRU model and metadata saved successfully');
            
        } catch (error) {
            Logger.error('Failed to save GRU model', { error: error.message });
            throw error;
        }
    }
    
    async load(modelPath) {
        try {
            Logger.info('Loading GRU model', { path: modelPath });
            
            this.model = await tf.loadLayersModel(`file://${modelPath}/model.json`);
            this.isCompiled = true;
            
            // Load metadata if available
            try {
                const fs = require('fs');
                const path = require('path');
                const metadataPath = path.join(modelPath, 'gru_metadata.json');
                
                if (fs.existsSync(metadataPath)) {
                    const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
                    
                    // Restore configuration from metadata
                    this.sequenceLength = metadata.config.sequenceLength;
                    this.features = metadata.config.features;
                    this.units = metadata.config.units;
                    this.layers = metadata.config.layers;
                    this.dropout = metadata.config.dropout;
                    this.recurrentDropout = metadata.config.recurrentDropout;
                    this.learningRate = metadata.config.learningRate;
                    
                    Logger.info('GRU metadata loaded', metadata.config);
                }
            } catch (metadataError) {
                Logger.warn('Failed to load GRU metadata', { error: metadataError.message });
            }
            
            Logger.info('GRU model loaded successfully', {
                totalParams: this.model.countParams(),
                layers: this.model.layers.length
            });
            
        } catch (error) {
            Logger.error('Failed to load GRU model', { error: error.message });
            throw error;
        }
    }
    
    getModelSummary() {
        if (!this.model) {
            return 'GRU model not built yet';
        }
        
        return {
            modelType: 'GRU',
            layers: this.model.layers.length,
            totalParams: this.model.countParams(),
            trainableParams: this.model.countParams(),
            inputShape: this.model.inputShape,
            outputShape: this.model.outputShape,
            isCompiled: this.isCompiled,
            isTraining: this.isTraining,
            config: {
                sequenceLength: this.sequenceLength,
                features: this.features,
                units: this.units,
                layers: this.layers,
                dropout: this.dropout,
                recurrentDropout: this.recurrentDropout,
                learningRate: this.learningRate
            },
            architecture: this.model.layers.map(layer => ({
                name: layer.name,
                type: layer.getClassName(),
                inputShape: layer.inputShape,
                outputShape: layer.outputShape,
                params: layer.countParams()
            }))
        };
    }
    
    // Get model predictions with additional GRU-specific analysis
    async predictWithAnalysis(inputX) {
        const predictions = await this.predict(inputX);
        
        // Convert predictions to analysis format
        const analysis = predictions.map((pred, index) => ({
            prediction: pred,
            confidence: Math.abs(pred - 0.5) * 2,
            direction: pred > 0.5 ? 'up' : 'down',
            strength: this.getStrengthLevel(pred),
            modelType: 'GRU',
            timestamp: Date.now(),
            index: index
        }));
        
        return analysis;
    }
    
    getStrengthLevel(prediction) {
        const distance = Math.abs(prediction - 0.5);
        if (distance > 0.4) return 'very_strong';
        if (distance > 0.3) return 'strong';
        if (distance > 0.2) return 'moderate';
        if (distance > 0.1) return 'weak';
        return 'very_weak';
    }
    
    // Memory management and cleanup
    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
            this.isCompiled = false;
            Logger.info('GRU model disposed');
        }
    }
    
    // Get current memory usage
    getMemoryUsage() {
        return {
            numTensors: tf.memory().numTensors,
            numBytes: tf.memory().numBytes,
            modelLoaded: !!this.model,
            isTraining: this.isTraining
        };
    }
    
    // Validate input data for GRU model
    validateInput(inputX) {
        if (!inputX || !inputX.shape) {
            throw new Error('Invalid input: must be a tensor');
        }
        
        const expectedShape = [null, this.sequenceLength, this.features];
        const actualShape = inputX.shape;
        
        if (actualShape.length !== 3) {
            throw new Error(`Invalid input shape: expected 3D tensor, got ${actualShape.length}D`);
        }
        
        if (actualShape[1] !== this.sequenceLength) {
            throw new Error(`Invalid sequence length: expected ${this.sequenceLength}, got ${actualShape[1]}`);
        }
        
        if (actualShape[2] !== this.features) {
            throw new Error(`Invalid feature count: expected ${this.features}, got ${actualShape[2]}`);
        }
        
        return true;
    }
}

module.exports = GRUModel;