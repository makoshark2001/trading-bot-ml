const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-backend-cpu'); // Add CPU backend
const { Logger } = require('../utils');

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
        
        // Set TensorFlow backend
        this.initializeTensorFlow();
        
        Logger.info('LSTMModel initialized', {
            sequenceLength: this.sequenceLength,
            features: this.features,
            units: this.units,
            layers: this.layers
        });
    }
    
    async initializeTensorFlow() {
        try {
            // Set platform to use CPU backend
            await tf.ready();
            Logger.info('TensorFlow.js initialized', {
                backend: tf.getBackend(),
                version: tf.version.tfjs
            });
        } catch (error) {
            Logger.error('Failed to initialize TensorFlow.js', { error: error.message });
        }
    }
    
    buildModel() {
        try {
            Logger.info('Building LSTM model...');
            
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
                layers: this.model.layers.length
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
        
        try {
            this.isTraining = true;
            
            const epochs = config.epochs || 50;
            const batchSize = config.batchSize || 32;
            const verbose = config.verbose !== undefined ? config.verbose : 1;
            
            Logger.info('Starting LSTM model training', {
                epochs,
                batchSize,
                trainSamples: trainX.shape[0],
                validationSamples: validationX ? validationX.shape[0] : 0
            });
            
            const callbacks = {
                onEpochEnd: (epoch, logs) => {
                    if (epoch % 10 === 0 || epoch === epochs - 1) {
                        Logger.info(`Epoch ${epoch + 1}/${epochs}`, {
                            loss: logs.loss.toFixed(4),
                            accuracy: logs.acc.toFixed(4),
                            valLoss: logs.val_loss?.toFixed(4),
                            valAccuracy: logs.val_acc?.toFixed(4)
                        });
                    }
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
            
            Logger.info('LSTM model training completed', {
                finalLoss: history.history.loss[history.history.loss.length - 1].toFixed(4),
                finalAccuracy: history.history.acc[history.history.acc.length - 1].toFixed(4)
            });
            
            return history;
            
        } catch (error) {
            this.isTraining = false;
            Logger.error('LSTM model training failed', { error: error.message });
            throw error;
        }
    }
    
    async predict(inputX) {
        if (!this.model) {
            throw new Error('Model must be built and trained before prediction');
        }
        
        try {
            Logger.debug('Making LSTM prediction', {
                inputShape: inputX.shape
            });
            
            const prediction = this.model.predict(inputX);
            const result = await prediction.data();
            
            // Clean up tensors
            prediction.dispose();
            
            return result;
            
        } catch (error) {
            Logger.error('LSTM prediction failed', { error: error.message });
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
                accuracy: accuracy[0]
            };
            
            Logger.info('LSTM model evaluation completed', results);
            
            return results;
            
        } catch (error) {
            Logger.error('LSTM model evaluation failed', { error: error.message });
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
            
            this.model = await tf.loadLayersModel(`file://${modelPath}/model.json`);
            this.isCompiled = true;
            
            Logger.info('LSTM model loaded successfully', {
                totalParams: this.model.countParams(),
                layers: this.model.layers.length
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
        
        return {
            layers: this.model.layers.length,
            totalParams: this.model.countParams(),
            trainableParams: this.model.countParams(),
            inputShape: this.model.inputShape,
            outputShape: this.model.outputShape,
            isCompiled: this.isCompiled,
            isTraining: this.isTraining
        };
    }
    
    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
            this.isCompiled = false;
            Logger.info('LSTM model disposed');
        }
    }
}

module.exports = LSTMModel;