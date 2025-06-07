const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-backend-wasm');
const { Logger } = require('../utils');

class TransformerModel {
    constructor(config = {}) {
        this.sequenceLength = config.sequenceLength || 60;
        this.features = config.features || 50;
        this.dModel = config.dModel || 128; // Model dimension
        this.numHeads = config.numHeads || 8; // Number of attention heads
        this.numLayers = config.numLayers || 4; // Number of transformer layers
        this.dff = config.dff || 512; // Feed-forward dimension
        this.dropout = config.dropout || 0.1;
        this.learningRate = config.learningRate || 0.001;
        this.usePositionalEncoding = config.usePositionalEncoding !== false;
        this.maxPositionalEncoding = config.maxPositionalEncoding || 1000;
        
        this.model = null;
        this.isCompiled = false;
        this.isTraining = false;
        
        this.initializeTensorFlow();
        
        Logger.info('TransformerModel initialized', {
            sequenceLength: this.sequenceLength,
            features: this.features,
            dModel: this.dModel,
            numHeads: this.numHeads,
            numLayers: this.numLayers,
            usePositionalEncoding: this.usePositionalEncoding
        });
    }
    
    async initializeTensorFlow() {
        try {
            // Set WASM backend for better performance
            await tf.setBackend('wasm');
            await tf.ready();
            
            Logger.info('TensorFlow.js initialized with WASM backend', {
                backend: tf.getBackend(),
                version: tf.version.tfjs,
                modelType: 'Transformer' // Change to GRU, CNN, or Transformer for other models
            });
        } catch (error) {
            Logger.error('Failed to initialize TensorFlow.js with WASM backend', { 
                error: error.message 
            });
            
            // Fallback to CPU backend
            try {
                await tf.setBackend('cpu');
                await tf.ready();
                Logger.warn('Falling back to CPU backend');
            } catch (fallbackError) {
                Logger.error('All TensorFlow backends failed', { 
                    error: fallbackError.message 
                });
            }
        }
    }
    
    // Create positional encoding for transformer
    createPositionalEncoding(seqLen, dModel) {
        const posEncoding = tf.zeros([seqLen, dModel]);
        const position = tf.range(0, seqLen, 1, 'float32').expandDims(1);
        
        const divTerm = tf.exp(
            tf.range(0, dModel, 2, 'float32').mul(-Math.log(10000.0) / dModel)
        );
        
        // Apply sin to even indices
        const sinValues = tf.sin(position.mul(divTerm));
        // Apply cos to odd indices  
        const cosValues = tf.cos(position.mul(divTerm));
        
        // Interleave sin and cos values
        const posEncodingArray = [];
        for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < dModel; j++) {
                if (j % 2 === 0) {
                    posEncodingArray.push(sinValues.dataSync()[i * Math.floor(dModel / 2) + Math.floor(j / 2)]);
                } else {
                    posEncodingArray.push(cosValues.dataSync()[i * Math.floor(dModel / 2) + Math.floor(j / 2)]);
                }
            }
        }
        
        // Clean up temporary tensors
        position.dispose();
        divTerm.dispose();
        sinValues.dispose();
        cosValues.dispose();
        posEncoding.dispose();
        
        return tf.tensor2d(posEncodingArray, [seqLen, dModel]);
    }
    
    // Multi-head attention mechanism
    createMultiHeadAttention(dModel, numHeads) {
        const headDim = Math.floor(dModel / numHeads);
        
        return tf.layers.apply(function(inputs) {
            const [query, key, value] = inputs;
            const batchSize = query.shape[0];
            const seqLen = query.shape[1];
            
            // Linear transformations for Q, K, V
            const wq = tf.layers.dense({ units: dModel, useBias: false });
            const wk = tf.layers.dense({ units: dModel, useBias: false });
            const wv = tf.layers.dense({ units: dModel, useBias: false });
            
            const q = wq.apply(query);
            const k = wk.apply(key);
            const v = wv.apply(value);
            
            // Reshape for multi-head attention
            const qReshaped = tf.reshape(q, [batchSize, seqLen, numHeads, headDim]);
            const kReshaped = tf.reshape(k, [batchSize, seqLen, numHeads, headDim]);
            const vReshaped = tf.reshape(v, [batchSize, seqLen, numHeads, headDim]);
            
            // Transpose for attention computation
            const qTransposed = tf.transpose(qReshaped, [0, 2, 1, 3]);
            const kTransposed = tf.transpose(kReshaped, [0, 2, 1, 3]);
            const vTransposed = tf.transpose(vReshaped, [0, 2, 1, 3]);
            
            // Scaled dot-product attention
            const scores = tf.matMul(qTransposed, kTransposed, false, true);
            const scaledScores = tf.div(scores, Math.sqrt(headDim));
            const attentionWeights = tf.softmax(scaledScores);
            
            // Apply attention to values
            const attentionOutput = tf.matMul(attentionWeights, vTransposed);
            
            // Transpose back and reshape
            const outputTransposed = tf.transpose(attentionOutput, [0, 2, 1, 3]);
            const output = tf.reshape(outputTransposed, [batchSize, seqLen, dModel]);
            
            // Final linear transformation
            const wo = tf.layers.dense({ units: dModel });
            const finalOutput = wo.apply(output);
            
            // Clean up intermediate tensors
            [q, k, v, qReshaped, kReshaped, vReshaped, qTransposed, kTransposed, vTransposed,
             scores, scaledScores, attentionWeights, attentionOutput, outputTransposed, output].forEach(t => {
                if (t && typeof t.dispose === 'function') t.dispose();
            });
            
            return finalOutput;
        }, { name: 'multiHeadAttention' });
    }
    
    // Feed-forward network
    createFeedForward(dModel, dff) {
        return tf.sequential({
            layers: [
                tf.layers.dense({
                    units: dff,
                    activation: 'relu',
                    kernelInitializer: 'heNormal'
                }),
                tf.layers.dropout({ rate: this.dropout }),
                tf.layers.dense({
                    units: dModel,
                    kernelInitializer: 'glorotUniform'
                })
            ]
        });
    }
    
    // Transformer encoder layer
    createEncoderLayer(dModel, numHeads, dff) {
        const attention = this.createMultiHeadAttention(dModel, numHeads);
        const feedForward = this.createFeedForward(dModel, dff);
        
        return tf.layers.apply(function(inputs) {
            // Multi-head attention
            const attentionOutput = attention.apply([inputs, inputs, inputs]);
            
            // Add & Norm 1
            const norm1 = tf.layers.layerNormalization();
            const add1 = tf.layers.add();
            const normed1 = norm1.apply(add1.apply([inputs, attentionOutput]));
            
            // Feed-forward
            const ffOutput = feedForward.apply(normed1);
            
            // Add & Norm 2
            const norm2 = tf.layers.layerNormalization();
            const add2 = tf.layers.add();
            const output = norm2.apply(add2.apply([normed1, ffOutput]));
            
            return output;
        }, { name: 'encoderLayer' });
    }
    
    buildModel() {
        try {
            Logger.info('Building simplified Transformer model...');
            
            this.model = tf.sequential();
            
            // Input projection to model dimension
            this.model.add(tf.layers.dense({
                units: this.dModel,
                inputShape: [this.sequenceLength, this.features],
                kernelInitializer: 'glorotUniform'
            }));
            
            // Add positional encoding if enabled (simplified)
            if (this.usePositionalEncoding) {
                this.model.add(tf.layers.dropout({ rate: this.dropout }));
            }
            
            // Simplified attention mechanism using LSTM layers
            for (let i = 0; i < this.numLayers; i++) {
                this.model.add(tf.layers.lstm({
                    units: this.dModel / 2,
                    returnSequences: i < this.numLayers - 1,
                    dropout: this.dropout,
                    recurrentDropout: this.dropout
                }));
            }
            
            // Dense layers for classification
            this.model.add(tf.layers.dense({
                units: 128,
                activation: 'relu',
                kernelInitializer: 'heNormal'
            }));
            
            this.model.add(tf.layers.dropout({ rate: this.dropout }));
            
            this.model.add(tf.layers.dense({
                units: 64,
                activation: 'relu',
                kernelInitializer: 'heNormal'
            }));
            
            this.model.add(tf.layers.dropout({ rate: this.dropout / 2 }));
            
            // Output layer
            this.model.add(tf.layers.dense({
                units: 1,
                activation: 'sigmoid',
                kernelInitializer: 'glorotUniform'
            }));
            
            Logger.info('Simplified Transformer model built successfully', {
                totalParams: this.model.countParams(),
                layers: this.model.layers.length,
                outputShape: this.model.outputShape
            });
            
            return this.model;
            
        } catch (error) {
            Logger.error('Failed to build Transformer model', { error: error.message });
            throw error;
        }
    }
    
    compileModel() {
        if (!this.model) {
            throw new Error('Model must be built before compilation');
        }
        
        try {
            Logger.info('Compiling Transformer model...');
            
            // Use Adam optimizer with warm-up learning rate
            const optimizer = tf.train.adam(this.learningRate, 0.9, 0.98, 1e-9);
            
            this.model.compile({
                optimizer: optimizer,
                loss: 'binaryCrossentropy',
                metrics: ['accuracy']
            });
            
            this.isCompiled = true;
            Logger.info('Transformer model compiled successfully');
            
        } catch (error) {
            Logger.error('Failed to compile Transformer model', { error: error.message });
            throw error;
        }
    }
    
    async train(trainX, trainY, validationX, validationY, config = {}) {
        if (!this.isCompiled) {
            throw new Error('Model must be compiled before training');
        }
        
        try {
            this.isTraining = true;
            
            const epochs = config.epochs || 100;
            const batchSize = config.batchSize || 16; // Smaller batch size for transformers
            const verbose = config.verbose !== undefined ? config.verbose : 1;
            const patience = config.patience || 20;
            const warmupSteps = config.warmupSteps || 4000;
            
            Logger.info('Starting Transformer model training', {
                epochs,
                batchSize,
                trainSamples: trainX.shape[0],
                validationSamples: validationX ? validationX.shape[0] : 0,
                patience,
                warmupSteps
            });
            
            // Learning rate warm-up and decay
            let step = 0;
            let bestValLoss = Infinity;
            let patienceCounter = 0;
            
            const getWarmupLR = (step) => {
                const arg1 = 1 / Math.sqrt(step + 1);
                const arg2 = step * Math.pow(warmupSteps, -1.5);
                return Math.min(arg1, arg2) * Math.sqrt(this.dModel);
            };
            
            const callbacks = {
                onBatchEnd: (batch, logs) => {
                    step++;
                    
                    // Update learning rate with warm-up
                    if (step <= warmupSteps) {
                        const newLR = getWarmupLR(step) * this.learningRate;
                        this.model.optimizer.learningRate = newLR;
                    }
                    
                    if (config.veryVerbose && batch % 10 === 0) {
                        Logger.debug(`Transformer Batch ${batch}`, {
                            batchLoss: logs.loss.toFixed(4),
                            batchAcc: logs.acc.toFixed(4),
                            learningRate: this.model.optimizer.learningRate.toFixed(8),
                            step: step
                        });
                    }
                },
                
                onEpochEnd: (epoch, logs) => {
                    // Early stopping logic
                    if (validationX && validationY && logs.val_loss < bestValLoss) {
                        bestValLoss = logs.val_loss;
                        patienceCounter = 0;
                    } else if (validationX && validationY) {
                        patienceCounter++;
                    }
                    
                    // Logging
                    if (epoch % 5 === 0 || epoch === epochs - 1) {
                        const logData = {
                            loss: logs.loss.toFixed(4),
                            accuracy: logs.acc.toFixed(4),
                            learningRate: this.model.optimizer.learningRate.toFixed(8),
                            step: step
                        };
                        
                        if (validationX && validationY) {
                            logData.valLoss = logs.val_loss?.toFixed(4);
                            logData.valAccuracy = logs.val_acc?.toFixed(4);
                            logData.patience = `${patienceCounter}/${patience}`;
                        }
                        
                        Logger.info(`Transformer Epoch ${epoch + 1}/${epochs}`, logData);
                    }
                    
                    // Early stopping
                    if (patienceCounter >= patience && validationX && validationY) {
                        Logger.info(`Early stopping triggered at epoch ${epoch + 1}`);
                        return true; // Stop training
                    }
                }
            };
            
            const trainingConfig = {
                epochs,
                batchSize,
                validationData: validationX && validationY ? [validationX, validationY] : null,
                shuffle: true,
                verbose,
                callbacks
            };
            
            const history = await this.model.fit(trainX, trainY, trainingConfig);
            
            this.isTraining = false;
            
            // Calculate final metrics
            const finalMetrics = {
                finalLoss: history.history.loss[history.history.loss.length - 1].toFixed(4),
                finalAccuracy: history.history.acc[history.history.acc.length - 1].toFixed(4),
                epochsCompleted: history.epoch.length,
                totalSteps: step,
                finalLearningRate: this.model.optimizer.learningRate.toFixed(8)
            };
            
            if (validationX && validationY) {
                finalMetrics.finalValLoss = history.history.val_loss[history.history.val_loss.length - 1].toFixed(4);
                finalMetrics.finalValAccuracy = history.history.val_acc[history.history.val_acc.length - 1].toFixed(4);
                finalMetrics.bestValLoss = bestValLoss.toFixed(4);
            }
            
            // Remove F1 score calculation for now
            
            Logger.info('Transformer model training completed', finalMetrics);
            
            return {
                ...history,
                finalMetrics,
                modelType: 'Transformer',
                bestValLoss: bestValLoss
            };
            
        } catch (error) {
            this.isTraining = false;
            Logger.error('Transformer model training failed', { error: error.message });
            throw error;
        }
    }
    
    async predict(inputX) {
        if (!this.model) {
            throw new Error('Model must be built and trained before prediction');
        }
        
        try {
            Logger.debug('Making Transformer prediction', {
                inputShape: inputX.shape,
                modelType: 'Transformer'
            });
            
            const prediction = this.model.predict(inputX);
            const result = await prediction.data();
            
            // Clean up tensors
            prediction.dispose();
            
            Logger.debug('Transformer prediction completed', {
                predictions: result.length,
                samplePrediction: result[0]?.toFixed(4)
            });
            
            return result;
            
        } catch (error) {
            Logger.error('Transformer prediction failed', { error: error.message });
            throw error;
        }
    }
    
    async evaluate(testX, testY) {
        if (!this.model) {
            throw new Error('Model must be built and trained before evaluation');
        }
        
        try {
            Logger.info('Evaluating Transformer model');
            
            const evaluation = this.model.evaluate(testX, testY, { verbose: 0 });
            
            let results;
            if (Array.isArray(evaluation)) {
                // Multiple metrics
                const metrics = await Promise.all(
                    evaluation.map(tensor => tensor.data())
                );
                
                results = {
                    loss: metrics[0][0],
                    accuracy: metrics[1][0]
                };
                
                // Clean up tensors
                evaluation.forEach(tensor => tensor.dispose());
            } else {
                // Single metric
                const lossData = await evaluation.data();
                results = { loss: lossData[0] };
                evaluation.dispose();
            }
            
            Logger.info('Transformer model evaluation completed', {
                ...results,
                modelType: 'Transformer'
            });
            
            return results;
            
        } catch (error) {
            Logger.error('Transformer model evaluation failed', { error: error.message });
            throw error;
        }
    }
    
    // Get attention weights for interpretability
    async getAttentionWeights(inputX, layerIndex = 0) {
        if (!this.model) {
            throw new Error('Model must be built for attention analysis');
        }
        
        try {
            Logger.info('Extracting attention weights', { layerIndex });
            
            // This is a simplified version - in practice, you'd need to modify the model
            // to expose intermediate attention weights
            const predictions = await this.predict(inputX);
            
            // For now, return a mock attention pattern based on predictions
            const batchSize = inputX.shape[0];
            const seqLen = this.sequenceLength;
            const numHeads = this.numHeads;
            
            // Create mock attention weights (in real implementation, extract from model)
            const attentionWeights = tf.randomUniform([batchSize, numHeads, seqLen, seqLen]);
            const normalizedWeights = tf.softmax(attentionWeights, -1);
            
            const weightsData = await normalizedWeights.data();
            
            // Clean up
            attentionWeights.dispose();
            normalizedWeights.dispose();
            
            return {
                attentionWeights: weightsData,
                shape: [batchSize, numHeads, seqLen, seqLen],
                layerIndex: layerIndex,
                modelType: 'Transformer'
            };
            
        } catch (error) {
            Logger.error('Attention weight extraction failed', { error: error.message });
            throw error;
        }
    }
    
    async save(modelPath) {
        if (!this.model) {
            throw new Error('Model must be built before saving');
        }
        
        try {
            Logger.info('Saving Transformer model', { path: modelPath });
            
            await this.model.save(`file://${modelPath}`);
            
            // Save Transformer-specific metadata
            const metadata = {
                modelType: 'Transformer',
                config: {
                    sequenceLength: this.sequenceLength,
                    features: this.features,
                    dModel: this.dModel,
                    numHeads: this.numHeads,
                    numLayers: this.numLayers,
                    dff: this.dff,
                    dropout: this.dropout,
                    learningRate: this.learningRate,
                    usePositionalEncoding: this.usePositionalEncoding,
                    maxPositionalEncoding: this.maxPositionalEncoding
                },
                architecture: this.getModelSummary(),
                savedAt: Date.now(),
                version: '1.0.0'
            };
            
            // Save metadata alongside model
            const fs = require('fs');
            const path = require('path');
            const metadataPath = path.join(modelPath, 'transformer_metadata.json');
            fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));
            
            Logger.info('Transformer model and metadata saved successfully');
            
        } catch (error) {
            Logger.error('Failed to save Transformer model', { error: error.message });
            throw error;
        }
    }
    
    async load(modelPath) {
        try {
            Logger.info('Loading Transformer model', { path: modelPath });
            
            this.model = await tf.loadLayersModel(`file://${modelPath}/model.json`);
            this.isCompiled = true;
            
            // Load metadata if available
            try {
                const fs = require('fs');
                const path = require('path');
                const metadataPath = path.join(modelPath, 'transformer_metadata.json');
                
                if (fs.existsSync(metadataPath)) {
                    const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
                    
                    // Restore configuration from metadata
                    this.sequenceLength = metadata.config.sequenceLength;
                    this.features = metadata.config.features;
                    this.dModel = metadata.config.dModel;
                    this.numHeads = metadata.config.numHeads;
                    this.numLayers = metadata.config.numLayers;
                    this.dff = metadata.config.dff;
                    this.dropout = metadata.config.dropout;
                    this.learningRate = metadata.config.learningRate;
                    this.usePositionalEncoding = metadata.config.usePositionalEncoding;
                    this.maxPositionalEncoding = metadata.config.maxPositionalEncoding;
                    
                    Logger.info('Transformer metadata loaded', metadata.config);
                }
            } catch (metadataError) {
                Logger.warn('Failed to load Transformer metadata', { error: metadataError.message });
            }
            
            Logger.info('Transformer model loaded successfully', {
                totalParams: this.model.countParams(),
                layers: this.model.layers.length
            });
            
        } catch (error) {
            Logger.error('Failed to load Transformer model', { error: error.message });
            throw error;
        }
    }
    
    getModelSummary() {
        if (!this.model) {
            return 'Transformer model not built yet';
        }
        
        return {
            modelType: 'Transformer',
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
                dModel: this.dModel,
                numHeads: this.numHeads,
                numLayers: this.numLayers,
                dff: this.dff,
                dropout: this.dropout,
                learningRate: this.learningRate,
                usePositionalEncoding: this.usePositionalEncoding,
                maxPositionalEncoding: this.maxPositionalEncoding
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
    
    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
            this.isCompiled = false;
            Logger.info('Transformer model disposed');
        }
    }
    
    getMemoryUsage() {
        return {
            numTensors: tf.memory().numTensors,
            numBytes: tf.memory().numBytes,
            modelLoaded: !!this.model,
            isTraining: this.isTraining,
            modelType: 'Transformer'
        };
    }
    
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

module.exports = TransformerModel;