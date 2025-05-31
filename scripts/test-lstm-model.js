require('dotenv').config();
const DataClient = require('../src/data/DataClient');
const FeatureExtractor = require('../src/data/FeatureExtractor');
const DataPreprocessor = require('../src/data/DataPreprocessor');
const LSTMModel = require('../src/models/LSTMModel');
const { Logger } = require('../src/utils');
const config = require('config');

async function testLSTMModel() {
    console.log('🚀 Testing LSTM Model...');
    
    const dataClient = new DataClient();
    const featureExtractor = new FeatureExtractor(config.get('ml.features'));
    const preprocessor = new DataPreprocessor(config.get('ml.models.lstm'));
    
    try {
        // Test 1: Get data and extract features
        console.log('\n📊 Test 1: Preparing data...');
        const rvnData = await dataClient.getPairData('RVN');
        
        // Extract features for multiple time points
        const featuresArray = [];
        const targets = [];
        
        // Simulate historical feature extraction (in real implementation, this would be stored)
        console.log('Simulating feature extraction for multiple time points...');
        for (let i = 60; i < rvnData.history.closes.length - 5; i++) {
            // Create a subset of data up to point i
            const historicalData = {
                pair: 'RVN',
                history: {
                    closes: rvnData.history.closes.slice(0, i + 1),
                    highs: rvnData.history.highs.slice(0, i + 1),
                    lows: rvnData.history.lows.slice(0, i + 1),
                    volumes: rvnData.history.volumes.slice(0, i + 1),
                    timestamps: rvnData.history.timestamps.slice(0, i + 1)
                },
                strategies: rvnData.strategies // Use current strategies for simplicity
            };
            
            const result = featureExtractor.extractFeatures(historicalData);
            featuresArray.push(result.features);
            
            // Create target: 1 if price goes up in next 5 periods, 0 otherwise
            const currentPrice = rvnData.history.closes[i];
            const futurePrice = rvnData.history.closes[i + 5];
            const target = futurePrice > currentPrice ? 1 : 0;
            targets.push(target);
        }
        
        console.log('✅ Data prepared:', {
            samples: featuresArray.length,
            features: featuresArray[0].length,
            targets: targets.length
        });
        
        // Test 2: Preprocess data
        console.log('\n📊 Test 2: Preprocessing data...');
        const processedData = await preprocessor.prepareTrainingData(featuresArray, targets);
        
        console.log('✅ Data preprocessed:', {
            trainSamples: processedData.trainX.shape[0],
            validationSamples: processedData.validationX.shape[0],
            testSamples: processedData.testX.shape[0],
            sequenceLength: processedData.trainX.shape[1],
            features: processedData.trainX.shape[2]
        });
        
        // Test 3: Build and compile model
        console.log('\n📊 Test 3: Building LSTM model...');
        const lstmConfig = {
            ...config.get('ml.models.lstm'),
            features: featuresArray[0].length
        };
        const model = new LSTMModel(lstmConfig);
        
        model.buildModel();
        model.compileModel();
        
        console.log('✅ Model built and compiled:', model.getModelSummary());
        
        // Test 4: Train model (short training for testing)
        console.log('\n📊 Test 4: Training LSTM model (quick test)...');
        const trainingConfig = {
            epochs: 5, // Reduced for testing
            batchSize: 16,
            verbose: 0
        };
        
        const history = await model.train(
            processedData.trainX,
            processedData.trainY,
            processedData.validationX,
            processedData.validationY,
            trainingConfig
        );
        
        console.log('✅ Model training completed');
        
        // Test 5: Make predictions
        console.log('\n📊 Test 5: Making predictions...');
        const predictions = await model.predict(processedData.testX.slice([0, 0, 0], [5, -1, -1])); // Test on first 5 samples
        
        console.log('✅ Predictions made:', {
            samples: predictions.length,
            predictions: Array.from(predictions).map(p => p.toFixed(4))
        });
        
        // Test 6: Evaluate model
        console.log('\n📊 Test 6: Evaluating model...');
        const evaluation = await model.evaluate(processedData.testX, processedData.testY);
        
        console.log('✅ Model evaluation:', {
            loss: evaluation.loss.toFixed(4),
            accuracy: evaluation.accuracy.toFixed(4)
        });
        
        // Test 7: Real-time prediction simulation
        console.log('\n📊 Test 7: Real-time prediction simulation...');
        const realtimeInput = await preprocessor.prepareRealTimeData(featuresArray.slice(-60));
        const realtimePrediction = await model.predict(realtimeInput);
        
        console.log('✅ Real-time prediction:', {
            prediction: realtimePrediction[0].toFixed(4),
            signal: realtimePrediction[0] > 0.5 ? 'BUY' : 'SELL',
            confidence: Math.abs(realtimePrediction[0] - 0.5) * 2
        });
        
        // Cleanup
        console.log('\n🧹 Cleaning up tensors...');
        processedData.trainX.dispose();
        processedData.trainY.dispose();
        processedData.validationX.dispose();
        processedData.validationY.dispose();
        processedData.testX.dispose();
        processedData.testY.dispose();
        realtimeInput.dispose();
        
        model.dispose();
        preprocessor.dispose();
        
        console.log('\n🎉 All LSTM model tests passed!');
        console.log('ML pipeline is ready for integration! 🚀');
        
    } catch (error) {
        console.error('\n❌ LSTM model test failed:', error.message);
        Logger.error('LSTM model test failed', { error: error.message });
        process.exit(1);
    }
}

testLSTMModel();