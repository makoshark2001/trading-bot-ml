# Trading Bot ML - Technical Manual

## 🤖 Overview

The **trading-bot-ml** is the machine learning service of the modular trading bot architecture, providing advanced LSTM neural network predictions, feature engineering, and AI-powered trading signal enhancement. Operating on **Port 3001**, it integrates seamlessly with trading-bot-core to deliver sophisticated price prediction capabilities.

### Key Capabilities
- **LSTM Neural Networks** for price direction and volatility prediction
- **Advanced Feature Engineering** from 11 technical indicators
- **Real-time ML Predictions** with confidence scoring
- **Feature Extraction Pipeline** optimized for time-series data
- **RESTful API** serving ML predictions and feature data
- **TensorFlow.js Integration** for browser-compatible ML models
- **Data Preprocessing** with normalization and sequence generation

---

## 🧠 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  TRADING-BOT-ML (Port 3001)                │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│  │   DataClient    │  │ FeatureExtractor │  │ DataPreprocessor││
│  │                 │  │                 │  │                 ││
│  │ • Core Service  │  │ • 50+ Features  │  │ • Normalization ││
│  │   Integration   │  │ • Multi-timeframe│  │ • Sequencing   ││
│  │ • Health Monitor │  │ • Technical +   │  │ • Train/Test   ││
│  │ • Data Fetching │  │   Price Features │  │   Splitting    ││
│  └─────────────────┘  └─────────────────┘  └─────────────────┘│
│                                 │                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│  │   LSTMModel     │  │   MLServer      │  │  Prediction     ││
│  │                 │  │                 │  │   Engine        ││
│  │ • TensorFlow.js │  │ • RESTful API   │  │ • Real-time     ││
│  │ • Sequence      │  │ • Model Mgmt    │  │   Inference     ││
│  │   Processing    │  │ • Training API  │  │ • Confidence    ││
│  │ • Training      │  │ • Health Checks │  │   Scoring       ││
│  └─────────────────┘  └─────────────────┘  └─────────────────┘│
└─────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────┼──────────┐
                    │          │          │
          ┌─────────▼─┐  ┌─────▼─────┐  ┌─▼──────────┐
          │   Core    │  │ Backtest  │  │ Dashboard  │
          │ Service   │  │ Service   │  │  Service   │
          │(Port 3000)│  │(Port 3002)│  │(Port 3005) │
          └───────────┘  └───────────┘  └────────────┘
```

---

## 🛠️ Quick Start

### Prerequisites
- **Node.js** >= 16.0.0
- **npm** >= 8.0.0
- **trading-bot-core** running on Port 3000
- **Minimum 4GB RAM** for TensorFlow.js operations

### Installation

1. **Clone and Setup**
```bash
git clone <repository-url>
cd trading-bot-ml
npm install
```

2. **Environment Configuration**
```bash
cp .env.example .env
# No API keys required - connects to core service
```

3. **Start the ML Service**
```bash
npm start
```

4. **Verify Installation**
```bash
# Check ML service health
curl http://localhost:3001/api/health

# Test prediction endpoint
curl http://localhost:3001/api/predictions/RVN
```

### Verify Core Service Connection
```bash
# Ensure core service is running first
curl http://localhost:3000/api/health

# ML service should show core connection as healthy
curl http://localhost:3001/api/health | jq '.core'
```

---

## 🔌 API Reference

### Base URL
```
http://localhost:3001
```

### Core Endpoints

#### 1. **GET /api/health**
ML service health check with core service connectivity status.

**Response:**
```json
{
  "status": "healthy",
  "service": "trading-bot-ml",
  "timestamp": 1704067200000,
  "uptime": "01:45:22",
  "core": {
    "status": "healthy",
    "dataCollection": {
      "totalDataPoints": 15420,
      "isCollecting": true
    }
  },
  "models": {
    "loaded": 2,
    "pairs": ["RVN", "XMR"]
  },
  "predictions": {
    "cached": 6,
    "lastUpdate": 1704067180000
  }
}
```

#### 2. **GET /api/predictions/:pair**
Get ML prediction for a specific trading pair.

**Parameters:**
- `pair` (string): Trading pair symbol (e.g., "RVN", "XMR")

**Response:**
```json
{
  "pair": "RVN",
  "prediction": {
    "direction": "up",
    "confidence": 0.742,
    "probability": 0.742,
    "signal": "BUY",
    "features": {
      "count": 52,
      "sample": [0.651, -0.234, 1.123, 0.445, -0.892]
    },
    "model": "LSTM",
    "version": "1.0.0"
  },
  "timestamp": 1704067200000,
  "cached": false
}
```

#### 3. **GET /api/predictions**
Get ML predictions for all available trading pairs.

**Response:**
```json
{
  "predictions": {
    "RVN": {
      "direction": "up",
      "confidence": 0.742,
      "signal": "BUY",
      "timestamp": 1704067200000
    },
    "XMR": {
      "direction": "down",
      "confidence": 0.653,
      "signal": "SELL",
      "timestamp": 1704067195000
    }
  },
  "timestamp": 1704067200000,
  "pairs": ["RVN", "XMR", "BEL", "DOGE", "KAS", "SAL"]
}
```

#### 4. **GET /api/features/:pair**
Get extracted features for a trading pair (debugging/analysis).

**Parameters:**
- `pair` (string): Trading pair symbol

**Response:**
```json
{
  "pair": "RVN",
  "features": {
    "count": 52,
    "names": [
      "price_currentPrice",
      "price_return_5",
      "price_return_10",
      "indicators_rsi_confidence",
      "indicators_macd_line",
      "volume_current_volume",
      "volatility_recent_volatility",
      "time_hour_sin"
    ],
    "values": [0.651, -0.234, 1.123, 0.445, -0.892, 2.156, 0.089, -0.456],
    "metadata": {
      "pair": "RVN",
      "dataPoints": 180,
      "extractedAt": "2024-01-01T12:00:00.000Z",
      "featureCount": 52
    }
  },
  "timestamp": 1704067200000
}
```

#### 5. **POST /api/train/:pair**
Start model training for a specific pair (asynchronous).

**Parameters:**
- `pair` (string): Trading pair symbol

**Request Body:**
```json
{
  "epochs": 100,
  "batchSize": 32,
  "learningRate": 0.001,
  "validationSplit": 0.2
}
```

**Response:**
```json
{
  "message": "Training started for RVN",
  "pair": "RVN",
  "config": {
    "epochs": 100,
    "batchSize": 32,
    "learningRate": 0.001
  },
  "timestamp": 1704067200000
}
```

#### 6. **GET /api/models/:pair/status**
Get model status and information for a pair.

**Response:**
```json
{
  "pair": "RVN",
  "hasModel": true,
  "modelInfo": {
    "layers": 4,
    "totalParams": 12847,
    "trainableParams": 12847,
    "inputShape": [null, 60, 52],
    "outputShape": [null, 1],
    "isCompiled": true,
    "isTraining": false
  },
  "timestamp": 1704067200000
}
```

---

## 🧪 Feature Engineering

The ML service extracts 50+ features from market data and technical indicators:

### 1. **Price-Based Features (12 features)**
```javascript
{
  // Current price metrics
  currentPrice: 0.0234,           // Normalized current price
  pricePosition: 0.65,            // Position in recent price range (0-1)
  
  // Returns for different periods
  return_5: 0.023,                // 5-period return
  return_10: -0.015,              // 10-period return  
  return_20: 0.045,               // 20-period return
  
  // High-low analysis
  hlSpread: 0.0002,               // High-low spread
  hlPosition: 0.75                // Price position in high-low range
}
```

### 2. **Technical Indicator Features (33 features)**
```javascript
{
  // RSI features
  rsi_confidence: 0.65,           // RSI signal confidence
  rsi_value: 45.2,               // Normalized RSI value
  rsi_overbought: 0,             // Binary overbought flag
  rsi_oversold: 0,               // Binary oversold flag
  
  // MACD features
  macd_line: 0.000045,           // MACD line value
  macd_signal: 0.000038,         // Signal line value
  macd_histogram: 0.000007,      // Histogram value
  
  // Bollinger Bands features
  bb_percentB: 0.65,             // %B position
  bb_bandwidth: 0.034,           // Band width
  bb_position: 0,                // Position relative to bands
  
  // Volume features
  volume_ratio: 1.25,            // Volume vs average
  volume_spike: 1,               // Volume spike detected
  
  // And 20+ more from all 11 indicators...
}
```

### 3. **Volume Features (4 features)**
```javascript
{
  current_volume: 125000,         // Current volume
  volume_ma_ratio: 1.25,         // Volume vs moving average
  volume_trend: 0.15             // Volume trend direction
}
```

### 4. **Volatility Features (3 features)**
```javascript
{
  volatility: 0.023,             // Recent volatility
  recent_volatility: 0.028,      // Very recent volatility
  volatility_ratio: 1.22        // Recent vs historical volatility
}
```

### 5. **Time-Based Features (6 features)**
```javascript
{
  hour_of_day: 0.5,              // Hour normalized (0-1)
  day_of_week: 0.3,              // Day normalized (0-1)
  hour_sin: 0.707,               // Sine-encoded hour
  hour_cos: 0.707,               // Cosine-encoded hour
  day_sin: 0.434,                // Sine-encoded day
  day_cos: 0.901                 // Cosine-encoded day
}
```

---

## 🔮 LSTM Model Architecture

### Model Configuration
```javascript
{
  sequenceLength: 60,     // 60 time steps (5 hours of 5-min data)
  features: 52,           // Number of input features
  units: 50,              // LSTM units per layer
  layers: 2,              // Number of LSTM layers
  dropout: 0.2,           // Dropout rate
  learningRate: 0.001     // Adam optimizer learning rate
}
```

### Model Structure
```
Input Layer: [batch_size, 60, 52]
    ↓
LSTM Layer 1: 50 units, return_sequences=true, dropout=0.2
    ↓
LSTM Layer 2: 50 units, return_sequences=false, dropout=0.2
    ↓
Dense Layer: 32 units, activation='relu'
    ↓
Dropout Layer: rate=0.2
    ↓
Output Layer: 1 unit, activation='sigmoid'
    ↓
Output: [batch_size, 1] (probability of price increase)
```

### Training Configuration
```javascript
{
  optimizer: 'adam',
  loss: 'binaryCrossentropy',
  metrics: ['accuracy'],
  epochs: 100,
  batchSize: 32,
  validationSplit: 0.2,
  shuffle: true
}
```

---

## 📊 Data Pipeline

### 1. **Data Collection**
```javascript
// Fetch from trading-bot-core
const coreData = await dataClient.getPairData('RVN');
// Returns: { history, strategies, pair }
```

### 2. **Feature Extraction**
```javascript
const features = featureExtractor.extractFeatures(coreData);
// Returns: { features: [52 values], featureNames: [...], metadata: {...} }
```

### 3. **Data Preprocessing**
```javascript
const processedData = await preprocessor.prepareTrainingData(featuresArray, targets);
// Returns: { trainX, trainY, validationX, validationY, testX, testY }
```

### 4. **Model Training**
```javascript
const history = await model.train(trainX, trainY, validationX, validationY, config);
// Returns: training history with loss and accuracy metrics
```

### 5. **Real-time Prediction**
```javascript
const prediction = await model.predict(realtimeInput);
// Returns: probability array [0.742] (74.2% chance of price increase)
```

---

## 🏗️ Integration Guide for Other Modules

### For Backtest Service (trading-bot-backtest)

#### ML Signal Integration
```javascript
const axios = require('axios');

class MLSignalProvider {
  constructor() {
    this.mlServiceUrl = 'http://localhost:3001';
  }
  
  async getMLSignal(pair) {
    try {
      const response = await axios.get(`${this.mlServiceUrl}/api/predictions/${pair}`);
      const prediction = response.data.prediction;
      
      return {
        signal: prediction.signal,           // 'BUY', 'SELL', 'HOLD'
        confidence: prediction.confidence,   // 0.0 to 1.0
        direction: prediction.direction,     // 'up' or 'down'
        probability: prediction.probability,  // Raw model output
        timestamp: response.data.timestamp
      };
    } catch (error) {
      console.warn(`ML signal unavailable for ${pair}:`, error.message);
      return null;
    }
  }
  
  // Enhanced backtest with ML signals
  async enhanceBacktestWithML(technicalSignals, pair) {
    const mlSignal = await this.getMLSignal(pair);
    
    if (!mlSignal || mlSignal.confidence < 0.6) {
      return technicalSignals; // Use only technical analysis
    }
    
    // Combine technical and ML signals
    const combinedSignal = {
      action: this.combineSignals(technicalSignals.action, mlSignal.signal),
      confidence: Math.max(technicalSignals.confidence, mlSignal.confidence),
      sources: {
        technical: technicalSignals,
        ml: mlSignal
      }
    };
    
    return combinedSignal;
  }
  
  combineSignals(technicalAction, mlSignal) {
    // Agreement between technical and ML
    if (technicalAction === mlSignal.toLowerCase()) {
      return technicalAction;
    }
    
    // ML override for high confidence predictions
    if (mlSignal.confidence > 0.8) {
      return mlSignal.toLowerCase();
    }
    
    // Default to hold when signals conflict
    return 'hold';
  }
}
```

### For Dashboard Service (trading-bot-dashboard)

#### ML Predictions Display
```javascript
// Fetch ML predictions for dashboard
const mlPredictions = await axios.get('http://localhost:3001/api/predictions');

const enhancedPairData = pairs.map(pair => ({
  pair,
  technicalSignal: coreData.strategies[pair].ensemble?.suggestion,
  technicalConfidence: coreData.strategies[pair].ensemble?.confidence,
  
  // ML enhancement
  mlSignal: mlPredictions.data.predictions[pair]?.signal,
  mlConfidence: mlPredictions.data.predictions[pair]?.confidence,
  mlDirection: mlPredictions.data.predictions[pair]?.direction,
  
  // Combined signal
  finalSignal: combineSignals(technical, ml),
  agreementLevel: calculateAgreement(technical, ml)
}));
```

### For Risk Management Service (trading-bot-risk)

#### ML-Enhanced Risk Assessment
```javascript
class MLRiskAssessment {
  async assessMLRisk(pair) {
    const predictions = await axios.get(`http://localhost:3001/api/predictions/${pair}`);
    const features = await axios.get(`http://localhost:3001/api/features/${pair}`);
    
    return {
      // Prediction uncertainty as risk metric
      predictionRisk: 1 - predictions.data.prediction.confidence,
      
      // Feature stability analysis
      featureStability: this.analyzeFeatureStability(features.data.features),
      
      // Model confidence trends
      confidenceTrend: this.calculateConfidenceTrend(pair),
      
      // Recommendation
      riskLevel: this.calculateOverallMLRisk(predictions, features)
    };
  }
  
  analyzeFeatureStability(features) {
    // Analyze feature distribution for outliers
    const volatilityFeatures = features.values.filter((_, i) => 
      features.names[i].includes('volatility')
    );
    
    return {
      volatilityScore: Math.max(...volatilityFeatures),
      outlierCount: features.values.filter(f => Math.abs(f) > 3).length,
      stabilityScore: 1 - (outlierCount / features.count)
    };
  }
}
```

### For Execution Service (trading-bot-execution)

#### ML-Guided Position Sizing
```javascript
class MLPositionSizer {
  async calculateMLAdjustedSize(pair, baseSize, mlSignal) {
    // Adjust position size based on ML confidence
    const confidenceMultiplier = Math.min(mlSignal.confidence * 1.5, 1.0);
    
    // Additional safety for low-confidence predictions
    const safetyMultiplier = mlSignal.confidence > 0.7 ? 1.0 : 0.5;
    
    const adjustedSize = baseSize * confidenceMultiplier * safetyMultiplier;
    
    return {
      originalSize: baseSize,
      adjustedSize: adjustedSize,
      confidenceMultiplier: confidenceMultiplier,
      safetyMultiplier: safetyMultiplier,
      reasoning: `ML confidence: ${(mlSignal.confidence * 100).toFixed(1)}%`
    };
  }
}
```

---

## 🧪 Testing & Validation

### Available Test Scripts
```bash
# Test ML service connectivity
npm run test:data

# Test feature extraction
npm run test:features  

# Test LSTM model functionality
npm run test:models

# Run all ML tests
npm run test:all
```

### Model Validation
```bash
# Check model performance
curl http://localhost:3001/api/models/RVN/status

# Validate feature extraction
curl http://localhost:3001/api/features/RVN | jq '.features.count'

# Test prediction pipeline
curl http://localhost:3001/api/predictions/RVN | jq '.prediction.confidence'
```

### Performance Benchmarks
- **Feature Extraction**: <500ms for 52 features
- **Model Prediction**: <200ms per pair
- **Training Time**: 5-15 minutes for 100 epochs
- **Memory Usage**: ~500MB during training, ~200MB during inference
- **Prediction Accuracy**: Target >65% for directional predictions

---

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Service Configuration
PORT=3001
NODE_ENV=development

# Core Service Connection
CORE_SERVICE_URL=http://localhost:3000

# ML Configuration
ML_SEQUENCE_LENGTH=60
ML_FEATURES_COUNT=52
ML_PREDICTION_CACHE_TTL=60000

# TensorFlow Configuration
TF_CPP_MIN_LOG_LEVEL=2
TF_FORCE_GPU_ALLOW_GROWTH=true

# Logging
LOG_LEVEL=info
```

### Configuration Files

#### config/default.json
```json
{
  "core": {
    "baseUrl": "http://localhost:3000",
    "endpoints": {
      "data": "/api/data",
      "pair": "/api/pair",
      "health": "/api/health"
    }
  },
  "ml": {
    "features": {
      "indicators": [
        "rsi", "macd", "bollinger", "ma", "volume", 
        "stochastic", "williamsR", "ichimoku", "adx", "cci", "parabolicSAR"
      ],
      "lookbackPeriods": [5, 10, 20],
      "targetPeriods": [1, 3, 5]
    },
    "models": {
      "lstm": {
        "sequenceLength": 60,
        "units": 50,
        "epochs": 100,
        "batchSize": 32,
        "validationSplit": 0.2
      }
    }
  },
  "server": {
    "port": 3001
  },
  "trading": {
    "pairs": ["XMR", "RVN", "BEL", "DOGE", "KAS", "SAL"]
  }
}
```

---

## 📊 Data Structures

### Prediction Response Format
```javascript
{
  direction: "up" | "down",        // Predicted price direction
  confidence: Number,              // 0.0 to 1.0
  probability: Number,             // Raw model output (0.0 to 1.0)
  signal: "BUY" | "SELL" | "HOLD", // Trading recommendation
  features: {
    count: Number,                 // Number of features used
    sample: [Number]               // Sample of feature values
  },
  model: "LSTM",                   // Model type
  version: "1.0.0"                 // Model version
}
```

### Feature Extraction Format
```javascript
{
  features: [Number],              // Array of 52 normalized feature values
  featureNames: [String],          // Names of all features
  metadata: {
    pair: String,                  // Trading pair
    dataPoints: Number,            // Input data points used
    extractedAt: String,           // ISO timestamp
    featureCount: Number           // Total features extracted
  }
}
```

### Model Training Status
```javascript
{
  layers: Number,                  // Number of model layers
  totalParams: Number,             // Total model parameters
  trainableParams: Number,         // Trainable parameters
  inputShape: [Number],            // Input tensor shape
  outputShape: [Number],           // Output tensor shape
  isCompiled: Boolean,             // Model compilation status
  isTraining: Boolean              // Currently training flag
}
```

---

## 🔍 Monitoring & Debugging

### Log Files
```bash
logs/
├── ml.log           # General ML service logs
└── ml-error.log     # ML-specific errors
```

### Debug Commands
```bash
# Enable verbose logging
LOG_LEVEL=debug npm start

# Monitor prediction accuracy
curl http://localhost:3001/api/predictions | jq '.predictions | to_entries[] | {pair: .key, confidence: .value.confidence}'

# Check feature extraction health
curl http://localhost:3001/api/features/RVN | jq '.features.count'

# Monitor model status
curl http://localhost:3001/api/models/RVN/status | jq '.modelInfo'
```

### Common Issues & Solutions

#### 1. **Core Service Connection Failed**
```bash
# Verify core service is running
curl http://localhost:3000/api/health

# Check ML service logs
tail -f logs/ml-error.log | grep "core"

# Restart ML service
npm restart
```

#### 2. **Feature Extraction Errors**
```bash
# Check data availability in core
curl http://localhost:3000/api/pair/RVN | jq '.history.closes | length'

# Verify sufficient data points (need 60+)
curl http://localhost:3001/api/features/RVN | jq '.metadata.dataPoints'
```

#### 3. **Model Training Issues**
```bash
# Check available memory
node -e "console.log(process.memoryUsage())"

# Monitor training progress
tail -f logs/ml.log | grep "training"

# Reduce batch size if memory issues
# Edit config: "batchSize": 16
```

#### 4. **Low Prediction Confidence**
```bash
# Check model status
curl http://localhost:3001/api/models/RVN/status

# Verify feature quality
curl http://localhost:3001/api/features/RVN | jq '.features.values | map(select(. != null)) | length'

# Consider retraining with more data
curl -X POST http://localhost:3001/api/train/RVN
```

---

## 🚀 Performance Optimization

### Memory Management
- **Tensor Disposal**: Automatic cleanup of TensorFlow tensors
- **Model Caching**: Efficient model storage and retrieval
- **Feature Caching**: 1-minute cache for feature calculations

### Prediction Optimization
- **Batch Processing**: Multiple predictions in single inference
- **Preprocessing Pipeline**: Optimized data transformation
- **Model Warm-up**: Keep models loaded for faster predictions

### Training Optimization
- **Early Stopping**: Prevent overfitting with validation monitoring
- **Learning Rate Scheduling**: Dynamic learning rate adjustment
- **Gradient Clipping**: Stable training for LSTM networks

---

## 🔒 Security & Production Considerations

### Model Security
- **Input Validation**: All features validated before processing
- **Output Sanitization**: Predictions bounded and validated
- **Model Versioning**: Track model versions and performance

### Production Deployment
```bash
# Production environment
NODE_ENV=production npm start

# Process management with PM2
pm2 start src/main.js --name trading-bot-ml

# Memory monitoring
pm2 monit trading-bot-ml
```

### Scaling Considerations
- **Horizontal Scaling**: Multiple ML service instances
- **Model Distribution**: Separate models per trading pair
- **Load Balancing**: Distribute prediction requests

---

## 📋 API Usage Examples

### Complete ML Integration
```javascript
const axios = require('axios');

class MLServiceClient {
  constructor(baseURL = 'http://localhost:3001') {
    this.client = axios.create({ baseURL, timeout: 30000 });
  }
  
  async getMLHealth() {
    const response = await this.client.get('/api/health');
    return response.data;
  }
  
  async getPrediction(pair) {
    const response = await this.client.get(`/api/predictions/${pair}`);
    return response.data.prediction;
  }
  
  async getAllPredictions() {
    const response = await this.client.get('/api/predictions');
    return response.data.predictions;
  }
  
  async getFeatures(pair) {
    const response = await this.client.get(`/api/features/${pair}`);
    return response.data.features;
  }
  
  async trainModel(pair, config = {}) {
    const response = await this.client.post(`/api/train/${pair}`, config);
    return response.data;
  }
  
  async getModelStatus(pair) {
    const response = await this.client.get(`/api/models/${pair}/status`);
    return response.data;
  }
  
  // Enhanced trading signal with ML confidence
  async getEnhancedSignal(pair, technicalSignal) {
    const mlPrediction = await this.getPrediction(pair);
    
    // Combine technical and ML signals
    const agreement = this.calculateAgreement(technicalSignal, mlPrediction);
    const combinedConfidence = this.combineConfidence(
      technicalSignal.confidence, 
      mlPrediction.confidence,
      agreement
    );
    
    return {
      pair,
      technical: technicalSignal,
      ml: mlPrediction,
      combined: {
        signal: this.selectFinalSignal(technicalSignal, mlPrediction),
        confidence: combinedConfidence,
        agreement: agreement
      }
    };
  }
  
  calculateAgreement(technical, ml) {
    const techSignal = technical.suggestion?.toLowerCase();
    const mlSignal = ml.signal?.toLowerCase();
    
    if (techSignal === mlSignal) return 'strong';
    if ((techSignal === 'buy' && mlSignal === 'hold') || 
        (techSignal === 'sell' && mlSignal === 'hold') ||
        (techSignal === 'hold' && mlSignal !== 'hold')) return 'weak';
    return 'conflict';
  }
  
  combineConfidence(techConf, mlConf, agreement) {
    const baseConfidence = Math.max(techConf, mlConf);
    
    switch(agreement) {
      case 'strong': return Math.min(baseConfidence * 1.2, 1.0);
      case 'weak': return baseConfidence * 0.8;
      case 'conflict': return baseConfidence * 0.5;
      default: return baseConfidence;
    }
  }
  
  selectFinalSignal(technical, ml) {
    // High confidence ML prediction takes precedence
    if (ml.confidence > 0.8) {
      return ml.signal;
    }
    
    // High confidence technical signal
    if (technical.confidence > 0.7) {
      return technical.suggestion?.toUpperCase();
    }
    
    // Default to more conservative signal
    if (technical.suggestion === 'hold' || ml.signal === 'HOLD') {
      return 'HOLD';
    }
    
    // When both have medium confidence, prefer ML
    return ml.confidence > technical.confidence ? ml.signal : technical.suggestion?.toUpperCase();
  }
}

// Usage Example
const mlClient = new MLServiceClient();

async function runMLAnalysis() {
  // Check ML service health
  const health = await mlClient.getMLHealth();
  console.log('ML Service Status:', health.status);
  console.log('Core Connection:', health.core.status);
  
  // Get predictions for all pairs
  const predictions = await mlClient.getAllPredictions();
  console.log('ML Predictions:', predictions);
  
  // Detailed analysis for specific pair
  const rvnPrediction = await mlClient.getPrediction('RVN');
  const rvnFeatures = await mlClient.getFeatures('RVN');
  
  console.log('RVN ML Analysis:', {
    signal: rvnPrediction.signal,
    confidence: rvnPrediction.confidence,
    direction: rvnPrediction.direction,
    featureCount: rvnFeatures.count
  });
}
```

### Feature Engineering Example
```javascript
class CustomFeatureExtractor {
  constructor(mlClient) {
    this.mlClient = mlClient;
  }
  
  async extractCustomFeatures(pair, technicalData) {
    // Get standard ML features
    const mlFeatures = await this.mlClient.getFeatures(pair);
    
    // Add custom technical analysis features
    const customFeatures = {
      // Momentum indicators
      rsi_divergence: this.detectRSIDivergence(technicalData),
      macd_momentum: this.calculateMACDMomentum(technicalData),
      
      // Volume analysis
      volume_profile: this.analyzeVolumeProfile(technicalData),
      accumulation_distribution: this.calculateADLine(technicalData),
      
      // Market structure
      support_resistance: this.identifySupportResistance(technicalData),
      trend_strength: this.calculateTrendStrength(technicalData),
      
      // Cross-timeframe analysis
      higher_timeframe_trend: this.getHTFTrend(pair),
      correlation_strength: this.calculateCorrelation(pair)
    };
    
    return {
      standard: mlFeatures,
      custom: customFeatures,
      combined: [...mlFeatures.values, ...Object.values(customFeatures)]
    };
  }
  
  detectRSIDivergence(data) {
    // Implement RSI divergence detection
    const rsiValues = data.strategies.rsi.history || [];
    const prices = data.history.closes;
    
    if (rsiValues.length < 20 || prices.length < 20) return 0;
    
    // Compare recent RSI peaks with price peaks
    const recentRSI = rsiValues.slice(-10);
    const recentPrices = prices.slice(-10);
    
    const rsiTrend = this.calculateTrend(recentRSI);
    const priceTrend = this.calculateTrend(recentPrices);
    
    // Divergence when trends oppose
    return Math.sign(rsiTrend) !== Math.sign(priceTrend) ? 1 : 0;
  }
  
  calculateTrend(values) {
    if (values.length < 2) return 0;
    return values[values.length - 1] - values[0];
  }
}
```

---

## 🔬 Model Development & Training

### Training Pipeline
```javascript
class ModelTrainingPipeline {
  constructor() {
    this.mlClient = new MLServiceClient();
  }
  
  async trainAllModels(config = {}) {
    const defaultConfig = {
      epochs: 100,
      batchSize: 32,
      learningRate: 0.001,
      validationSplit: 0.2,
      earlyStoppingPatience: 10
    };
    
    const trainingConfig = { ...defaultConfig, ...config };
    
    // Get available pairs
    const health = await this.mlClient.getMLHealth();
    const pairs = ['RVN', 'XMR', 'BEL', 'DOGE', 'KAS', 'SAL'];
    
    const results = {};
    
    for (const pair of pairs) {
      try {
        console.log(`Starting training for ${pair}...`);
        
        // Start training
        const trainingResponse = await this.mlClient.trainModel(pair, trainingConfig);
        console.log(`Training initiated for ${pair}:`, trainingResponse.message);
        
        // Monitor training progress
        const modelStatus = await this.monitorTraining(pair);
        results[pair] = modelStatus;
        
      } catch (error) {
        console.error(`Training failed for ${pair}:`, error.message);
        results[pair] = { error: error.message };
      }
    }
    
    return results;
  }
  
  async monitorTraining(pair, maxWaitTime = 600000) { // 10 minutes max
    const startTime = Date.now();
    
    while (Date.now() - startTime < maxWaitTime) {
      const status = await this.mlClient.getModelStatus(pair);
      
      if (!status.modelInfo.isTraining) {
        return {
          pair,
          completed: true,
          modelInfo: status.modelInfo,
          trainingTime: Date.now() - startTime
        };
      }
      
      console.log(`${pair} still training... (${Math.round((Date.now() - startTime) / 1000)}s elapsed)`);
      await new Promise(resolve => setTimeout(resolve, 30000)); // Wait 30 seconds
    }
    
    return {
      pair,
      completed: false,
      timeout: true,
      trainingTime: maxWaitTime
    };
  }
  
  async evaluateModelPerformance(pair) {
    const prediction = await this.mlClient.getPrediction(pair);
    const features = await this.mlClient.getFeatures(pair);
    const modelStatus = await this.mlClient.getModelStatus(pair);
    
    return {
      pair,
      predictionQuality: {
        confidence: prediction.confidence,
        signal: prediction.signal,
        direction: prediction.direction
      },
      featureQuality: {
        featureCount: features.count,
        dataPoints: features.metadata.dataPoints,
        extractionTime: features.metadata.extractedAt
      },
      modelHealth: {
        hasModel: modelStatus.hasModel,
        parameters: modelStatus.modelInfo?.totalParams,
        isCompiled: modelStatus.modelInfo?.isCompiled
      }
    };
  }
}
```

### Model Validation & Testing
```javascript
class ModelValidator {
  async validatePredictionAccuracy(pair, testPeriodDays = 7) {
    const predictions = [];
    const actualResults = [];
    
    // Collect predictions and actual results over test period
    for (let i = 0; i < testPeriodDays; i++) {
      const prediction = await this.mlClient.getPrediction(pair);
      predictions.push({
        direction: prediction.direction,
        confidence: prediction.confidence,
        timestamp: Date.now()
      });
      
      // Wait for actual result (simplified for example)
      await new Promise(resolve => setTimeout(resolve, 24 * 60 * 60 * 1000)); // 24 hours
      
      const actualResult = await this.getActualPriceMovement(pair);
      actualResults.push(actualResult);
    }
    
    return this.calculateAccuracyMetrics(predictions, actualResults);
  }
  
  calculateAccuracyMetrics(predictions, actualResults) {
    if (predictions.length !== actualResults.length) {
      throw new Error('Prediction and result arrays must have same length');
    }
    
    let correct = 0;
    let highConfidenceCorrect = 0;
    let highConfidenceTotal = 0;
    
    for (let i = 0; i < predictions.length; i++) {
      const predicted = predictions[i].direction;
      const actual = actualResults[i] > 0 ? 'up' : 'down';
      
      if (predicted === actual) {
        correct++;
        
        if (predictions[i].confidence > 0.7) {
          highConfidenceCorrect++;
        }
      }
      
      if (predictions[i].confidence > 0.7) {
        highConfidenceTotal++;
      }
    }
    
    return {
      overallAccuracy: correct / predictions.length,
      highConfidenceAccuracy: highConfidenceTotal > 0 ? 
        highConfidenceCorrect / highConfidenceTotal : 0,
      totalPredictions: predictions.length,
      correctPredictions: correct,
      highConfidencePredictions: highConfidenceTotal
    };
  }
}
```

---

## 📚 Advanced Integration Patterns

### 1. **Multi-Model Ensemble**
```javascript
class MLEnsemble {
  constructor() {
    this.models = ['lstm', 'gru', 'transformer']; // Future model types
    this.weights = { lstm: 0.6, gru: 0.3, transformer: 0.1 };
  }
  
  async getEnsemblePrediction(pair) {
    // Currently only LSTM available, but structure for future models
    const lstmPrediction = await this.mlClient.getPrediction(pair);
    
    // Future: Add other model predictions
    // const gruPrediction = await this.getGRUPrediction(pair);
    // const transformerPrediction = await this.getTransformerPrediction(pair);
    
    return {
      ensemble: {
        signal: lstmPrediction.signal,
        confidence: lstmPrediction.confidence * this.weights.lstm,
        models: {
          lstm: lstmPrediction
          // gru: gruPrediction,
          // transformer: transformerPrediction
        }
      }
    };
  }
}
```

### 2. **Adaptive Learning**
```javascript
class AdaptiveLearning {
  async adaptModelToMarketConditions(pair) {
    const recentPerformance = await this.assessRecentPerformance(pair);
    
    if (recentPerformance.accuracy < 0.6) {
      console.log(`Model performance degraded for ${pair}, initiating retraining...`);
      
      // Trigger retraining with updated parameters
      const adaptedConfig = {
        epochs: 150,  // More epochs for difficult market conditions
        learningRate: 0.0005,  // Lower learning rate for stability
        batchSize: 16,  // Smaller batch size for better generalization
        dropout: 0.3   // Higher dropout to prevent overfitting
      };
      
      await this.mlClient.trainModel(pair, adaptedConfig);
    }
  }
  
  async assessRecentPerformance(pair, lookbackDays = 7) {
    // Simplified performance assessment
    const predictions = await this.getRecentPredictions(pair, lookbackDays);
    const accuracy = this.calculateAccuracy(predictions);
    
    return {
      accuracy,
      predictionCount: predictions.length,
      averageConfidence: predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length
    };
  }
}
```

---

## 🚨 Error Handling & Resilience

### Graceful Degradation
```javascript
class MLServiceFallback {
  constructor() {
    this.fallbackStrategies = {
      'core_unavailable': this.useCachedData,
      'model_error': this.useSimpleModel,
      'prediction_timeout': this.useLastKnownPrediction
    };
  }
  
  async getResilientPrediction(pair) {
    try {
      // Primary prediction attempt
      return await this.mlClient.getPrediction(pair);
      
    } catch (error) {
      console.warn(`Primary ML prediction failed for ${pair}:`, error.message);
      
      // Determine fallback strategy
      const errorType = this.classifyError(error);
      const fallbackStrategy = this.fallbackStrategies[errorType];
      
      if (fallbackStrategy) {
        return await fallbackStrategy(pair, error);
      }
      
      // Ultimate fallback: neutral prediction
      return {
        signal: 'HOLD',
        confidence: 0.1,
        direction: 'neutral',
        fallback: true,
        reason: error.message
      };
    }
  }
  
  classifyError(error) {
    if (error.message.includes('core')) return 'core_unavailable';
    if (error.message.includes('model')) return 'model_error';
    if (error.code === 'TIMEOUT') return 'prediction_timeout';
    return 'unknown';
  }
  
  async useCachedData(pair) {
    // Implementation for cached prediction fallback
    return {
      signal: 'HOLD',
      confidence: 0.3,
      direction: 'neutral',
      fallback: true,
      source: 'cache'
    };
  }
}
```

---

## 📊 Metrics & Analytics

### Performance Tracking
```javascript
class MLMetricsCollector {
  constructor() {
    this.metrics = {
      predictions: [],
      accuracy: {},
      latency: {},
      errors: []
    };
  }
  
  async collectMetrics(pair) {
    const startTime = Date.now();
    
    try {
      const prediction = await this.mlClient.getPrediction(pair);
      const latency = Date.now() - startTime;
      
      this.recordMetrics(pair, prediction, latency, null);
      return prediction;
      
    } catch (error) {
      this.recordMetrics(pair, null, Date.now() - startTime, error);
      throw error;
    }
  }
  
  recordMetrics(pair, prediction, latency, error) {
    const timestamp = Date.now();
    
    if (prediction) {
      this.metrics.predictions.push({
        pair,
        prediction,
        latency,
        timestamp
      });
      
      this.metrics.latency[pair] = this.metrics.latency[pair] || [];
      this.metrics.latency[pair].push(latency);
    }
    
    if (error) {
      this.metrics.errors.push({
        pair,
        error: error.message,
        timestamp
      });
    }
  }
  
  generateReport() {
    return {
      totalPredictions: this.metrics.predictions.length,
      averageLatency: this.calculateAverageLatency(),
      errorRate: this.calculateErrorRate(),
      pairPerformance: this.calculatePairPerformance(),
      timeRange: this.getTimeRange()
    };
  }
  
  calculateAverageLatency() {
    const allLatencies = Object.values(this.metrics.latency).flat();
    return allLatencies.length > 0 ? 
      allLatencies.reduce((sum, l) => sum + l, 0) / allLatencies.length : 0;
  }
}
```

---

## 🎯 Best Practices

### 1. **Prediction Confidence Thresholds**
```javascript
const CONFIDENCE_THRESHOLDS = {
  HIGH: 0.8,     // Strong signal, safe to act
  MEDIUM: 0.6,   // Moderate signal, use with caution
  LOW: 0.4,      // Weak signal, prefer technical analysis
  IGNORE: 0.3    // Very weak signal, ignore ML prediction
};

function interpretMLSignal(prediction) {
  if (prediction.confidence >= CONFIDENCE_THRESHOLDS.HIGH) {
    return { action: prediction.signal, weight: 1.0 };
  } else if (prediction.confidence >= CONFIDENCE_THRESHOLDS.MEDIUM) {
    return { action: prediction.signal, weight: 0.7 };
  } else if (prediction.confidence >= CONFIDENCE_THRESHOLDS.LOW) {
    return { action: prediction.signal, weight: 0.4 };
  } else {
    return { action: 'HOLD', weight: 0.0 };
  }
}
```

### 2. **Feature Quality Validation**
```javascript
function validateFeatureQuality(features) {
  const qualityChecks = {
    completeness: features.values.filter(v => v !== null && !isNaN(v)).length / features.count,
    outlierRate: features.values.filter(v => Math.abs(v) > 5).length / features.count,
    varianceCheck: calculateVariance(features.values) > 0.01
  };
  
  return {
    isValid: qualityChecks.completeness > 0.95 && 
             qualityChecks.outlierRate < 0.1 && 
             qualityChecks.varianceCheck,
    checks: qualityChecks
  };
}
```

### 3. **Model Health Monitoring**
```javascript
async function monitorModelHealth(pair) {
  const status = await mlClient.getModelStatus(pair);
  const recentPredictions = await getRecentPredictions(pair, 24); // Last 24 hours
  
  const healthMetrics = {
    modelLoaded: status.hasModel,
    modelCompiled: status.modelInfo?.isCompiled,
    predictionFrequency: recentPredictions.length,
    averageConfidence: recentPredictions.reduce((sum, p) => sum + p.confidence, 0) / recentPredictions.length,
    lastPredictionAge: Date.now() - Math.max(...recentPredictions.map(p => p.timestamp))
  };
  
  const isHealthy = healthMetrics.modelLoaded && 
                   healthMetrics.modelCompiled && 
                   healthMetrics.predictionFrequency > 0 &&
                   healthMetrics.averageConfidence > 0.3 &&
                   healthMetrics.lastPredictionAge < 300000; // 5 minutes
  
  return { isHealthy, metrics: healthMetrics };
}
```

---

## 📚 Additional Resources

### Related Documentation
- **Trading-Bot-Core Integration**: See `trading-bot-core/README.md`
- **Backtest ML Integration**: See `trading-bot-backtest/README.md`
- **Dashboard ML Visualization**: See `trading-bot-dashboard/README.md`

### TensorFlow.js Resources
- **TensorFlow.js Documentation**: https://www.tensorflow.org/js
- **LSTM Tutorial**: https://www.tensorflow.org/tutorials/text/text_generation
- **Time Series Prediction**: https://www.tensorflow.org/tutorials/structured_data/time_series

### Machine Learning References
- **Feature Engineering for Time Series**: Standard practices for financial data
- **LSTM Networks**: Understanding recurrent neural networks
- **Ensemble Methods**: Combining multiple model predictions

---

## 📊 Version Information

- **Current Version**: 1.0.0
- **TensorFlow.js Version**: ^4.22.0
- **Node.js Compatibility**: >=16.0.0
- **Last Updated**: January 2025
- **API Stability**: Production Ready

### Changelog
- **v1.0.0**: Initial release with LSTM models, feature extraction, and RESTful API
- **v0.x.x**: Development versions (deprecated)

---

## 🎯 Future Roadmap

### Planned Features
- **Multiple Model Types**: GRU, Transformer architectures
- **Hyperparameter Optimization**: Automated parameter tuning
- **Real-time Model Updates**: Continuous learning capabilities
- **Advanced Feature Engineering**: Automated feature selection
- **Model Ensemble**: Multiple model combination strategies

### Performance Goals
- **Sub-100ms Predictions**: Target <100ms prediction latency
- **70%+ Accuracy**: Target >70% directional accuracy
- **Auto-scaling**: Dynamic resource allocation
- **Real-time Training**: Online learning capabilities

---

*This technical manual serves as the complete reference for integrating with the trading-bot-ml service. The ML service enhances trading decisions by providing data-driven predictions with confidence scoring, enabling more sophisticated trading strategies when combined with technical analysis.*