# trading-bot-ml - Development Guide

**Repository**: https://github.com/makoshark2001/trading-bot-ml  
**Port**: 3001  
**Priority**: 2 (Depends on trading-bot-core)

## 🎯 Service Purpose

Machine learning prediction service providing LSTM neural network predictions and AI-enhanced trading signals. Integrates with trading-bot-core for technical analysis data.

## 💬 Chat Instructions for Claude

```
I'm building the ML prediction service for the trading bot. This service integrates with trading-bot-core to get technical analysis data and provides LSTM neural network predictions. The core service is already running on port 3000.

Key requirements:
- LSTM models for price prediction
- Feature engineering from technical indicators  
- RESTful API on port 3001
- Integration with core service at http://localhost:3000
- TensorFlow.js implementation
- 50+ features extracted from technical analysis

The core service provides technical indicators and market data. I need to enhance this with ML predictions.
```

## 📋 Implementation To-Do List

### ✅ Phase 2A: Project Setup & Integration

- [ ] **Project Infrastructure**
  - [ ] Initialize Node.js project: `npm init -y`
  - [ ] Install ML dependencies:
    ```bash
    npm install @tensorflow/tfjs-node axios express cors dotenv winston mathjs lodash
    npm install --save-dev jest nodemon
    ```
  - [ ] Create folder structure:
    ```
    src/
    ├── server/
    ├── ml/
    ├── features/
    ├── data/
    ├── routes/
    └── utils/
    models/
    config/
    logs/
    tests/
    ```
  - [ ] Create `.env.example` and configuration files

- [ ] **Core Service Integration**
  - [ ] File: `src/data/DataClient.js` - Core service communication
  - [ ] Implement health monitoring for core service
  - [ ] Add fallback mechanisms for core service outages
  - [ ] Test integration with http://localhost:3000/api/data

### ✅ Phase 2B: Feature Engineering

- [ ] **Feature Extraction Engine**
  - [ ] File: `src/features/FeatureExtractor.js` - Main feature extractor
  - [ ] **Price-Based Features (12 features)**:
    - Current price (normalized)
    - Price position in range
    - Returns for 5, 10, 20 periods
    - High-low spread and position
  - [ ] **Technical Indicator Features (33 features)**:
    - RSI features (value, confidence, overbought/oversold flags)
    - MACD features (line, signal, histogram)
    - Bollinger Bands features (%B, bandwidth, position)
    - Volume features (ratio, spike detection)
    - All 11 indicators with confidence scores
  - [ ] **Volume Features (4 features)**:
    - Current volume, MA ratio, trend
  - [ ] **Volatility Features (3 features)**:
    - Recent volatility, volatility ratio
  - [ ] **Time-Based Features (6 features)**:
    - Hour/day normalization with sine/cosine encoding

- [ ] **Data Preprocessing**
  - [ ] File: `src/data/DataPreprocessor.js` - Data preparation
  - [ ] Implement data normalization (feature scaling)
  - [ ] Add sequence generation for LSTM (60-step sequences)
  - [ ] Create train/validation/test splits
  - [ ] Handle missing data and outliers

### ✅ Phase 2C: LSTM Model Implementation

- [ ] **Model Architecture**
  - [ ] File: `src/ml/LSTMModel.js` - TensorFlow.js LSTM implementation
  - [ ] Model structure:
    ```javascript
    Input: [batch_size, 60, 52]  // 60 timesteps, 52 features
    LSTM Layer 1: 50 units, return_sequences=true, dropout=0.2
    LSTM Layer 2: 50 units, return_sequences=false, dropout=0.2
    Dense Layer: 32 units, activation='relu'
    Dropout: rate=0.2
    Output: 1 unit, activation='sigmoid'  // Probability of price increase
    ```

- [ ] **Training System**
  - [ ] File: `src/ml/ModelTrainer.js` - Training pipeline
  - [ ] Implement training data preparation
  - [ ] Add model training with validation
  - [ ] Create model saving/loading to `models/` directory
  - [ ] Add training performance metrics (accuracy, loss)
  - [ ] Implement early stopping and learning rate scheduling

- [ ] **Prediction Engine**
  - [ ] File: `src/ml/PredictionEngine.js` - Real-time inference
  - [ ] Real-time feature extraction from core service
  - [ ] Model inference with confidence scoring
  - [ ] Prediction caching (1-minute TTL)

### ✅ Phase 2D: ML API Implementation

- [ ] **API Routes** (Create in `src/routes/`)
  - [ ] `health.js` - GET /api/health (ML service + core connection status)
  - [ ] `predictions.js` - GET /api/predictions/:pair (individual prediction)
  - [ ] `predictions.js` - GET /api/predictions (all pair predictions)
  - [ ] `features.js` - GET /api/features/:pair (feature data for debugging)
  - [ ] `training.js` - POST /api/train/:pair (start model training)
  - [ ] `models.js` - GET /api/models/:pair/status (model information)

- [ ] **Server Setup**
  - [ ] File: `src/server/app.js` - Express server on port 3001
  - [ ] Add CORS for dashboard integration
  - [ ] Implement request logging
  - [ ] Add error handling middleware

### ✅ Phase 2E: Testing & Production

- [ ] **Testing Suite** (Create in `tests/`)
  - [ ] Unit tests for feature extraction
  - [ ] Model training/prediction tests
  - [ ] Core service integration tests
  - [ ] API endpoint tests
  - [ ] Performance benchmarks

- [ ] **Production Features**
  - [ ] Memory management for TensorFlow operations
  - [ ] Model versioning system
  - [ ] Graceful degradation when core service unavailable
  - [ ] Performance monitoring and logging

- [ ] **Documentation**
  - [ ] Complete README.md with ML-specific documentation
  - [ ] Feature engineering documentation
  - [ ] Model architecture documentation
  - [ ] Integration examples

## 📊 Key API Endpoints to Implement

```javascript
// ML service health
GET /api/health
Response: { 
  status: "healthy", 
  service: "trading-bot-ml",
  core: { status: "healthy", dataCollection: {...} },
  models: { loaded: 2, pairs: ["RVN", "XMR"] }
}

// Individual prediction
GET /api/predictions/:pair
Response: {
  pair: "RVN",
  prediction: {
    direction: "up",
    confidence: 0.742,
    probability: 0.742,
    signal: "BUY"
  }
}

// All predictions
GET /api/predictions
Response: {
  predictions: {
    "RVN": { direction: "up", confidence: 0.742, signal: "BUY" },
    "XMR": { direction: "down", confidence: 0.653, signal: "SELL" }
  }
}

// Feature data (debugging)
GET /api/features/:pair
Response: {
  pair: "RVN",
  features: {
    count: 52,
    names: ["price_currentPrice", "indicators_rsi_confidence", ...],
    values: [0.651, -0.234, 1.123, ...]
  }
}

// Start training
POST /api/train/:pair
Body: { epochs: 100, batchSize: 32, learningRate: 0.001 }
Response: { message: "Training started for RVN", timestamp: ... }

// Model status
GET /api/models/:pair/status
Response: {
  pair: "RVN",
  hasModel: true,
  modelInfo: { layers: 4, totalParams: 12847, isTraining: false }
}
```

## 🧠 LSTM Model Configuration

```javascript
{
  sequenceLength: 60,        // 60 time steps (5 hours of 5-min data)
  features: 52,              // Number of input features
  units: 50,                 // LSTM units per layer
  layers: 2,                 // Number of LSTM layers
  dropout: 0.2,              // Dropout rate
  learningRate: 0.001,       // Adam optimizer learning rate
  
  // Training config
  epochs: 100,
  batchSize: 32,
  validationSplit: 0.2,
  
  // Model architecture
  optimizer: 'adam',
  loss: 'binaryCrossentropy',
  metrics: ['accuracy']
}
```

## 🔧 Feature Engineering Details

### Feature Categories:
1. **Price Features (12)**: Current price, returns, position metrics
2. **Technical Indicators (33)**: All 11 indicators with confidence scores  
3. **Volume Features (4)**: Volume analysis and trends
4. **Volatility Features (3)**: Recent vs historical volatility
5. **Time Features (6)**: Cyclical time encoding

### Example Feature Extraction:
```javascript
{
  // Price features
  price_currentPrice: 0.0234,
  price_return_5: 0.023,
  price_return_10: -0.015,
  
  // Technical features  
  indicators_rsi_confidence: 0.65,
  indicators_macd_line: 0.000045,
  indicators_bollinger_percentB: 0.65,
  
  // Volume features
  volume_current_volume: 125000,
  volume_ma_ratio: 1.25,
  
  // Time features
  time_hour_sin: 0.707,
  time_day_cos: 0.901
}
```

## ⚙️ Configuration Requirements

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
```

## 🧪 Testing & Validation

```bash
# Test commands to implement
npm run test:data        # Core service integration
npm run test:features    # Feature extraction
npm run test:models      # LSTM model functionality
npm run test:all         # All ML tests

# Health verification
curl http://localhost:3001/api/health
curl http://localhost:3001/api/predictions/RVN
curl http://localhost:3001/api/features/RVN | jq '.features.count'
```

## 📈 Performance Benchmarks

- **Feature Extraction**: <500ms for 52 features
- **Model Prediction**: <200ms per pair
- **Training Time**: 5-15 minutes for 100 epochs
- **Memory Usage**: ~500MB during training, ~200MB during inference
- **Prediction Accuracy**: Target >65% for directional predictions

## 🔗 Integration Points

**Consumes from:**
- trading-bot-core (Port 3000) - Technical analysis and market data

**Provides to:**
- trading-bot-backtest (Port 3002) - ML predictions for strategy testing
- trading-bot-risk (Port 3003) - ML features for risk assessment
- trading-bot-execution (Port 3004) - Real-time predictions for trading
- trading-bot-dashboard (Port 3005) - ML predictions for visualization

## ✅ Success Criteria

**Phase 2A Complete When:**
- ML service connects successfully to core service
- Health checks show both services operational

**Phase 2B Complete When:**
- Feature extractor produces 52 features from core data
- Data preprocessing creates proper LSTM sequences

**Phase 2C Complete When:**
- LSTM model trains successfully on historical data
- Real-time predictions generate with confidence scores

**Phase 2D Complete When:**
- All API endpoints return properly formatted ML data
- Integration with core service is stable

**Phase 2E Complete When:**
- Model accuracy meets targets (>65% directional)
- Performance benchmarks achieved
- Memory usage optimized

---

*Save this file as `DEVELOPMENT_GUIDE.md` in the trading-bot-ml repository root*