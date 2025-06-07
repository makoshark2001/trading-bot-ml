const express = require("express");
const config = require("config");
const DataClient = require("../data/DataClient");
const FeatureExtractor = require("../data/FeatureExtractor");
const DataPreprocessor = require("../data/DataPreprocessor");
const LSTMModel = require("../models/LSTMModel");
const GRUModel = require("../models/GRUModel");
const CNNModel = require("../models/CNNModel");
const TransformerModel = require("../models/TransformerModel");
const ModelEnsemble = require("../models/ModelEnsemble");
const { Logger, MLStorage, TrainingQueueManager } = require("../utils");

class MLServer {
  constructor() {
    this.app = express();
    this.port = config.get("server.port") || 3001;
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

    // Model configuration - NOW READS FROM CONFIG
    this.modelTypes = ["lstm", "gru", "cnn", "transformer"];
    this.enabledModels = config.get("ml.ensemble.enabledModels") || [
      "lstm",
      "gru",
      "cnn",
    ];
    this.ensembleStrategy = config.get("ml.ensemble.strategy") || "weighted";

    // Performance optimization flags - ENSEMBLE MODE ENABLED
    this.quickMode = process.env.ML_QUICK_MODE === "true" || false; // Disable quick mode for ensemble
    this.cacheTimeout = config.get("ml.prediction.cacheTimeout") || 60000; // 60 second cache for ensemble

    // Initialize services first
    this.initializeServices();

    // Create training queue AFTER everything is initialized
    this.initializeTrainingQueue();
    this.initializePeriodicTraining();
    this.setupRoutes();
    this.setupMiddleware();
  }

  initializeServices() {
    Logger.info("Initializing ML services with ENSEMBLE MODELS enabled...", {
      enabledModels: this.enabledModels,
      quickMode: this.quickMode,
    });

    // Initialize data client
    this.dataClient = new DataClient();

    // Initialize feature extractor
    this.featureExtractor = new FeatureExtractor(config.get("ml.features"));

    // Initialize data preprocessor
    this.preprocessor = new DataPreprocessor(config.get("ml.models.lstm"));

    // Initialize advanced ML storage with NEW consolidated format
    this.mlStorage = new MLStorage({
      baseDir: config.get("ml.storage.baseDir"),
      saveInterval: config.get("ml.storage.saveInterval"),
      maxAgeHours: config.get("ml.storage.maxAgeHours"),
      enableCache: config.get("ml.storage.enableCache"),
    });

    Logger.info("ML services initialized successfully", {
      enabledModels: this.enabledModels,
      ensembleStrategy: this.ensembleStrategy,
      quickMode: this.quickMode,
      cacheTimeout: this.cacheTimeout,
      storageType: "consolidated",
    });
  }

  initializeTrainingQueue() {
    try {
      // Add a small delay to ensure Logger is fully ready
      setTimeout(() => {
        this.trainingQueue = new TrainingQueueManager({
          maxConcurrentTraining: 1, // Only 1 training at a time
          trainingCooldown: 1800000, // 30 minutes between training sessions
          processingInterval: 5000, // Check queue every 5 seconds
        });

        Logger.info("Training queue manager initialized successfully", {
          maxConcurrentTraining: 1,
          cooldownMinutes: 30,
        });
      }, 100); // 100ms delay
    } catch (error) {
      console.error("Failed to initialize training queue:", error.message);
      // Create a dummy training queue to prevent errors
      this.trainingQueue = {
        getQueueStatus: () => ({
          active: { count: 0, jobs: [] },
          queued: { count: 0, jobs: [] },
        }),
        addTrainingJob: () =>
          Promise.reject(new Error("Training queue not available")),
        canTrain: () => ({
          allowed: false,
          reason: "Training queue not initialized",
        }),
        cancelTraining: () => false,
        emergencyStop: () => ({ queuedJobsCancelled: 0, activeJobsMarked: 0 }),
        clearCooldown: () => false,
        clearAllCooldowns: () => 0,
        maxConcurrentTraining: 1,
        trainingCooldown: 1800000,
        shutdown: () => Promise.resolve(),
      };
    }
  }

  initializePeriodicTraining() {
    try {
      const periodicConfig = config.get("ml.training");

      if (!periodicConfig.periodicTraining) {
        Logger.info("Periodic training disabled in configuration");
        this.periodicTrainingEnabled = false;
        return;
      }

      this.periodicTrainingEnabled = true;
      this.periodicTrainingInterval =
        periodicConfig.periodicInterval || 3600000; // 1 hour default
      this.periodicTrainingConfig = {
        epochs: periodicConfig.periodicEpochs || 15,
        batchSize: periodicConfig.periodicTrainingConfig?.batchSize || 32,
        verbose: 0,
        priority: 5, // Lower priority than manual training
        maxAttempts: 1,
        source: "periodic",
        ...periodicConfig.periodicTrainingConfig,
      };

      // Start periodic training after service is fully initialized
      setTimeout(() => {
        this.startPeriodicTraining();
      }, 60000); // 1 minute delay to ensure everything is ready

      Logger.info("Periodic training initialized for ALL ENSEMBLE MODELS", {
        enabled: this.periodicTrainingEnabled,
        interval: this.periodicTrainingInterval / 1000 / 60 + " minutes",
        modelsToTrain: this.enabledModels,
        config: this.periodicTrainingConfig,
      });
    } catch (error) {
      Logger.error("Failed to initialize periodic training", {
        error: error.message,
      });
      this.periodicTrainingEnabled = false;
    }
  }

  startPeriodicTraining() {
    if (!this.periodicTrainingEnabled) {
      Logger.info("Periodic training not enabled, skipping start");
      return;
    }

    if (this.periodicTrainingTimer) {
      clearInterval(this.periodicTrainingTimer);
    }

    // Set up the periodic timer
    this.periodicTrainingTimer = setInterval(async () => {
      try {
        await this.performPeriodicTraining();
      } catch (error) {
        Logger.error("Periodic training cycle failed", {
          error: error.message,
        });
      }
    }, this.periodicTrainingInterval);

    Logger.info(
      "🔄 Periodic training started - ALL ENSEMBLE MODELS will be trained",
      {
        interval: this.periodicTrainingInterval / 1000 / 60 + " minutes",
        modelsPerCycle: this.enabledModels.length,
        nextTraining: new Date(
          Date.now() + this.periodicTrainingInterval
        ).toLocaleString(),
      }
    );

    // Optional: Run first training cycle after a short delay
    setTimeout(async () => {
      Logger.info("🚀 Running initial periodic training cycle...");
      try {
        await this.performPeriodicTraining();
      } catch (error) {
        Logger.error("Initial periodic training failed", {
          error: error.message,
        });
      }
    }, 300000); // 5 minutes after startup
  }

  async performPeriodicTraining() {
    if (!this.periodicTrainingEnabled) {
      Logger.debug("Periodic training disabled, skipping cycle");
      return;
    }

    const cycleStart = Date.now();
    Logger.info("🔄 Starting periodic training cycle for ALL ENSEMBLE MODELS", {
      enabledModels: this.enabledModels,
      timestamp: new Date().toLocaleString(),
    });

    try {
      // Get all active trading pairs
      const activePairs = await this.getActiveTradingPairs();

      if (activePairs.length === 0) {
        Logger.info("No active trading pairs found for periodic training");
        return;
      }

      Logger.info(
        `📊 Periodic training cycle for ${
          activePairs.length
        } pairs: ${activePairs.join(", ")}`
      );

      let totalQueued = 0;
      let totalSkipped = 0;
      let totalFailed = 0;
      const results = [];

      // Train ALL enabled models for each active pair
      for (const pair of activePairs) {
        const pairResults = {
          pair: pair,
          models: {},
        };

        for (const modelType of this.enabledModels) {
          try {
            // Check if we can train this model (respects cooldowns)
            const canTrain = this.trainingQueue.canTrain(pair, modelType);

            if (!canTrain.allowed) {
              Logger.debug(
                `Skipping periodic training for ${pair}:${modelType}`,
                {
                  reason: canTrain.reason,
                  cooldownRemaining: canTrain.cooldownRemainingMinutes,
                }
              );
              totalSkipped++;
              pairResults.models[modelType] = {
                status: "skipped",
                reason: canTrain.reason,
              };
              continue;
            }

            // Queue periodic training job
            const jobId = await this.trainingQueue.addTrainingJob(
              pair,
              modelType,
              this.performModelTraining.bind(this),
              this.periodicTrainingConfig
            );

            Logger.info(`✅ Periodic training queued: ${pair}:${modelType}`, {
              jobId,
            });
            totalQueued++;
            pairResults.models[modelType] = {
              status: "queued",
              jobId: jobId,
            };
          } catch (error) {
            Logger.warn(
              `❌ Failed to queue periodic training for ${pair}:${modelType}`,
              {
                error: error.message,
              }
            );
            totalFailed++;
            pairResults.models[modelType] = {
              status: "failed",
              error: error.message,
            };
          }
        }

        results.push(pairResults);
      }

      const cycleDuration = Date.now() - cycleStart;
      const nextCycle = new Date(Date.now() + this.periodicTrainingInterval);

      Logger.info("🎯 Periodic training cycle completed", {
        duration: Math.round(cycleDuration / 1000) + "s",
        pairs: activePairs.length,
        totalModelsAttempted: activePairs.length * this.enabledModels.length,
        totalQueued,
        totalSkipped,
        totalFailed,
        nextCycle: nextCycle.toLocaleString(),
        results: results,
      });

      // Get current queue status
      try {
        const queueStatus = this.trainingQueue.getQueueStatus();
        Logger.info("📊 Training queue status after periodic cycle", {
          activeJobs: queueStatus.active.count,
          queuedJobs: queueStatus.queued.count,
          totalJobs: queueStatus.active.count + queueStatus.queued.count,
        });
      } catch (error) {
        Logger.warn("Failed to get queue status", { error: error.message });
      }
    } catch (error) {
      Logger.error("Periodic training cycle failed", {
        error: error.message,
        duration: Math.round((Date.now() - cycleStart) / 1000) + "s",
      });
    }
  }

  async getActiveTradingPairs() {
    const activePairs = new Set();

    try {
      // Method 1: Get pairs from core service configuration
      const coreResponse = await this.dataClient.checkCoreHealth();
      if (coreResponse) {
        try {
          // Try to get config from core
          const configUrl = `${this.dataClient.baseUrl}/api/config`;
          const response = await require("axios").get(configUrl, {
            timeout: 10000,
          });
          if (response.data.config && response.data.config.pairs) {
            response.data.config.pairs.forEach((pair) =>
              activePairs.add(pair.toUpperCase())
            );
            Logger.debug("Got pairs from core config", {
              pairs: response.data.config.pairs,
            });
          }
        } catch (error) {
          Logger.debug("Could not get pairs from core config", {
            error: error.message,
          });
        }
      }
    } catch (error) {
      Logger.debug("Core service not available for pair discovery", {
        error: error.message,
      });
    }

    try {
      // Method 2: Get pairs from our feature counts (recently accessed pairs)
      const recentPairs = Object.keys(this.featureCounts);
      recentPairs.forEach((pair) => activePairs.add(pair.toUpperCase()));
      if (recentPairs.length > 0) {
        Logger.debug("Got pairs from feature counts", { pairs: recentPairs });
      }
    } catch (error) {
      Logger.debug("Could not get pairs from feature counts", {
        error: error.message,
      });
    }

    try {
      // Method 3: Get pairs from existing models
      const modelPairs = Object.keys(this.models);
      modelPairs.forEach((pair) => activePairs.add(pair.toUpperCase()));
      if (modelPairs.length > 0) {
        Logger.debug("Got pairs from existing models", { pairs: modelPairs });
      }
    } catch (error) {
      Logger.debug("Could not get pairs from models", { error: error.message });
    }

    try {
      // Method 4: Get pairs from ensemble models
      const ensemblePairs = Object.keys(this.ensembles);
      ensemblePairs.forEach((pair) => activePairs.add(pair.toUpperCase()));
      if (ensemblePairs.length > 0) {
        Logger.debug("Got pairs from ensembles", { pairs: ensemblePairs });
      }
    } catch (error) {
      Logger.debug("Could not get pairs from ensembles", {
        error: error.message,
      });
    }

    // Convert to array and filter out any invalid pairs
    const pairsArray = Array.from(activePairs).filter(
      (pair) => pair && typeof pair === "string" && pair.length >= 2
    );

    // If no pairs found, use fallback pairs
    if (pairsArray.length === 0) {
      const fallbackPairs = ["BTC", "ETH", "XMR", "RVN"];
      Logger.info("No active pairs found, using fallback pairs", {
        fallbackPairs,
      });
      return fallbackPairs;
    }

    Logger.debug("Active trading pairs discovered", {
      count: pairsArray.length,
      pairs: pairsArray,
    });

    return pairsArray;
  }

  stopPeriodicTraining() {
    if (this.periodicTrainingTimer) {
      clearInterval(this.periodicTrainingTimer);
      this.periodicTrainingTimer = null;
      Logger.info("🛑 Periodic training stopped");
    }
    this.periodicTrainingEnabled = false;
  }

  // Add periodic training status to health endpoint
  getPeriodicTrainingStatus() {
    if (!this.periodicTrainingEnabled) {
      return {
        enabled: false,
        status: "disabled",
      };
    }

    const nextTraining = this.periodicTrainingTimer
      ? new Date(Date.now() + this.periodicTrainingInterval)
      : null;

    return {
      enabled: true,
      status: this.periodicTrainingTimer ? "active" : "inactive",
      interval: this.periodicTrainingInterval,
      intervalMinutes: this.periodicTrainingInterval / 1000 / 60,
      nextTraining: nextTraining ? nextTraining.toISOString() : null,
      config: this.periodicTrainingConfig,
      modelsToTrain: this.enabledModels,
    };
  }

  setupMiddleware() {
    this.app.use(express.json());

    // CORS middleware
    this.app.use((req, res, next) => {
      res.header("Access-Control-Allow-Origin", "*");
      res.header(
        "Access-Control-Allow-Methods",
        "GET, POST, PUT, DELETE, OPTIONS"
      );
      res.header(
        "Access-Control-Allow-Headers",
        "Origin, X-Requested-With, Content-Type, Accept, Authorization"
      );
      if (req.method === "OPTIONS") {
        res.sendStatus(200);
      } else {
        next();
      }
    });

    // Request timing middleware
    this.app.use((req, res, next) => {
      req.startTime = Date.now();
      res.on("finish", () => {
        const duration = Date.now() - req.startTime;
        if (duration > 5000) {
          // Log slow requests
          Logger.warn("Slow request detected", {
            method: req.method,
            url: req.url,
            duration: `${duration}ms`,
            ip: req.ip,
          });
        }
      });
      next();
    });
  }

  setupRoutes() {
    // Health check with ensemble status
    this.app.get("/api/health", async (req, res) => {
      try {
        // Use cached health check if recent
        if (
          this.lastHealthCheck &&
          Date.now() - this.lastHealthCheck.timestamp < 10000
        ) {
          return res.json(this.lastHealthCheck);
        }

        // Get training queue status
        const queueStatus = this.trainingQueue.getQueueStatus();

        // Enhanced health check with ensemble info
        const healthData = {
          status: "healthy",
          service: "trading-bot-ml-ensemble-enabled",
          timestamp: Date.now(),
          uptime: this.getUptime(),
          ensembleMode: !this.quickMode,
          quickMode: this.quickMode,
          models: {
            individual: {
              loaded: Object.keys(this.models).length,
              pairs: Object.keys(this.models),
            },
            ensembles: {
              loaded: Object.keys(this.ensembles).length,
              pairs: Object.keys(this.ensembles),
            },
            enabledTypes: this.enabledModels,
            strategy: this.ensembleStrategy,
            featureCounts: this.featureCounts,
          },
          predictions: {
            cached: this.predictionCache.size,
            lastUpdate: this.getLastPredictionTime(),
          },
          training: {
            queue: queueStatus,
            maxConcurrent: this.trainingQueue.maxConcurrentTraining,
            cooldownMinutes: this.trainingQueue.trainingCooldown / 1000 / 60,
          },
          performance: {
            cacheHits: this.getCacheHitRate(),
            avgResponseTime: this.getAverageResponseTime(),
            cacheTimeout: this.cacheTimeout,
          },
          periodicTraining: this.getPeriodicTrainingStatus(),
        };

        // Try core health check in background (non-blocking)
        this.checkCoreHealthBackground().catch((err) => {
          Logger.warn("Background core health check failed", {
            error: err.message,
          });
        });

        this.lastHealthCheck = healthData;
        res.json(healthData);
      } catch (error) {
        Logger.error("Health check failed", { error: error.message });
        res.status(500).json({
          status: "unhealthy",
          service: "trading-bot-ml-ensemble-enabled",
          error: error.message,
          timestamp: Date.now(),
        });
      }
    });

    // ENHANCED prediction endpoint with full ensemble support
    this.app.get("/api/predictions/:pair", async (req, res) => {
      const requestStart = Date.now();

      try {
        const pair = req.params.pair.toUpperCase();
        const useEnsemble = req.query.ensemble !== "false"; // Default to ensemble
        const strategy = req.query.strategy || this.ensembleStrategy;
        const singleModel = req.query.model;

        // Create cache key
        const cacheKey = singleModel
          ? `${pair}_single_${singleModel}`
          : `${pair}_${useEnsemble ? "ensemble" : "single"}_${strategy}`;

        // Check cache first
        const cached = this.predictionCache.get(cacheKey);
        if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
          Logger.debug(`Cache hit for prediction: ${pair}`, {
            cacheAge: Date.now() - cached.timestamp,
            responseTime: Date.now() - requestStart,
          });

          return res.json({
            ...cached.data,
            cached: true,
            cacheAge: Date.now() - cached.timestamp,
            responseTime: Date.now() - requestStart,
          });
        }

        let prediction;
        if (singleModel) {
          // Specific model requested
          prediction = await this.getSingleModelPrediction(pair, singleModel);
        } else if (useEnsemble && !this.quickMode) {
          // Full ensemble prediction
          prediction = await this.getEnsemblePrediction(pair, { strategy });
        } else {
          // Single best model (LSTM)
          prediction = await this.getSingleModelPrediction(pair, "lstm");
        }

        // Cache the result
        this.predictionCache.set(cacheKey, {
          data: {
            pair,
            prediction,
            ensemble: useEnsemble && !singleModel,
            strategy: useEnsemble && !singleModel ? strategy : null,
            singleModel: singleModel || null,
            timestamp: Date.now(),
            responseTime: Date.now() - requestStart,
          },
          timestamp: Date.now(),
        });

        // Clean old cache entries
        this.cleanOldCacheEntries();

        res.json({
          pair,
          prediction,
          ensemble: useEnsemble && !singleModel,
          strategy: useEnsemble && !singleModel ? strategy : null,
          singleModel: singleModel || null,
          timestamp: Date.now(),
          cached: false,
          responseTime: Date.now() - requestStart,
        });

        // Save prediction to persistent storage (async, non-blocking)
        this.mlStorage
          .savePredictionHistory(pair, {
            ...prediction,
            timestamp: Date.now(),
            requestId: `${pair}_${Date.now()}`,
            useEnsemble: useEnsemble && !singleModel,
            strategy: useEnsemble && !singleModel ? strategy : null,
            singleModel: singleModel || null,
          })
          .catch((error) => {
            Logger.warn("Failed to save prediction history", {
              error: error.message,
            });
          });
      } catch (error) {
        const responseTime = Date.now() - requestStart;
        Logger.error(`Prediction failed for ${req.params.pair}`, {
          error: error.message,
          responseTime,
        });
        res.status(500).json({
          error: "Prediction failed",
          message: error.message,
          pair: req.params.pair.toUpperCase(),
          responseTime,
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
    this.app.get("/api/training/queue", (req, res) => {
      try {
        const queueStatus = this.trainingQueue.getQueueStatus();
        res.json({
          queue: queueStatus,
          timestamp: Date.now(),
        });
      } catch (error) {
        Logger.error("Failed to get training queue status", {
          error: error.message,
        });
        res.status(500).json({
          error: "Failed to get training queue status",
          message: error.message,
        });
      }
    });

    // Enhanced training endpoint - supports all model types
    this.app.post("/api/train/:pair/:modelType?", async (req, res) => {
      try {
        const pair = req.params.pair.toUpperCase();
        const modelType = req.params.modelType?.toLowerCase();
        const trainingConfig = req.body || {};

        // If no model type specified, train all enabled models
        const modelsToTrain = modelType ? [modelType] : this.enabledModels;

        const results = [];
        for (const model of modelsToTrain) {
          // Check if training is allowed
          const canTrain = this.trainingQueue.canTrain(pair, model);
          if (!canTrain.allowed) {
            results.push({
              pair,
              modelType: model,
              error: "Training not allowed",
              reason: canTrain.reason,
              details: canTrain,
            });
            continue;
          }

          try {
            // Add training job to queue
            const jobId = await this.trainingQueue.addTrainingJob(
              pair,
              model,
              this.performModelTraining.bind(this), // Bind training function
              {
                ...trainingConfig,
                priority: trainingConfig.priority || 5,
                maxAttempts: trainingConfig.maxAttempts || 2,
              }
            );

            results.push({
              pair,
              modelType: model,
              jobId,
              status: "queued",
            });
          } catch (error) {
            results.push({
              pair,
              modelType: model,
              error: error.message,
              status: "failed_to_queue",
            });
          }
        }

        res.json({
          message: `Training jobs processed for ${pair}`,
          results,
          queueStatus: this.trainingQueue.getQueueStatus(),
          timestamp: Date.now(),
        });
      } catch (error) {
        Logger.error(`Failed to process training for ${req.params.pair}`, {
          error: error.message,
        });
        res.status(500).json({
          error: "Failed to process training",
          message: error.message,
          pair: req.params.pair.toUpperCase(),
        });
      }
    });

    // Cancel training job
    this.app.delete("/api/training/job/:jobId", async (req, res) => {
      try {
        const jobId = req.params.jobId;
        const reason = req.body.reason || "User requested cancellation";

        const cancelled = await this.trainingQueue.cancelTraining(
          jobId,
          reason
        );

        if (cancelled) {
          res.json({
            message: "Training job cancelled",
            jobId,
            reason,
            timestamp: Date.now(),
          });
        } else {
          res.status(404).json({
            error: "Training job not found",
            jobId,
          });
        }
      } catch (error) {
        Logger.error(`Failed to cancel training job ${req.params.jobId}`, {
          error: error.message,
        });
        res.status(500).json({
          error: "Failed to cancel training job",
          message: error.message,
          jobId: req.params.jobId,
        });
      }
    });

    // Emergency stop all training
    this.app.post("/api/training/emergency-stop", (req, res) => {
      try {
        const result = this.trainingQueue.emergencyStop();

        Logger.warn("Emergency stop activated via API", result);

        res.json({
          message: "Emergency stop activated",
          result,
          timestamp: Date.now(),
        });
      } catch (error) {
        Logger.error("Failed to execute emergency stop", {
          error: error.message,
        });
        res.status(500).json({
          error: "Failed to execute emergency stop",
          message: error.message,
        });
      }
    });

    // Clear training cooldowns (admin function)
    this.app.post("/api/training/clear-cooldowns", (req, res) => {
      try {
        const pair = req.body.pair;
        const modelType = req.body.modelType;

        let result;
        if (pair && modelType) {
          result = this.trainingQueue.clearCooldown(pair, modelType);
          res.json({
            message: `Cooldown cleared for ${pair}:${modelType}`,
            cleared: result,
            timestamp: Date.now(),
          });
        } else {
          result = this.trainingQueue.clearAllCooldowns();
          res.json({
            message: "All cooldowns cleared",
            count: result,
            timestamp: Date.now(),
          });
        }
      } catch (error) {
        Logger.error("Failed to clear cooldowns", { error: error.message });
        res.status(500).json({
          error: "Failed to clear cooldowns",
          message: error.message,
        });
      }
    });
  }

  // Model-related routes
  setupModelRoutes() {
    // Enhanced model status endpoint with ensemble info
    this.app.get("/api/models/:pair/status", async (req, res) => {
      try {
        const pair = req.params.pair.toUpperCase();

        // Check cache first
        const cacheKey = `status_${pair}`;
        const cached = this.modelStatusCache.get(cacheKey);
        if (cached && Date.now() - cached.timestamp < 60000) {
          // 1 minute cache
          return res.json({
            ...cached.data,
            cached: true,
            cacheAge: Date.now() - cached.timestamp,
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
            modelInfo: model
              ? {
                  totalParams: model.model?.countParams?.() || 0,
                  layers: model.model?.layers?.length || 0,
                  isCompiled: model.isCompiled || false,
                  isTraining: model.isTraining || false,
                }
              : null,
            training: trainingStatus[modelType],
            hasWeights: this.mlStorage.hasTrainedWeights(pair, modelType),
          };
        }

        const statusData = {
          pair,
          featureCount: this.featureCounts[pair] || "unknown",
          individual: individualModels,
          ensemble: {
            hasEnsemble: !!ensemble,
            stats: ensemble
              ? {
                  modelCount: ensemble.models.size,
                  votingStrategy: ensemble.votingStrategy,
                  performanceHistorySize: ensemble.performanceHistory.size,
                }
              : null,
            strategy: this.ensembleStrategy,
            enabledModels: this.enabledModels,
            canCreateEnsemble: this.canCreateEnsemble(pair),
          },
          trainingQueue: this.trainingQueue.getQueueStatus(),
          ensembleMode: !this.quickMode,
          quickMode: this.quickMode,
          timestamp: Date.now(),
        };

        // Cache the result
        this.modelStatusCache.set(cacheKey, {
          data: statusData,
          timestamp: Date.now(),
        });

        res.json({
          ...statusData,
          cached: false,
        });
      } catch (error) {
        Logger.error(`Model status failed for ${req.params.pair}`, {
          error: error.message,
        });
        res.status(500).json({
          error: "Model status failed",
          message: error.message,
          pair: req.params.pair.toUpperCase(),
        });
      }
    });
  }

  // Utility routes
  setupUtilityRoutes() {
    // Fast features endpoint with caching
    this.app.get("/api/features/:pair", async (req, res) => {
      try {
        const pair = req.params.pair.toUpperCase();

        // Check cache first
        const cacheKey = `features_${pair}`;
        const cached = this.featureCache.get(cacheKey);
        if (cached && Date.now() - cached.timestamp < 300000) {
          // 5 minute cache
          return res.json({
            ...cached.data,
            cached: true,
            cacheAge: Date.now() - cached.timestamp,
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
            metadata: features.metadata,
          },
          timestamp: Date.now(),
          cached: false,
        };

        // Cache the result
        this.featureCache.set(cacheKey, {
          data: featureData,
          timestamp: Date.now(),
        });

        res.json(featureData);

        // Save to persistent cache (async, non-blocking)
        this.mlStorage
          .saveFeatureCache(pair, {
            count: features.features.length,
            names: features.featureNames,
            values: features.features.slice(0, 10),
            metadata: features.metadata,
          })
          .catch((error) => {
            Logger.warn("Failed to save feature cache", {
              error: error.message,
            });
          });
      } catch (error) {
        Logger.error(`Feature extraction failed for ${req.params.pair}`, {
          error: error.message,
        });
        res.status(500).json({
          error: "Feature extraction failed",
          message: error.message,
          pair: req.params.pair.toUpperCase(),
        });
      }
    });

    // Storage stats endpoint
    this.app.get("/api/storage/stats", (req, res) => {
      try {
        const stats = this.mlStorage.getStorageStats();
        res.json({
          storage: stats,
          performance: {
            predictionCacheSize: this.predictionCache.size,
            featureCacheSize: this.featureCache.size,
            modelStatusCacheSize: this.modelStatusCache.size,
          },
          trainingQueue: this.trainingQueue.getQueueStatus(),
          timestamp: Date.now(),
        });
      } catch (error) {
        Logger.error("Storage stats failed", { error: error.message });
        res.status(500).json({
          error: "Failed to get storage stats",
          message: error.message,
        });
      }
    });

    // Add these routes to the setupUtilityRoutes() method in MLServer.js
    // Add after the existing storage stats endpoint

    // REPLACE the existing POST /api/storage/migrate endpoint in MLServer.js with this:
        
        this.app.post('/api/storage/migrate', async (req, res) => {
            try {
                Logger.info('Starting storage migration to consolidated format via API');
                
                // Check for different migration method names
                let migrationMethod = null;
                if (typeof this.mlStorage.migrateLegacyData === 'function') {
                    migrationMethod = 'migrateLegacyData';
                } else if (typeof this.mlStorage.migrateFromOldFormat === 'function') {
                    migrationMethod = 'migrateFromOldFormat';
                } else {
                    // List available methods for debugging
                    const availableMethods = Object.getOwnPropertyNames(Object.getPrototypeOf(this.mlStorage))
                        .filter(name => typeof this.mlStorage[name] === 'function' && name.includes('migrat'));
                    
                    return res.status(501).json({
                        error: 'Migration not supported',
                        message: 'Migration functionality not available in current storage implementation',
                        availableMigrationMethods: availableMethods,
                        storageType: this.mlStorage.constructor.name,
                        allMethods: Object.getOwnPropertyNames(Object.getPrototypeOf(this.mlStorage))
                    });
                }
                
                Logger.info(`Using migration method: ${migrationMethod}`);
                const migrationResults = await this.mlStorage[migrationMethod]();
                
                Logger.info('Storage migration completed via API', migrationResults);
                
                res.json({
                    message: 'Storage migration completed',
                    method: migrationMethod,
                    results: migrationResults,
                    timestamp: Date.now()
                });
                
            } catch (error) {
                Logger.error('Storage migration failed via API', { error: error.message });
                res.status(500).json({
                    error: 'Migration failed',
                    message: error.message,
                    stack: error.stack,
                    timestamp: Date.now()
                });
            }
        });

    // Get detailed asset information
    this.app.get("/api/storage/asset/:pair", (req, res) => {
      try {
        const pair = req.params.pair.toUpperCase();

        if (typeof this.mlStorage.getAssetInfo !== "function") {
          return res.status(501).json({
            error: "Asset info not supported",
            message: "Asset information functionality not available",
          });
        }

        const assetInfo = this.mlStorage.getAssetInfo(pair);

        res.json({
          asset: assetInfo,
          timestamp: Date.now(),
        });
      } catch (error) {
        Logger.error(`Failed to get asset info for ${req.params.pair}`, {
          error: error.message,
        });
        res.status(500).json({
          error: "Failed to get asset information",
          message: error.message,
          pair: req.params.pair.toUpperCase(),
        });
      }
    });

    // Get list of all assets in consolidated storage
    this.app.get("/api/storage/assets", (req, res) => {
      try {
        const fs = require("fs");
        const assets = [];

        if (fs.existsSync(this.mlStorage.consolidatedDir)) {
          const files = fs.readdirSync(this.mlStorage.consolidatedDir);

          files.forEach((file) => {
            if (file.endsWith("_complete.json")) {
              const pair = file.replace("_complete.json", "").toUpperCase();
              const filePath = this.mlStorage.getAssetFilePath(pair);
              const fileStat = fs.statSync(filePath);

              try {
                const assetData = this.mlStorage.loadAssetData(pair);
                assets.push({
                  pair: pair,
                  file: file,
                  size: fileStat.size,
                  lastModified: fileStat.mtime.toISOString(),
                  modelsCount: Object.keys(assetData.models || {}).length,
                  trainingSessions:
                    assetData.training?.totalTrainingSessions || 0,
                  totalPredictions:
                    assetData.predictions?.totalPredictions || 0,
                  hasFeatureCache: !!assetData.features?.cache,
                });
              } catch (error) {
                assets.push({
                  pair: pair,
                  file: file,
                  size: fileStat.size,
                  lastModified: fileStat.mtime.toISOString(),
                  error: error.message,
                });
              }
            }
          });
        }

        res.json({
          assets: assets,
          count: assets.length,
          storageFormat: "CONSOLIDATED_SINGLE_FILE",
          consolidatedDir: this.mlStorage.consolidatedDir,
          timestamp: Date.now(),
        });
      } catch (error) {
        Logger.error("Failed to get assets list", { error: error.message });
        res.status(500).json({
          error: "Failed to get assets list",
          message: error.message,
        });
      }
    });

    // Force cleanup with consolidated storage
    this.app.post("/api/storage/cleanup", async (req, res) => {
      try {
        const maxAgeHours = req.body.maxAgeHours || this.mlStorage.maxAgeHours;

        Logger.info(
          `Starting consolidated storage cleanup via API, max age: ${maxAgeHours} hours`
        );

        const cleanedItems = await this.mlStorage.cleanup(maxAgeHours);

        Logger.info("Consolidated storage cleanup completed via API", {
          cleanedItems,
        });

        res.json({
          message: "Consolidated storage cleanup completed",
          cleanedItems: cleanedItems,
          maxAgeHours: maxAgeHours,
          timestamp: Date.now(),
        });
      } catch (error) {
        Logger.error("Consolidated storage cleanup failed via API", {
          error: error.message,
        });
        res.status(500).json({
          error: "Cleanup failed",
          message: error.message,
        });
      }
    });

    // Force save all cached data
    this.app.post("/api/storage/save", async (req, res) => {
      try {
        Logger.info("Force saving consolidated storage via API");

        const savedCount = await this.mlStorage.forceSave();

        Logger.info("Force save completed via API", { savedCount });

        res.json({
          message: "Consolidated storage force save completed",
          savedAssets: savedCount,
          timestamp: Date.now(),
        });
      } catch (error) {
        Logger.error("Force save failed via API", { error: error.message });
        res.status(500).json({
          error: "Force save failed",
          message: error.message,
        });
      }
    });

    // Storage health check
    this.app.get("/api/storage/health", (req, res) => {
      try {
        const fs = require("fs");
        const health = {
          status: "healthy",
          storageFormat: "CONSOLIDATED_SINGLE_FILE",
          baseDir: this.mlStorage.baseDir,
          consolidatedDir: this.mlStorage.consolidatedDir,
          directories: {
            baseExists: fs.existsSync(this.mlStorage.baseDir),
            consolidatedExists: fs.existsSync(this.mlStorage.consolidatedDir),
          },
          cache: {
            assetDataCacheSize: this.mlStorage.assetDataCache?.size || 0,
            enableCache: this.mlStorage.enableCache,
          },
          periodicSave: {
            enabled: !!this.mlStorage.saveIntervalId,
            interval: this.mlStorage.saveInterval,
          },
          trainedModels: this.mlStorage.getTrainedModelsList().length,
          timestamp: Date.now(),
        };

        // Check for any issues
        if (
          !health.directories.baseExists ||
          !health.directories.consolidatedExists
        ) {
          health.status = "warning";
          health.issues = ["Missing storage directories"];
        }

        res.json(health);
      } catch (error) {
        Logger.error("Storage health check failed", { error: error.message });
        res.status(500).json({
          status: "unhealthy",
          error: error.message,
          timestamp: Date.now(),
        });
      }
    });

    // Get training history for a specific pair
    this.app.get("/api/storage/training/:pair", (req, res) => {
      try {
        const pair = req.params.pair.toUpperCase();
        const trainingHistory = this.mlStorage.loadTrainingHistory(pair);

        if (!trainingHistory) {
          return res.status(404).json({
            error: "No training history found",
            pair: pair,
          });
        }

        res.json({
          ...trainingHistory,
          timestamp: Date.now(),
        });
      } catch (error) {
        Logger.error(`Failed to get training history for ${req.params.pair}`, {
          error: error.message,
        });
        res.status(500).json({
          error: "Failed to get training history",
          message: error.message,
          pair: req.params.pair.toUpperCase(),
        });
      }
    });

    // Get prediction history for a specific pair
    this.app.get("/api/storage/predictions/:pair", (req, res) => {
      try {
        const pair = req.params.pair.toUpperCase();
        const limit = parseInt(req.query.limit) || 100;
        const predictionHistory = this.mlStorage.loadPredictionHistory(pair);

        if (!predictionHistory) {
          return res.status(404).json({
            error: "No prediction history found",
            pair: pair,
          });
        }

        // Limit the number of predictions returned
        if (
          predictionHistory.predictions &&
          predictionHistory.predictions.length > limit
        ) {
          predictionHistory.predictions = predictionHistory.predictions
            .sort((a, b) => b.timestamp - a.timestamp)
            .slice(0, limit);
          predictionHistory.limited = true;
          predictionHistory.limit = limit;
        }

        res.json({
          ...predictionHistory,
          timestamp: Date.now(),
        });
      } catch (error) {
        Logger.error(
          `Failed to get prediction history for ${req.params.pair}`,
          { error: error.message }
        );
        res.status(500).json({
          error: "Failed to get prediction history",
          message: error.message,
          pair: req.params.pair.toUpperCase(),
        });
      }
    });

    // Updated API info endpoint
    this.app.get("/api", (req, res) => {
      res.json({
        service: "trading-bot-ml-ensemble-enabled",
        version: "2.1.0-ensemble-enabled",
        ensembleMode: !this.quickMode,
        quickMode: this.quickMode,
        features: [
          "ENSEMBLE MODELS ENABLED",
          "Multiple Model Types (LSTM, GRU, CNN)",
          "Weighted Ensemble Predictions",
          "Training Queue Management",
          "Intelligent Caching",
          "Pre-trained Weight Loading",
          "Performance Monitoring",
        ],
        endpoints: [
          "GET /api/health - Health check with ensemble status",
          "GET /api/predictions/:pair - Ensemble predictions (default) or single model",
          "GET /api/predictions/:pair?model=lstm - Specific model prediction",
          "GET /api/predictions/:pair?ensemble=false - Disable ensemble",
          "GET /api/predictions/:pair?strategy=weighted - Ensemble strategy",
          "GET /api/features/:pair - Feature extraction with caching",
          "GET /api/models/:pair/status - Model status with ensemble info",
          "GET /api/training/queue - Training queue status",
          "POST /api/train/:pair - Train all enabled models",
          "POST /api/train/:pair/:modelType - Train specific model",
          "DELETE /api/training/job/:jobId - Cancel training job",
          "POST /api/training/emergency-stop - Emergency stop all training",
          "POST /api/training/clear-cooldowns - Clear training cooldowns",
          "GET /api/storage/stats - Storage and performance statistics",
        ],
        ensemble: {
          enabledModels: this.enabledModels,
          strategy: this.ensembleStrategy,
          cacheTimeout: this.cacheTimeout,
          currentEnsembles: Object.keys(this.ensembles).length,
        },
        performance: {
          currentCacheSize: {
            predictions: this.predictionCache.size,
            features: this.featureCache.size,
            modelStatus: this.modelStatusCache.size,
          },
        },
        trainingQueue: {
          maxConcurrentTraining: this.trainingQueue.maxConcurrentTraining,
          cooldownMinutes: this.trainingQueue.trainingCooldown / 1000 / 60,
          currentStatus: this.trainingQueue.getQueueStatus(),
        },
        timestamp: Date.now(),
      });
    });
    // Get periodic training status
    this.app.get("/api/training/periodic/status", (req, res) => {
      try {
        const status = this.getPeriodicTrainingStatus();
        res.json({
          periodicTraining: status,
          timestamp: Date.now(),
        });
      } catch (error) {
        Logger.error("Failed to get periodic training status", {
          error: error.message,
        });
        res.status(500).json({
          error: "Failed to get periodic training status",
          message: error.message,
        });
      }
    });

    // Enable/disable periodic training
    this.app.post("/api/training/periodic/toggle", (req, res) => {
      try {
        const { enabled } = req.body;

        if (enabled && !this.periodicTrainingEnabled) {
          this.periodicTrainingEnabled = true;
          this.startPeriodicTraining();
          Logger.info("Periodic training enabled via API");
        } else if (!enabled && this.periodicTrainingEnabled) {
          this.stopPeriodicTraining();
          Logger.info("Periodic training disabled via API");
        }

        res.json({
          message: `Periodic training ${enabled ? "enabled" : "disabled"}`,
          status: this.getPeriodicTrainingStatus(),
          timestamp: Date.now(),
        });
      } catch (error) {
        Logger.error("Failed to toggle periodic training", {
          error: error.message,
        });
        res.status(500).json({
          error: "Failed to toggle periodic training",
          message: error.message,
        });
      }
    });

    // Force run periodic training cycle now
    this.app.post("/api/training/periodic/run-now", async (req, res) => {
      try {
        Logger.info("Manual periodic training cycle requested via API");

        // Run periodic training in background
        this.performPeriodicTraining().catch((error) => {
          Logger.error("Manual periodic training failed", {
            error: error.message,
          });
        });

        res.json({
          message: "Periodic training cycle started",
          status: this.getPeriodicTrainingStatus(),
          timestamp: Date.now(),
        });
      } catch (error) {
        Logger.error("Failed to start manual periodic training", {
          error: error.message,
        });
        res.status(500).json({
          error: "Failed to start periodic training",
          message: error.message,
        });
      }
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
      const binaryTargets =
        targets[`direction_${config.targetPeriods || 5}`] ||
        targets["direction_5"];

      if (!binaryTargets || binaryTargets.length === 0) {
        throw new Error("No training targets available");
      }

      // Prepare training data
      const featuresArray = Array(binaryTargets.length)
        .fill()
        .map(() => features.features);
      const processedData = await this.preprocessor.prepareTrainingData(
        featuresArray,
        binaryTargets
      );

      // Get or create model WITH PROPER COMPILATION
      const model = await this.getOrCreateModel(
        pair,
        modelType,
        currentFeatureCount
      );

      // 🔧 CRITICAL FIX: Ensure model is compiled before training
      if (!model.isCompiled) {
        Logger.warn(
          `Model ${pair}:${modelType} not compiled, compiling now...`
        );
        model.compileModel();
      }

      // Double check compilation
      if (!model.isCompiled) {
        throw new Error(`Failed to compile ${modelType} model for ${pair}`);
      }

      // Get model-specific training config
      const modelConfig = this.getModelConfig(modelType);
      const modelTrainingConfig = {
        epochs: config.epochs || modelConfig.epochs || 25,
        batchSize: config.batchSize || modelConfig.batchSize || 32,
        verbose: 0, // Silent training
        ...config,
      };

      Logger.info(`Starting training for ${pair}:${modelType}`, {
        isCompiled: model.isCompiled,
        config: modelTrainingConfig,
        featureCount: currentFeatureCount,
      });

      // Perform training
      const history = await model.train(
        processedData.trainX,
        processedData.trainY,
        processedData.validationX,
        processedData.validationY,
        modelTrainingConfig
      );

      // Save model weights if training was successful
      if (
        history.finalMetrics &&
        parseFloat(history.finalMetrics.finalAccuracy) > 0.5
      ) {
        try {
          await this.mlStorage.saveModelWeights(pair, modelType, model);
          Logger.info(`Model weights saved for ${pair}:${modelType}`);
        } catch (saveError) {
          Logger.warn(`Failed to save model weights for ${pair}:${modelType}`, {
            error: saveError.message,
          });
        }
      }

      const trainingResults = {
        pair: pair,
        modelType: modelType,
        status: "completed",
        finalMetrics: history.finalMetrics,
        epochsCompleted:
          history.epochsCompleted ||
          history.finalMetrics?.epochsCompleted ||
          history.epoch?.length ||
          0,
        featureCount: currentFeatureCount,
        timestamp: Date.now(),
      };

      // Clean up tensors
      processedData.trainX.dispose();
      processedData.trainY.dispose();
      processedData.validationX.dispose();
      processedData.validationY.dispose();
      processedData.testX.dispose();
      processedData.testY.dispose();

      // Save training history
      await this.mlStorage.saveTrainingHistory(
        pair,
        modelType,
        trainingResults
      );

      // Recreate ensemble if we have enough models
      await this.recreateEnsembleIfNeeded(pair);

      Logger.info(
        `Queued training completed for ${pair}:${modelType}`,
        trainingResults.finalMetrics
      );

      return trainingResults;
    } catch (error) {
      Logger.error(`Queued training failed for ${pair}:${modelType}`, {
        error: error.message,
      });
      throw error;
    }
  }

  // Get model configuration without using config.get
  getModelConfig(modelType) {
    const defaultConfigs = {
      lstm: {
        sequenceLength: 60,
        units: 50,
        layers: 2,
        epochs: 50,
        batchSize: 32,
        dropout: 0.2,
        learningRate: 0.001,
      },
      gru: {
        sequenceLength: 60,
        units: 50,
        layers: 2,
        epochs: 40,
        batchSize: 32,
        dropout: 0.2,
        learningRate: 0.001,
      },
      cnn: {
        sequenceLength: 60,
        filters: [32, 64, 128],
        epochs: 30,
        batchSize: 32,
        dropout: 0.3,
        learningRate: 0.001,
      },
      transformer: {
        sequenceLength: 60,
        epochs: 100,
        batchSize: 16,
        dropout: 0.1,
        learningRate: 0.001,
      },
    };

    return defaultConfigs[modelType] || defaultConfigs.lstm;
  }

  // FULL ENSEMBLE PREDICTION IMPLEMENTATION
  async getEnsemblePrediction(pair, options = {}) {
    const cacheKey = `${pair}_ensemble_${
      options.strategy || this.ensembleStrategy
    }`;

    // Check cache first
    if (
      this.predictions[cacheKey] &&
      Date.now() - this.predictions[cacheKey].timestamp < this.cacheTimeout
    ) {
      return this.predictions[cacheKey];
    }

    try {
      // Get or create ensemble for this pair
      let ensemble = this.ensembles[pair];
      if (!ensemble) {
        ensemble = await this.createEnsemble(pair);
        if (!ensemble) {
          Logger.warn(
            `No ensemble available for ${pair}, falling back to LSTM`
          );
          return this.getSingleModelPrediction(pair, "lstm");
        }
      }

      // Get data and extract features
      const pairData = await this.dataClient.getPairData(pair);
      const features = this.featureExtractor.extractFeatures(pairData);
      const currentFeatureCount = features.features.length;

      // Update feature count tracking
      this.featureCounts[pair] = currentFeatureCount;

      // Prepare input for prediction
      const inputData = await this.prepareRealTimeInput(features.features);

      // Make ensemble prediction
      const ensemblePrediction = await ensemble.predict(inputData, options);

      const result = {
        ...ensemblePrediction,
        metadata: {
          timestamp: Date.now(),
          version: "2.1.0-ensemble-enabled",
          type: "ensemble_prediction",
          featureCount: currentFeatureCount,
          ensembleMode: true,
        },
      };

      // Cache result
      this.predictions[cacheKey] = {
        ...result,
        timestamp: Date.now(),
        type: "ensemble",
      };

      // Clean up input tensor
      if (inputData && typeof inputData.dispose === "function") {
        inputData.dispose();
      }

      return this.predictions[cacheKey];
    } catch (error) {
      Logger.error(`Ensemble prediction failed for ${pair}`, {
        error: error.message,
      });

      // Fallback to single model
      Logger.info(`Falling back to single model for ${pair}`);
      return this.getSingleModelPrediction(pair, "lstm");
    }
  }

  // Single model prediction (optimized but not quick mode)
  async getSingleModelPrediction(pair, modelType = "lstm") {
    const cacheKey = `${pair}_${modelType}`;

    // Check cache first
    if (
      this.predictions[cacheKey] &&
      Date.now() - this.predictions[cacheKey].timestamp < this.cacheTimeout
    ) {
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
        model = await this.getOrCreateModel(
          pair,
          modelType,
          currentFeatureCount
        );
      }

      // Prepare input for prediction
      const inputData = await this.prepareRealTimeInput(features.features);

      // Make prediction
      const predictions = await model.predict(inputData);
      let prediction;

      // Handle TensorFlow tensor data properly
      if (Array.isArray(predictions)) {
        prediction = predictions[0];
      } else {
        prediction = predictions;
      }

      // Ensure we get a simple number, not a tensor object
      if (typeof prediction === "object" && prediction !== null) {
        // Handle tensor data format like {"0": 0.4962764084339142}
        if (typeof prediction[0] === "number") {
          prediction = prediction[0];
        } else if (prediction.dataSync) {
          // Handle actual TensorFlow tensor
          const data = prediction.dataSync();
          prediction = data[0];
        } else {
          // Fallback for unknown object format
          prediction = 0.5;
        }
      }

      const result = {
        prediction: prediction,
        confidence: Math.abs(prediction - 0.5) * 2,
        direction: prediction > 0.5 ? "up" : "down",
        signal: this.getTradeSignal(prediction, Math.abs(prediction - 0.5) * 2),
        modelType: modelType,
        individual: {
          prediction: prediction,
          confidence: Math.abs(prediction - 0.5) * 2,
        },
        metadata: {
          timestamp: Date.now(),
          version: "2.1.0-ensemble-enabled",
          type: "individual_prediction",
          featureCount: currentFeatureCount,
          ensembleMode: false,
        },
      };

      // Cache result
      this.predictions[cacheKey] = {
        ...result,
        timestamp: Date.now(),
        type: "individual",
      };

      // Clean up input tensor
      if (inputData && typeof inputData.dispose === "function") {
        inputData.dispose();
      }

      return this.predictions[cacheKey];
    } catch (error) {
      Logger.error(`Individual prediction failed for ${pair}:${modelType}`, {
        error: error.message,
      });

      // Return a fallback prediction
      return {
        prediction: 0.5,
        confidence: 0.1,
        direction: "neutral",
        signal: "HOLD",
        modelType: modelType,
        individual: {
          prediction: 0.5,
          confidence: 0.1,
        },
        metadata: {
          timestamp: Date.now(),
          version: "2.1.0-ensemble-enabled",
          type: "fallback_prediction",
          error: error.message,
          ensembleMode: false,
        },
      };
    }
  }

  // Create ensemble for a pair
  async createEnsemble(pair) {
    try {
      Logger.info(`Creating ensemble for ${pair}`, {
        enabledModels: this.enabledModels,
        strategy: this.ensembleStrategy,
      });

      // Check if we have enough models
      if (!this.models[pair]) {
        Logger.warn(`No models available for ${pair} ensemble`);
        return null;
      }

      const availableModels = Object.keys(this.models[pair]);
      if (availableModels.length < 2) {
        Logger.warn(`Insufficient models for ${pair} ensemble`, {
          available: availableModels.length,
          required: 2,
        });
        return null;
      }

      // Create ensemble
      const ensemble = new ModelEnsemble({
        modelTypes: this.enabledModels,
        weights: this.getDefaultWeights(),
        votingStrategy: this.ensembleStrategy,
      });

      // Add models to ensemble
      let modelsAdded = 0;
      for (const modelType of this.enabledModels) {
        if (this.models[pair][modelType]) {
          ensemble.addModel(
            modelType,
            this.models[pair][modelType],
            this.getModelWeight(modelType),
            { pair: pair, addedAt: Date.now() }
          );
          modelsAdded++;
        }
      }

      if (modelsAdded < 2) {
        Logger.warn(`Not enough models added to ensemble for ${pair}`, {
          modelsAdded,
          required: 2,
        });
        return null;
      }

      this.ensembles[pair] = ensemble;

      Logger.info(`Ensemble created for ${pair}`, {
        modelsAdded,
        strategy: this.ensembleStrategy,
        weights: ensemble.weights,
      });

      return ensemble;
    } catch (error) {
      Logger.error(`Failed to create ensemble for ${pair}`, {
        error: error.message,
      });
      return null;
    }
  }

  // Check if we can create an ensemble
  canCreateEnsemble(pair) {
    if (!this.models[pair]) return false;
    const availableModels = Object.keys(this.models[pair]);
    return availableModels.length >= 2;
  }

  // Recreate ensemble after training
  async recreateEnsembleIfNeeded(pair) {
    try {
      if (this.canCreateEnsemble(pair)) {
        // Dispose old ensemble if exists
        if (this.ensembles[pair]) {
          this.ensembles[pair].dispose();
          delete this.ensembles[pair];
        }

        // Create new ensemble
        const ensemble = await this.createEnsemble(pair);
        if (ensemble) {
          Logger.info(`Ensemble recreated for ${pair} after training`);
        }
      }
    } catch (error) {
      Logger.error(`Failed to recreate ensemble for ${pair}`, {
        error: error.message,
      });
    }
  }

  // Get default model weights
  getDefaultWeights() {
    const weights = {};
    this.enabledModels.forEach((modelType) => {
      weights[modelType] = this.getModelWeight(modelType);
    });
    return weights;
  }

  // Get weight for specific model type
  getModelWeight(modelType) {
    const weights = {
      lstm: 1.0, // Strong baseline
      gru: 0.9, // Slightly lower than LSTM
      cnn: 0.8, // Good for pattern recognition
      transformer: 0.7, // Complex but sometimes unstable
    };
    return weights[modelType] || 1.0;
  }

  // Create individual model - enhanced for ensemble use with FIXED COMPILATION
  async getOrCreateModel(pair, modelType, featureCount) {
    if (!this.models[pair]) {
      this.models[pair] = {};
    }

    // Check if model exists and has correct feature count
    if (this.models[pair][modelType]) {
      const existingModel = this.models[pair][modelType];
      if (existingModel.features === featureCount) {
        // 🔧 CRITICAL FIX: Ensure existing model is compiled
        if (!existingModel.isCompiled) {
          Logger.warn(
            `Existing model ${pair}:${modelType} not compiled, compiling now...`
          );
          existingModel.compileModel();
        }
        return existingModel; // Model is correct, return it
      } else {
        // Feature count mismatch, dispose and recreate
        if (existingModel.dispose) {
          existingModel.dispose();
        }
        delete this.models[pair][modelType];
      }
    }

    Logger.info(`Creating ${modelType} model for ${pair}`, { featureCount });

    // Create config with proper feature count
    const baseConfig = {
      sequenceLength: this.quickMode ? 30 : 60, // Full sequence in ensemble mode
      features: featureCount,
    };

    // Get model-specific config
    const modelSpecificConfig = this.getModelConfig(modelType);
    const finalConfig = {
      ...modelSpecificConfig,
      ...baseConfig,
    };

    // Try to load pre-trained weights first
    let model;
    const ModelClass = this.getModelClass(modelType);

    if (this.mlStorage.hasTrainedWeights(pair, modelType)) {
      try {
        model = await this.mlStorage.loadModelWeights(
          pair,
          modelType,
          ModelClass,
          finalConfig
        );
        if (model) {
          // 🔧 CRITICAL FIX: Ensure loaded model is compiled
          if (!model.isCompiled) {
            Logger.warn(
              `Loaded model ${pair}:${modelType} not compiled, compiling now...`
            );
            model.compileModel();
          }

          Logger.info(`Loaded pre-trained ${modelType} model for ${pair}`, {
            featureCount,
            params: model.model?.countParams?.() || 0,
            isCompiled: model.isCompiled,
          });
          model.features = featureCount;
          this.models[pair][modelType] = model;
          return model;
        }
      } catch (loadError) {
        Logger.warn(
          `Failed to load pre-trained weights for ${pair}:${modelType}`,
          {
            error: loadError.message,
          }
        );
      }
    }

    // Create new model if loading failed or no weights exist
    model = new ModelClass(finalConfig);
    model.buildModel();
    model.compileModel(); // 🔧 CRITICAL FIX: Always compile new models

    // 🔧 CRITICAL FIX: Verify compilation succeeded
    if (!model.isCompiled) {
      throw new Error(`Failed to compile new ${modelType} model for ${pair}`);
    }

    model.features = featureCount; // Store feature count for quick access

    this.models[pair][modelType] = model;

    Logger.info(`New ${modelType} model created for ${pair}`, {
      featureCount,
      params: model.model?.countParams?.() || 0,
      hasPretrainedWeights: false,
      isCompiled: model.isCompiled,
    });

    return model;
  }

  // Get model class by type
  getModelClass(modelType) {
    switch (modelType) {
      case "lstm":
        return LSTMModel;
      case "gru":
        return GRUModel;
      case "cnn":
        return CNNModel;
      case "transformer":
        return TransformerModel;
      default:
        throw new Error(`Unknown model type: ${modelType}`);
    }
  }

  // Enhanced input preparation
  async prepareRealTimeInput(features) {
    const tf = require("@tensorflow/tfjs");
    const sequenceLength = this.quickMode ? 30 : 60; // Full sequence for ensemble

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
      Logger.warn("Background core health check failed", {
        error: error.message,
      });
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
      modelStatus: this.modelStatusCache.size,
    };
  }

  getAverageResponseTime() {
    return "Not implemented";
  }

  // Get trade signal based on prediction and confidence
  getTradeSignal(prediction, confidence) {
    const strongThreshold = 0.7;
    const weakThreshold = 0.55;

    if (confidence > strongThreshold) {
      return prediction > 0.5 ? "STRONG_BUY" : "STRONG_SELL";
    } else if (confidence > weakThreshold) {
      return prediction > 0.5 ? "BUY" : "SELL";
    } else {
      return "HOLD";
    }
  }

  getUptime() {
    const uptimeMs = Date.now() - this.startTime;
    const hours = Math.floor(uptimeMs / 3600000);
    const minutes = Math.floor((uptimeMs % 3600000) / 60000);
    const seconds = Math.floor((uptimeMs % 60000) / 1000);
    return `${hours.toString().padStart(2, "0")}:${minutes
      .toString()
      .padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
  }

  getLastPredictionTime() {
    const times = Array.from(this.predictionCache.values()).map(
      (p) => p.timestamp
    );
    return times.length > 0 ? Math.max(...times) : null;
  }

  async start() {
    try {
      Logger.info(
        "Starting ML Server with 4-MODEL ENSEMBLE + CONSOLIDATED STORAGE..."
      );

      // Check if consolidation migration is needed
      try {
        const fs = require("fs");
        const path = require("path");
        const legacyWeightsDir = path.join(
          this.mlStorage.weightsDir ||
            path.join(this.mlStorage.baseDir, "weights")
        );

        if (
          fs.existsSync(legacyWeightsDir) &&
          typeof this.mlStorage.migrateLegacyData === "function"
        ) {
          Logger.info("🔄 Legacy storage detected, starting migration...");
          const migrationResults = await this.mlStorage.migrateLegacyData();
          Logger.info("✅ Migration completed", migrationResults);
        } else if (fs.existsSync(legacyWeightsDir)) {
          Logger.info("⚠️ Legacy storage detected but migration not available");
        } else {
          Logger.info("✅ Using consolidated storage, no migration needed");
        }
      } catch (migrationError) {
        Logger.warn("Migration check failed", {
          error: migrationError.message,
        });
      }

      // Wait for core service (not in quick mode anymore)
      await this.dataClient.waitForCoreService();

      // Start HTTP server
      this.server = this.app.listen(this.port, () => {
        Logger.info(
          `4-Model Ensemble ML Server with Consolidated Storage running at http://localhost:${this.port}`
        );
        console.log(
          `🚀 4-MODEL ENSEMBLE ML API available at: http://localhost:${this.port}/api`
        );
        console.log(
          `⚡ Health check: http://localhost:${this.port}/api/health`
        );
        console.log(
          `🎯 Ensemble predictions: http://localhost:${this.port}/api/predictions/BTC`
        );
        console.log(
          `🤖 LSTM model: http://localhost:${this.port}/api/predictions/BTC?model=lstm`
        );
        console.log(
          `🔄 GRU model: http://localhost:${this.port}/api/predictions/BTC?model=gru`
        );
        console.log(
          `📊 CNN model: http://localhost:${this.port}/api/predictions/BTC?model=cnn`
        );
        console.log(
          `🔮 Transformer model: http://localhost:${this.port}/api/predictions/BTC?model=transformer`
        );
        console.log(
          `📊 Model status: http://localhost:${this.port}/api/models/BTC/status`
        );
        console.log(
          `🔄 Training queue: http://localhost:${this.port}/api/training/queue`
        );
        console.log("");
        console.log("🤖 4-MODEL ENSEMBLE FEATURES ACTIVE:");
        console.log(
          `   • Ensemble Mode: ${!this.quickMode ? "ENABLED" : "DISABLED"}`
        );
        console.log(
          `   • Quick Mode: ${this.quickMode ? "ENABLED" : "DISABLED"}`
        );
        console.log(`   • Enabled Models: ${this.enabledModels.join(", ")}`);
        console.log(`   • Ensemble Strategy: ${this.ensembleStrategy}`);
        console.log(`   • Cache Timeout: ${this.cacheTimeout}ms`);
        console.log(
          `   • Max Concurrent Training: ${
            this.trainingQueue?.maxConcurrentTraining || "Not Ready"
          }`
        );
        console.log(
          `   • Training Cooldown: ${
            this.trainingQueue?.trainingCooldown
              ? this.trainingQueue.trainingCooldown / 1000 / 60 + " minutes"
              : "Not Ready"
          }`
        );
        console.log(`   • Consolidated Storage: ENABLED`);
        console.log(`   • Intelligent Caching: ENABLED`);
        console.log(
          `   • Training Queue: ${
            this.trainingQueue ? "ACTIVE" : "INITIALIZING"
          }`
        );
        console.log(`   • Storage Type: Consolidated Pair-Based`);
      });
    } catch (error) {
      Logger.error("Failed to start 4-Model Ensemble ML server", {
        error: error.message,
      });
      process.exit(1);
    }
  }

  async stop() {
    Logger.info("Stopping Ensemble ML Server...");

    this.stopPeriodicTraining();

    // Stop training queue first
    if (this.trainingQueue) {
      await this.trainingQueue.shutdown();
    }

    // Clear all caches
    this.predictionCache.clear();
    this.featureCache.clear();
    this.modelStatusCache.clear();

    // Dispose of all ensembles
    Object.values(this.ensembles).forEach((ensemble) => {
      if (ensemble && typeof ensemble.dispose === "function") {
        ensemble.dispose();
      }
    });

    // Dispose of all models
    Object.values(this.models).forEach((pairModels) => {
      Object.values(pairModels).forEach((model) => {
        if (model && typeof model.dispose === "function") {
          model.dispose();
        }
      });
    });

    // Shutdown storage system gracefully
    if (this.mlStorage) {
      await this.mlStorage.shutdown();
    }

    if (this.preprocessor) {
      this.preprocessor.dispose();
    }

    if (this.server) {
      this.server.close();
    }

    Logger.info("Ensemble ML Server stopped");
  }
}

module.exports = MLServer;
