const { Logger } = require('../utils');

class TrainingQueueManager {
    constructor(config = {}) {
        this.maxConcurrentTraining = config.maxConcurrentTraining || 1; // Only 1 training at a time by default
        this.trainingQueue = [];
        this.activeTraining = new Map(); // Track active training sessions
        this.trainingHistory = new Map(); // Track training history
        this.isProcessing = false;
        
        // Training cooldown periods
        this.trainingCooldown = config.trainingCooldown || 1800000; // 30 minutes between training sessions
        this.lastTrainingTimes = new Map(); // Track last training time per pair
        
        // Queue processing interval
        this.processingInterval = config.processingInterval || 5000; // Check queue every 5 seconds
        this.queueProcessor = null;
        
        // Use console.log initially to avoid circular dependency
        console.log('TrainingQueueManager initialized', {
            maxConcurrentTraining: this.maxConcurrentTraining,
            trainingCooldown: this.trainingCooldown / 1000 / 60 + ' minutes',
            processingInterval: this.processingInterval / 1000 + ' seconds'
        });
        
        // Import Logger after a delay to avoid circular dependency
        setTimeout(() => {
            try {
                const { Logger } = require('./index');
                this.Logger = Logger;
                this.Logger.info('TrainingQueueManager Logger initialized');
            } catch (error) {
                console.log('TrainingQueueManager: Logger not available, using console');
                this.Logger = {
                    info: console.log,
                    warn: console.warn,
                    error: console.error,
                    debug: console.log
                };
            }
        }, 500);
        
        this.startQueueProcessor();
    }

    // Helper method to safely use Logger
    log(level, message, meta = {}) {
        if (this.Logger && this.Logger[level]) {
            this.Logger[level](message, meta);
        } else {
            console[level === 'error' ? 'error' : 'log'](`[${level.toUpperCase()}] ${message}`, meta);
        }
    }
    
    // Add training job to queue
    async addTrainingJob(pair, modelType, trainingFunction, config = {}) {
        const jobId = `${pair}_${modelType}_${Date.now()}`;
        const priority = config.priority || 5; // 1-10, lower number = higher priority
        
        // Check if this pair/model is already in queue or training
        const existingJob = this.findExistingJob(pair, modelType);
        if (existingJob) {
            this.log('warn',`Training job already exists for ${pair}:${modelType}`, {
                existingJobId: existingJob.id,
                status: existingJob.status
            });
            return existingJob.id;
        }
        
        // Check cooldown period
        if (this.isInCooldown(pair, modelType)) {
            const cooldownRemaining = this.getCooldownRemaining(pair, modelType);
            this.log('info',`Training for ${pair}:${modelType} is in cooldown`, {
                cooldownRemaining: Math.round(cooldownRemaining / 1000 / 60) + ' minutes'
            });
            throw new Error(`Training cooldown active. ${Math.round(cooldownRemaining / 1000 / 60)} minutes remaining.`);
        }
        
        const job = {
            id: jobId,
            pair: pair.toUpperCase(),
            modelType: modelType.toLowerCase(),
            trainingFunction,
            config,
            priority,
            status: 'queued',
            queuedAt: Date.now(),
            attempts: 0,
            maxAttempts: config.maxAttempts || 2
        };
        
        // Insert job in priority order
        this.insertJobByPriority(job);
        
        this.log('info',`Training job queued: ${pair}:${modelType}`, {
            jobId,
            priority,
            queuePosition: this.trainingQueue.findIndex(j => j.id === jobId) + 1,
            queueSize: this.trainingQueue.length
        });
        
        return jobId;
    }
    
    // Insert job in queue by priority
    insertJobByPriority(job) {
        let inserted = false;
        for (let i = 0; i < this.trainingQueue.length; i++) {
            if (job.priority < this.trainingQueue[i].priority) {
                this.trainingQueue.splice(i, 0, job);
                inserted = true;
                break;
            }
        }
        
        if (!inserted) {
            this.trainingQueue.push(job);
        }
    }
    
    // Find existing job for pair/model
    findExistingJob(pair, modelType) {
        const pairUpper = pair.toUpperCase();
        const modelLower = modelType.toLowerCase();
        
        // Check active training
        for (const [jobId, job] of this.activeTraining.entries()) {
            if (job.pair === pairUpper && job.modelType === modelLower) {
                return { ...job, status: 'training' };
            }
        }
        
        // Check queue
        const queuedJob = this.trainingQueue.find(job => 
            job.pair === pairUpper && job.modelType === modelLower
        );
        
        return queuedJob;
    }
    
    // Check if pair/model is in cooldown
    isInCooldown(pair, modelType) {
        const key = `${pair.toUpperCase()}_${modelType.toLowerCase()}`;
        const lastTraining = this.lastTrainingTimes.get(key);
        
        if (!lastTraining) {
            return false;
        }
        
        return (Date.now() - lastTraining) < this.trainingCooldown;
    }
    
    // Get remaining cooldown time
    getCooldownRemaining(pair, modelType) {
        const key = `${pair.toUpperCase()}_${modelType.toLowerCase()}`;
        const lastTraining = this.lastTrainingTimes.get(key);
        
        if (!lastTraining) {
            return 0;
        }
        
        const elapsed = Date.now() - lastTraining;
        return Math.max(0, this.trainingCooldown - elapsed);
    }
    
    // Start queue processor
    startQueueProcessor() {
        if (this.queueProcessor) {
            clearInterval(this.queueProcessor);
        }
        
        this.queueProcessor = setInterval(() => {
            this.processQueue().catch(error => {
                this.log('error','Queue processing error', { error: error.message });
            });
        }, this.processingInterval);
        
        this.log('info','Training queue processor started');
    }
    
    // Stop queue processor
    stopQueueProcessor() {
        if (this.queueProcessor) {
            clearInterval(this.queueProcessor);
            this.queueProcessor = null;
            this.log('info','Training queue processor stopped');
        }
    }
    
    // Process training queue
    async processQueue() {
        if (this.isProcessing) {
            return; // Already processing
        }
        
        this.isProcessing = true;
        
        try {
            // Check if we can start new training
            if (this.activeTraining.size >= this.maxConcurrentTraining) {
                return; // Max concurrent training reached
            }
            
            // Get next job from queue
            const job = this.trainingQueue.shift();
            if (!job) {
                return; // No jobs in queue
            }
            
            // Double-check cooldown before starting
            if (this.isInCooldown(job.pair, job.modelType)) {
                this.log('warn',`Job ${job.id} skipped due to cooldown`, {
                    pair: job.pair,
                    modelType: job.modelType
                });
                
                // Put job back in queue with lower priority
                job.priority = Math.min(10, job.priority + 1);
                this.insertJobByPriority(job);
                return;
            }
            
            // Start training
            await this.startTraining(job);
            
        } finally {
            this.isProcessing = false;
        }
    }
    
    // Start individual training job
    async startTraining(job) {
        const startTime = Date.now();
        job.status = 'training';
        job.startedAt = startTime;
        job.attempts++;
        
        // Add to active training
        this.activeTraining.set(job.id, job);
        
        this.log('info',`Starting training: ${job.pair}:${job.modelType}`, {
            jobId: job.id,
            attempt: job.attempts,
            maxAttempts: job.maxAttempts,
            queuedFor: startTime - job.queuedAt + 'ms',
            activeTraining: this.activeTraining.size
        });
        
        try {
            // Execute training function
            const result = await job.trainingFunction(job.pair, job.modelType, job.config);
            
            // Training completed successfully
            const duration = Date.now() - startTime;
            job.status = 'completed';
            job.completedAt = Date.now();
            job.duration = duration;
            job.result = result;
            
            // Update cooldown timer
            const key = `${job.pair}_${job.modelType}`;
            this.lastTrainingTimes.set(key, Date.now());
            
            // Remove from active training
            this.activeTraining.delete(job.id);
            
            // Add to history
            this.trainingHistory.set(job.id, job);
            
            this.log('info',`Training completed: ${job.pair}:${job.modelType}`, {
                jobId: job.id,
                duration: Math.round(duration / 1000) + 's',
                success: true,
                activeTraining: this.activeTraining.size
            });
            
        } catch (error) {
            // Training failed
            const duration = Date.now() - startTime;
            job.status = 'failed';
            job.completedAt = Date.now();
            job.duration = duration;
            job.error = error.message;
            
            this.log('error',`Training failed: ${job.pair}:${job.modelType}`, {
                jobId: job.id,
                attempt: job.attempts,
                maxAttempts: job.maxAttempts,
                duration: Math.round(duration / 1000) + 's',
                error: error.message
            });
            
            // Retry if attempts remaining
            if (job.attempts < job.maxAttempts) {
                job.status = 'queued';
                job.priority = Math.min(10, job.priority + 2); // Lower priority for retries
                
                // Put back in queue with delay
                setTimeout(() => {
                    this.insertJobByPriority(job);
                    this.log('info',`Training job requeued for retry: ${job.pair}:${job.modelType}`, {
                        jobId: job.id,
                        attempt: job.attempts + 1,
                        maxAttempts: job.maxAttempts
                    });
                }, 30000); // 30 second delay before retry
            } else {
                // Max attempts reached, mark as permanently failed
                job.status = 'failed_permanent';
                this.trainingHistory.set(job.id, job);
            }
            
            // Remove from active training
            this.activeTraining.delete(job.id);
        }
    }
    
    // Cancel training job
    async cancelTraining(jobId, reason = 'User requested') {
        // Check if job is in queue
        const queueIndex = this.trainingQueue.findIndex(job => job.id === jobId);
        if (queueIndex !== -1) {
            const job = this.trainingQueue.splice(queueIndex, 1)[0];
            job.status = 'cancelled';
            job.cancelledAt = Date.now();
            job.cancelReason = reason;
            this.trainingHistory.set(jobId, job);
            
            this.log('info',`Training job cancelled from queue: ${jobId}`, { reason });
            return true;
        }
        
        // Check if job is actively training
        if (this.activeTraining.has(jobId)) {
            const job = this.activeTraining.get(jobId);
            job.status = 'cancelling';
            job.cancelReason = reason;
            
            this.log('warn',`Training job marked for cancellation: ${jobId}`, { 
                reason,
                note: 'Active training cannot be immediately stopped'
            });
            
            // Note: We can't actually stop TensorFlow training mid-process
            // The training will complete but won't be saved
            return true;
        }
        
        this.log('warn',`Training job not found for cancellation: ${jobId}`);
        return false;
    }
    
    // Get queue status
    getQueueStatus() {
        const activeJobs = Array.from(this.activeTraining.values()).map(job => ({
            id: job.id,
            pair: job.pair,
            modelType: job.modelType,
            status: job.status,
            startedAt: job.startedAt,
            duration: Date.now() - job.startedAt
        }));
        
        const queuedJobs = this.trainingQueue.map((job, index) => ({
            id: job.id,
            pair: job.pair,
            modelType: job.modelType,
            priority: job.priority,
            queuePosition: index + 1,
            queuedAt: job.queuedAt,
            queuedFor: Date.now() - job.queuedAt
        }));
        
        return {
            active: {
                count: activeJobs.length,
                maxConcurrent: this.maxConcurrentTraining,
                jobs: activeJobs
            },
            queued: {
                count: queuedJobs.length,
                jobs: queuedJobs
            },
            history: {
                total: this.trainingHistory.size,
                recent: this.getRecentHistory(10)
            },
            cooldowns: this.getCooldownStatus(),
            isProcessing: this.isProcessing
        };
    }
    
    // Get recent training history
    getRecentHistory(limit = 10) {
        const historyArray = Array.from(this.trainingHistory.values())
            .sort((a, b) => (b.completedAt || b.queuedAt) - (a.completedAt || a.queuedAt))
            .slice(0, limit);
        
        return historyArray.map(job => ({
            id: job.id,
            pair: job.pair,
            modelType: job.modelType,
            status: job.status,
            queuedAt: job.queuedAt,
            startedAt: job.startedAt,
            completedAt: job.completedAt,
            duration: job.duration,
            attempts: job.attempts,
            error: job.error
        }));
    }
    
    // Get cooldown status for all pairs/models
    getCooldownStatus() {
        const cooldowns = [];
        
        for (const [key, lastTraining] of this.lastTrainingTimes.entries()) {
            const remaining = this.trainingCooldown - (Date.now() - lastTraining);
            if (remaining > 0) {
                const [pair, modelType] = key.split('_');
                cooldowns.push({
                    pair,
                    modelType,
                    lastTraining,
                    cooldownRemaining: remaining,
                    cooldownRemainingMinutes: Math.round(remaining / 1000 / 60)
                });
            }
        }
        
        return cooldowns;
    }
    
    // Clear cooldown for specific pair/model (admin function)
    clearCooldown(pair, modelType) {
        const key = `${pair.toUpperCase()}_${modelType.toLowerCase()}`;
        const removed = this.lastTrainingTimes.delete(key);
        
        if (removed) {
            this.log('info',`Cooldown cleared for ${pair}:${modelType}`);
        }
        
        return removed;
    }
    
    // Clear all cooldowns (admin function)
    clearAllCooldowns() {
        const count = this.lastTrainingTimes.size;
        this.lastTrainingTimes.clear();
        
        this.log('info',`All cooldowns cleared`, { count });
        return count;
    }
    
    // Check if training is allowed
    canTrain(pair, modelType) {
        // Check if already training or queued
        const existingJob = this.findExistingJob(pair, modelType);
        if (existingJob) {
            return {
                allowed: false,
                reason: `Already ${existingJob.status}`,
                jobId: existingJob.id
            };
        }
        
        // Check cooldown
        if (this.isInCooldown(pair, modelType)) {
            const remaining = this.getCooldownRemaining(pair, modelType);
            return {
                allowed: false,
                reason: 'Cooldown active',
                cooldownRemaining: remaining,
                cooldownRemainingMinutes: Math.round(remaining / 1000 / 60)
            };
        }
        
        // Check capacity
        if (this.activeTraining.size >= this.maxConcurrentTraining && this.trainingQueue.length > 0) {
            return {
                allowed: true,
                reason: 'Will be queued',
                queuePosition: this.trainingQueue.length + 1
            };
        }
        
        return {
            allowed: true,
            reason: 'Can start immediately'
        };
    }
    
    // Cleanup old history entries
    cleanupHistory(maxAge = 7 * 24 * 60 * 60 * 1000) { // 7 days default
        const cutoff = Date.now() - maxAge;
        let cleaned = 0;
        
        for (const [jobId, job] of this.trainingHistory.entries()) {
            const jobTime = job.completedAt || job.queuedAt;
            if (jobTime < cutoff) {
                this.trainingHistory.delete(jobId);
                cleaned++;
            }
        }
        
        if (cleaned > 0) {
            this.log('info',`Cleaned up old training history`, { cleaned, remaining: this.trainingHistory.size });
        }
        
        return cleaned;
    }
    
    // Emergency stop all training
    emergencyStop() {
        this.log('warn','Emergency stop activated - clearing all training queues');
        
        // Clear queue
        const queuedCount = this.trainingQueue.length;
        this.trainingQueue.length = 0;
        
        // Mark active training as cancelled
        for (const [jobId, job] of this.activeTraining.entries()) {
            job.status = 'emergency_stopped';
            job.cancelledAt = Date.now();
            job.cancelReason = 'Emergency stop';
        }
        
        const activeCount = this.activeTraining.size;
        
        this.log('warn','Emergency stop completed', {
            queuedJobsCancelled: queuedCount,
            activeJobsMarked: activeCount
        });
        
        return { queuedJobsCancelled: queuedCount, activeJobsMarked: activeCount };
    }
    
    // Shutdown queue manager
    async shutdown() {
        this.log('info','Shutting down TrainingQueueManager...');
        
        this.stopQueueProcessor();
        
        // Wait for active training to complete (with timeout)
        const maxWait = 60000; // 1 minute max wait
        const startWait = Date.now();
        
        while (this.activeTraining.size > 0 && (Date.now() - startWait) < maxWait) {
            this.log('info',`Waiting for ${this.activeTraining.size} active training jobs to complete...`);
            await new Promise(resolve => setTimeout(resolve, 5000));
        }
        
        if (this.activeTraining.size > 0) {
            this.log('warn',`Forced shutdown with ${this.activeTraining.size} active training jobs`);
        }
        
        // Clear everything
        this.trainingQueue.length = 0;
        this.activeTraining.clear();
        
        this.log('info','TrainingQueueManager shutdown completed');
    }
}

module.exports = TrainingQueueManager;