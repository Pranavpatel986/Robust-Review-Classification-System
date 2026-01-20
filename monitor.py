import numpy as np

class SelfHealingMonitor:
    def __init__(self, threshold=0.2):
        self.error_log = []
        self.threshold = threshold

    def record_prediction(self, factors):
        # 0 = failure (no tags), 1 = success
        status = 1 if len(factors) > 0 else 0
        self.error_log.append(status)
        
        if len(self.error_log) >= 5:
            failure_rate = 1 - np.mean(self.error_log[-5:])
            if failure_rate > self.threshold:
                print("!!! ALERT: Concept Drift Detected. Triggering Model Refresh. !!!")

if __name__ == "__main__":
    print("Self-Healing Monitor Initialized.")