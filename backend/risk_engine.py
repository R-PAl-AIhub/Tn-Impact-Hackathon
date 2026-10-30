from collections import deque
import time

class RiskEngine:
    def __init__(self,window_seconds = 300):
        self.window_seconds = window_seconds
        self.violation = deque()
    
    def add_violation(self,violation_type:str):
        self.violation.append(
            {
                "type":violation_type, 
                "time":time.time()
            }
        )
    
    def compute_risk_score(self):
        now = time.time()
        recent = [v for v in self.violation if now-v["time"]<=self.window_seconds]
        helmet_count = sum(v["type"] == "helmet" for v in recent)
        harness_count = sum(v["type"] == "harness" for v in recent)

        score = 100 - (helmet_count * 10) - (harness_count * 20)
        return max(score,0)
    
    def escalation_level(self):
        score = self.compute_risk_score()
        if  score < 50:
            return "CRITICAL"
        elif score < 70:
            return "HIGH"
        elif score < 90:
            return "WARNING"
        return "NORMAL"