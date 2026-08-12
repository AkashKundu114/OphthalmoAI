from __future__ import annotations
import random
import time
from typing import Dict, Any, List

class FederatedNode:
    def __init__(self, node_id: str, institution: str):
        self.node_id = node_id
        self.institution = institution
        self.last_sync = None
        self.samples_trained = 0
        self.local_accuracy = 0.0

    def simulate_local_training(self) -> Dict[str, Any]:
        self.samples_trained = random.randint(100, 1500)
        self.local_accuracy = round(random.uniform(0.85, 0.94), 4)
        self.last_sync = time.time()

        param_gradients_norm = round(random.uniform(0.01, 0.08), 5)
        return {
            "node_id": self.node_id,
            "institution": self.institution,
            "samples": self.samples_trained,
            "accuracy": self.local_accuracy,
            "gradients_norm": param_gradients_norm,
        }

class FederatedServer:
    def __init__(self):
        self.nodes = [
            FederatedNode("NODE-NYU", "NYU Langone Eye Center"),
            FederatedNode("NODE-MOOR", "Moorfields Eye Hospital"),
            FederatedNode("NODE-STAN", "Stanford Byers Eye Institute"),
            FederatedNode("NODE-SANK", "Sankara Nethralaya"),
        ]
        self.global_round = 12
        self.global_accuracy = 0.915

    def trigger_aggregation_round(self) -> Dict[str, Any]:
        self.global_round += 1
        node_updates = []
        total_samples = 0
        weighted_acc = 0.0

        for node in self.nodes:
            update = node.simulate_local_training()
            node_updates.append(update)
            total_samples += update["samples"]
            weighted_acc += update["accuracy"] * update["samples"]


        self.global_accuracy = round(weighted_acc / total_samples, 4)


        param_drift = round(random.uniform(0.002, 0.015), 5)

        return {
            "global_round": self.global_round,
            "aggregated_accuracy": self.global_accuracy,
            "total_nodes_synced": len(self.nodes),
            "total_samples_aggregated": total_samples,
            "parameter_drift": param_drift,
            "node_updates": node_updates,
            "status": "success",
            "message": f"Federated aggregation round {self.global_round} finished via FedAvg protocol."
        }
