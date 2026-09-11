# OphthalmoAI Kubernetes Deployment Guide

This directory contains the production-ready Kubernetes manifests for deploying OphthalmoAI on a cluster (e.g., Azure Kubernetes Service (AKS), Google Kubernetes Engine (GKE), Amazon EKS, or local Minikube/Kind).

---

## 1. Directory Structure

```
k8s/
├── namespace.yaml           # Creates the `ophthalmoai` namespace
├── configmap.yaml           # Application environment configuration
├── secret.example.yaml      # Secret template for sensitive credentials
├── backend-deployment.yaml  # Backend deployment (startupProbe, readiness, liveness)
├── backend-service.yaml     # ClusterIP service exposing backend on port 8000
├── frontend-deployment.yaml # Frontend Nginx web server deployment
├── frontend-service.yaml    # ClusterIP service exposing frontend on port 80
├── ingress.yaml             # Ingress with 25MB body size & 180s proxy timeout
├── hpa.yaml                 # HorizontalPodAutoscaler for backend pods
├── migration-job.yaml       # One-shot batch Job for Alembic DB migrations
└── kustomization.yaml       # Kustomize manifest bundle
```

---

## 2. Prerequisites

- A running Kubernetes cluster (v1.26+)
- `kubectl` configured with cluster admin context
- Container images built and pushed to a registry:
  ```bash
  # Build backend
  docker build -t <your-registry>/ophthalmoai-backend:latest -f backend/Dockerfile .
  docker push <your-registry>/ophthalmoai-backend:latest

  # Build frontend
  docker build -t <your-registry>/ophthalmoai-frontend:latest -f frontend/Dockerfile --build-arg VITE_API_URL=/api .
  docker push <your-registry>/ophthalmoai-frontend:latest
  ```
- An NGINX Ingress Controller installed in the cluster (or cloud provider ingress)

---

## 3. Step-by-Step Deployment

### Step 1: Create Namespace and Secrets

1. Create the `ophthalmoai` namespace:
   ```bash
   kubectl apply -f namespace.yaml
   ```

2. Provision the required production secrets:
   ```bash
   kubectl create secret generic ophthalmoai-secrets --namespace ophthalmoai \
     --from-literal=JWT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))") \
     --from-literal=GEMINI_API_KEY="your-gemini-api-key" \
     --from-literal=DATABASE_URL="postgresql+asyncpg://user:password@postgres-host:5432/ophthalmoai?ssl=require"
   ```
   *(Alternatively, copy `secret.example.yaml` to `secret.yaml`, fill in the values, and run `kubectl apply -f secret.yaml`)*

### Step 2: Configure Environment Settings

Review `configmap.yaml`:
- Set `CORS_ORIGINS` to your external ingress domain(s) (e.g. `https://ophthalmoai.yourdomain.com`).
- Ensure `FORCE_CPU` is `"true"` unless GPU nodes with NVIDIA runtime are attached.

### Step 3: Run Database Migrations

Before rolling out the backend pods, run the one-shot Alembic migration Job to ensure all database tables and schema versions are up to date:
```bash
kubectl apply -f migration-job.yaml
kubectl wait --for=condition=complete --timeout=120s job/ophthalmoai-migration -n ophthalmoai
```

### Step 4: Deploy All Workloads via Kustomize

Apply all manifests bundled in `kustomization.yaml`:
```bash
kubectl apply -k .
```

Verify rollout status:
```bash
kubectl -n ophthalmoai rollout status deployment/backend
kubectl -n ophthalmoai rollout status deployment/frontend
```

---

## 4. Key Architectural Configurations

### PyTorch Cold-Start Protection (`startupProbe`)
Loading deep learning model weights (`efficientnet_b4.pth`) into CPU memory during container boot can take 15–30 seconds. A `startupProbe` is configured on the backend deployment with a 150-second grace window (`failureThreshold: 30`, `periodSeconds: 5`), preventing Kubernetes from restarting initializing pods prematurely.

### Medical Image Uploads (Ingress Body Size)
Default NGINX Ingress limits payload size to 1MB. `ingress.yaml` includes:
```yaml
nginx.ingress.kubernetes.io/proxy-body-size: "25m"
nginx.ingress.kubernetes.io/proxy-read-timeout: "180"
```
This ensures high-resolution fundus retinal scans (5–20MB) and multi-pass Monte Carlo dropout inference succeed without HTTP 413 or 504 errors.

### Autoscaling
`hpa.yaml` automatically scales the backend from 1 to 5 replicas based on average CPU (75%) and memory (80%) utilization.
