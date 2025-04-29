# Makefile
K8S_DIR := k8-specs
NAMESPACE := vector-db

.PHONY: run delete restart status logs port-forward shell

# Apply all manifests
run:
	kubectl apply -f $(K8S_DIR)

# Delete all manifests
delete:
	kubectl delete -f $(K8S_DIR)

# Restart deployment (forces a rollout restart)
restart:
	kubectl rollout restart deployment/chromadb -n $(NAMESPACE)

# Check status of pods
status:
	kubectl get pods -n $(NAMESPACE)

# Stream logs from chromadb container
logs:
	kubectl logs -n $(NAMESPACE) deployment/chromadb -f

# Port forward to local machine
port-forward:
	kubectl port-forward -n $(NAMESPACE) svc/chromadb-service 8000:80

# Open a shell inside the chromadb pod
shell:
	kubectl exec -n $(NAMESPACE) -it $$(kubectl get pod -n $(NAMESPACE) -l app=chromadb -o jsonpath="{.items[0].metadata.name}") -- /bin/bash

# Validate YAML manifests
validate:
	kubectl apply --dry-run=client -f $(K8S_DIR)

# List services
services:
	kubectl get svc -n $(NAMESPACE)

# Describe a specific pod
describe:
	kubectl describe pod -n $(NAMESPACE) $$(kubectl get pod -n $(NAMESPACE) -l app=chromadb -o jsonpath="{.items[0].metadata.name}")
