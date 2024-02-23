#!/bin/bash

# Apply Namespace Configurations
kubectl apply -f deployments/kubernetes/namespaces/dev-namespace.yaml
kubectl apply -f deployments/kubernetes/namespaces/staging-namespace.yaml
kubectl apply -f deployments/kubernetes/namespaces/prod-namespace.yaml


echo "Namespaces created successfully."
