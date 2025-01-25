-   Note: If pytorch is not found, then install with conda

# RUN API

> uvicorn api.main:app --reload

https://nr07p992-8000.inc1.devtunnels.ms/docs


pip3 install transformers pillow bitsandbytes accelerate fastapi uvicorn

# steps for scaling up the GCP cluster for demo purpose
gcloud container clusters create cpu-cluster --zone us-central1-c --num-nodes=1 --machine-type=e2-standard-4
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# steps to scale down the GCP cluster
gcloud container clusters delete cpu-cluster --zone us-central1-c

# Resize the nodes to scale up the pod in an existing cluster
gcloud container clusters resize cpu-cluster --node-pool default-pool --num-nodes=1 --zone us-central1-c

# Resize the nodes to scale down the pod in an existing cluster (This does not delete the cluster)
gcloud container clusters resize cpu-cluster --node-pool default-pool --num-nodes=0 --zone us-central1-c

# verify active cluster list
gcloud container clusters list

# create a static IP and reserve it
gcloud compute addresses create my-static-ip --region us-central1

# get the static IP resorved to connect android application with backend server
gcloud compute addresses list

# disable auto-scaling in a standard cluster
gcloud container node-pools update default-pool --cluster=cpu-cluster --zone=us-central1-c --no-enable-autoscaling

# descrive the cluster
gcloud container clusters describe cpu-cluster --zone us-central1-c 

# get list of node-pools in a cluster
gcloud container node-pools list --cluster cpu-cluster --zone us-central1-c 

# build a new backend image and push to GCR (update the tag everytime)
docker build -t gcr.io/glass-cedar-448715-c5/fastapi-app:v2 .
docker push gcr.io/glass-cedar-448715-c5/fastapi-app:v2


