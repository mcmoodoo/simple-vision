DATA_CENTER_ID := "US-IL-1"

default:
    @just --list

# commit and push everything so runpod can pull
push:
    git add . && git commit -m "pushing" && git push

# Create a pod --ports "8888/http, 3003/http, 22/tcp" 
create-pod:
    runpodctl create pod --gpuType "NVIDIA RTX 4090" --secureCloud --imageName "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04" --volumeSize 50 --containerDiskSize 100 --name shorts-maker --startSSH --dataCenterId {{DATA_CENTER_ID}} --env "ACCEPT_EULA=true"

# Get running pod ID
get-running-pod-id:
    runpodctl get pod | awk 'NR>1 {print $1}'

# Destroy the pod(s)
destroy-pod:
    runpodctl remove pod $(just get-running-pod-id)

