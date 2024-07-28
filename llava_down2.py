from huggingface_hub import snapshot_download
from huggingface_hub import hf_hub_download

# snapshot_download(repo_id="liuhaotian/llava-v1.6-mistral-7b", local_dir="./models/llava-mistral",
#                   local_dir_use_symlinks=False, ignore_patterns=["original/*"],)

hf_hub_download(repo_id="xtuner/llava-llama-3-8b-v1_1-gguf", filename="llava-llama-3-8b-v1_1-int4.gguf",
                   local_dir="./models/llava-llama",
                  local_dir_use_symlinks=False)

#cjpais/llava-1.6-mistral-7b-gguf filename: llava-v1.6-mistral-7b.Q4_K_M.gguf (4.37GB) / mmproj-model-f16.gguf (624MB)

# xtuner/llava-llama-3-8b-v1_1-gguf 
	# llava-llama-3-8b-v1_1-int4.gguf (4.92 GB) / llava-llama-3-8b-v1_1-mmproj-f16.gguf

# /home/jetson/cmap/athirdmapper/exp0610_ViT-B-16-SigLIP_3_copy/n_images/2.png
