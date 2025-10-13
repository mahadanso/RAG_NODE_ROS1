# RAG_NODE_ROS1
# Upanzi_rag_system
A rag system for the upanzi network at CMU-Africa

llama-server -hf ggml-org/SmolLM3-3B-GGUF --jinja --n-gpu-layers 1000

docker run -v ./chroma-data:/data -p 8000:8000 chromadb/chroma