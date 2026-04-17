python 3.11
Cuda 11.8

# Install PyTorch, Torchvision, and Torchaudio for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install ONNX Runtime with GPU support (ensure it aligns with CUDA 11.x)
pip install onnxruntime-gpu

# Rest of them 
pip install pygetwindow bettercam opencv-python pandas numpy pywin32