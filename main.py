import torch; 
print('PyTorch version:', torch.__version__); 
print('CUDA available:', torch.cuda.is_available()); 
print('CUDA version:', torch.version.cuda); 
print('Number of GPUs:', torch.cuda.device_count()); 
print('Current GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')