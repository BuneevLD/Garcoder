# Garcoder
Generic arithmetic codec implementation for CUDA

Data models supported:
- static binary
- adaptive binary
- context-adaptive binary
- context-adaptive byte

Context-adaptive byte model seems to give the best efficiency, while being marginally slower than others. Requires a lot of video memory.  
