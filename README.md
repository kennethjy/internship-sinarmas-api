Run ```uvicorn main_api:app```
## Other changes:
- remove line 16 ```pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'```
- add flash-attn to qwen model at line 220 ```attn_implementation="flash_attention_2",```
- install flash-attn with ```pip install flash-attn``` (for high-end devices)
- for ssh: run ```python -m uvicorn main_api:app --host 127.0.0.1 --port 8000``` instead
- then run ```ssh -N -L localhost:8001:localhost:8000 <ssh server here>``` on separate console
