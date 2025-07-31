# LyraW4fp8

LyraW4fp8 is a library designed for W4AFP8 Mix-of-Experts (MoE) and GEMMs.

This library innovatively realized Machete Grouped-GEMM and MoE architecture, by optimizing the Machete GEMM kernels.

We further employed the tuning methodology for multi matrix shape to achieve peak acceleration.

We also integrated both MoE and Linear method into SGLang for W4AFP8-quantized models. The result simultaneously boost sustained throughput and cut time latency on a single H20 or H100.


# Model

We offered W4AFP8 AWQ quantized models :

- [DeepSeek-R1-AWQ-W4AFP8](https://huggingface.co/TMElyralab/DeepSeek-R1-AWQ-W4AFP8)
- [DeepSeek-R1-0528-AWQ-W4AFP8](https://huggingface.co/TMElyralab/DeepSeek-R1-0528-AWQ-W4AFP8)
- [DeepSeek-V3-0324-AWQ-W4AFP8](https://huggingface.co/TMElyralab/DeepSeek-V3-0324-AWQ-W4AFP8)

## Installation

Development build:

```bash
make build
```

If you want to clear history and rebuild,  try using `make rebuild`.

## Tuning

The tuning script searches for the fastest-performing structural schedule for multi matrix shape.

```bash
# for moe
python -m tune.tuning_fused_moe

# for gemm kernel
python -m tune.tuning_machete_mm
```

### Testing

The test scripts for MoE and GEMM are located in the `tests/` directory.

```bash
# for moe
pytest tests/test_fused_moe.py

# for gemm kernel
pytest tests/test_machete_mm.py
```


## SGLang Integration:

We have also integrated our implementation into SGLang.

Using DeepSeek-R1-AWQ-W4AFP8 as example：

```
python3 -m sglang.launch_server --model-path ${MODEL_PATH} --tp 8 --trust-remote-code --host 0.0.0.0 --port 8000 --mem-fraction-static 0.9 --quantization w4a8_machete --dtype half --cuda-graph-max-bs 128 --max-running-requests 128
```

Test condition:  input/output len = 1000/1000, qps=64, max_concurrency=64, num_prompt=128 on 1 single 8*H20 with tp8:

```
============ Serving Benchmark Result ============
Backend:                                 sglang       
Max request concurrency:                 64        
Successful requests:                     128       
Benchmark duration (s):                  105.50    
Total input tokens:                      128000    
Total generated tokens:                  128000    
Total generated tokens (retokenized):    127551    
Request throughput (req/s):              1.21      
Input token throughput (tok/s):          1213.24   
Output token throughput (tok/s):         1213.24   
Total token throughput (tok/s):          2426.49   
Concurrency:                             63.97     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   52728.31  
Median E2E Latency (ms):                 52728.33  
---------------Time to First Token----------------
Mean TTFT (ms):                          5444.26   
Median TTFT (ms):                        5425.69   
P99 TTFT (ms):                           8768.54   
---------------Inter-Token Latency----------------
Mean ITL (ms):                           47.33     
Median ITL (ms):                         44.18     
P95 ITL (ms):                            46.58     
P99 ITL (ms):                            46.76     
Max ITL (ms):                            7819.3
==================================================
```

Accuracy: AIME 2024 benchmark, attained accuracy of 78.3 %

Compared to the original DeepSeek-R1 model, throughput has increased by 56%

### baseline
```
python3 -m sglang.launch_server --model-path /path/to/DeepSeek-R1 --tp 8 --trust-remote-code --host 0.0.0.0 --port 8000 --mem-fraction-static 0.9 --cuda-graph-max-bs 128 --max-running-requests 128
```
```
============ Serving Benchmark Result ============
Backend:                                 sglang        
Max request concurrency:                 64        
Successful requests:                     128       
Benchmark duration (s):                  164.54    
Total input tokens:                      128000    
Total generated tokens:                  128000    
Total generated tokens (retokenized):    127694    
Request throughput (req/s):              0.78      
Input token throughput (tok/s):          777.91    
Output token throughput (tok/s):         777.91    
Total token throughput (tok/s):          1555.83   
Concurrency:                             52.88     
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   67975.09  
Median E2E Latency (ms):                 74667.77  
---------------Time to First Token----------------
Mean TTFT (ms):                          26348.64  
Median TTFT (ms):                        34605.67  
P99 TTFT (ms):                           47797.71  
---------------Inter-Token Latency----------------
Mean ITL (ms):                           41.67     
Median ITL (ms):                         40.17     
P95 ITL (ms):                            42.27     
P99 ITL (ms):                            43.59     
Max ITL (ms):                            15026.43  
==================================================
```

## Citation

We are TMElyralab, the Acceleration Team from Tencent Music Entertainment (TME).

```bibtex
@Misc{TMElyralab_2025,
  author =       {Sa Xiao, Mian Peng, Haoxiong Su, Kangjian Wu, Bin Wu, Yibo Lu, Qiwen Mao, Wenjiang Zhou},
  howpublished = {\url{https://github.com/TMElyralab}},
  year =         {2025}
}
```

## Acknowledgments

lyraW4afp8 is inspired by many open-source libraries, including (but not limited to)

- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [SGLang](https://github.com/sgl-project/sglang)
- [VLLM](https://github.com/vllm-project/vllm)

