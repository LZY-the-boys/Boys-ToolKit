
# OpenAI Server with VLLM Backend

This project provides a multi-process client and server setup to interact with OpenAI's API, including support for the VLLM backend. The setup includes data parallelism support for VLLM, which significantly improves generation speed.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.7+
- `pip`

### Install Dependencies

1. **OpenAI API (Old Version)**:
   ```bash
   pip install openai==0.28
   ```

2. **VLLM**:
   ```bash
   pip install vllm==0.5.3
   ```

## Usage

### Client

- **Gradio Client**: Use the `gradio_client.py` script for a simple interface.
- **Ray Client**: Use the `openai_client.py` script for multi-process support.

### Server

- **FastAPI Server**: Use the `fast_api.py` script for an HTTP server.
- **VLLM Server**: Use the `openai_server.py` script for the VLLM backend.

### Data Parallelism with VLLM

VLLM supports tensor parallelism and pipeline parallelism, but this project adds support for data parallelism. You can split data into multiple chunks and let multiple VLLM engines generate results simultaneously.

#### Example

```bash
bash eval.sh
```

## Performance

The table below shows the performance improvements with different configurations:

| Configuration                                      | Time  |
|----------------------------------------------------|-------|
| VLLM==0.5.3, data=4666                             |       |
| dp=8 + enable_chunked_prefill                      | 1:17  |
| tp=1                                               | 8:24  |
| tp=8                                               | 6:03  |
| pp=1 (Pipeline parallelism not supported)          | N/A   |
| tp=8 + enable_prefix_caching                       | Segfault |
| tp=8 + enable_chunked_prefill=True                 | 5:28  |
| tp=8 + enable_chunked_prefill + max_num_seqs 1024 + max_num_batched_tokens 1024 | 5:43 |
| tp=8 + enable_chunked_prefill + max_num_seqs 4096 + max_num_batched_tokens 4096 | 6:27 |
| max_num_batched_tokens (512) must be >= max_num_seqs (4096) | Error |
