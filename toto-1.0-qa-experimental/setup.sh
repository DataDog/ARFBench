# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.

. .venv/bin/activate

pip install "trl==0.25.1" deepspeed
pip install toto-ts
pip install "torch==2.8.0" "transformers==4.57.1"
pip install "peft==0.18.1"
pip install "qwen-vl-utils==0.0.14"
pip install "torchvision==0.23.0" # Required by qwen-vl-utils
pip install "lightning==2.6.1"