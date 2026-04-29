"""
Reproduction script for Figure 6b in the paper.
A10G + LLaMA-3.1-8B + Azure Code (AC) trace.
Latency vs request rate comparison between NEO and vLLM.
"""
import asyncio
import json
import os
from server import start_server, stop_server
from benchmark import run_test, prepare_real_test
from illustrator import draw_one_rl_diagram

# Tweak hyperparameters here:
vllm_rates = [0.5, 1.0, 1.5, 1.6, 1.7, 1.8]
ours_rates = [0.5, 1.0, 1.5, 1.6, 1.7, 1.8, 1.9]

cur_dir = os.path.dirname(os.path.realpath(__file__))
with open(f"{cur_dir}/configs/config-a10-8b.json", "r") as f:
    config = json.load(f)

async def one_round(server_name: str):
    start_server(server_name, config)
    try:
        if server_name == "ours":
            for rate in ours_rates:
                await run_test(*prepare_real_test("ac", config, server_name), rate=rate)
        if server_name == "vllm":
            for rate in vllm_rates:
                await run_test(*prepare_real_test("ac", config, server_name), rate=rate)
    finally:
        stop_server()
    await asyncio.sleep(30)

async def main():
    #await one_round("vllm")
    await one_round("ours")

if __name__ == "__main__":
    asyncio.run(main())
    draw_one_rl_diagram(
        title="fig6b",
        data_name="ac",
        sys_file_names=["vllm", "ours"],
        sys_legend_names=["VLLM", "Ours"],
        rate_lists=[vllm_rates, ours_rates],
        ylim=2,
        markers=["o", "x"],
        set_ylabel=True
    )
