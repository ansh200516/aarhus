import os
import re
import time
import asyncio
import logging
import argparse
import numpy as np
from diskcache import Cache
from openai import AsyncOpenAI
from omegaconf import OmegaConf
from together import AsyncTogether
logger = logging.getLogger(__name__)

import sys
sys.path.append(os.getcwd())
from cachesaver.pipelines import OnlineAPI
from src.utils import tokens2cost, clean_log
from src.algorithms import *
from src.models import OnlineLLM, API
from src.typedefs import DecodingParameters
from src.tasks.humaneval import *

def build_method(method_name: str, params: DecodingParameters, api: API, config: OmegaConf):
    # Setup the method
    if method_name == "het_foa":
        step_agents = []

        # build the fleet of agents here
        step_agents.append({
            "agent": AgentReactHumanEval,
            "params": params,
            "num_agents": config.het_foa.num_agents - config.het_foa.num_agents // 2,
        })

        step_agents.append({
            "agent": AgentTerminalReflectHumanEval,
            "params": params,
            "num_agents": config.het_foa.num_agents // 2,
        })
        
        agents = AgentDictHeterogenousFOA(
            evaluate=AgentEvaluateHumanEval,
            eval_params=params,
            step_agents=step_agents
        )

        logger.info(f"Using these agents for Heterogenous FOA:")
        for i in range(len(agents["step_agents"])):
            logger.info(f"{step_agents[i]['agent'].__name__} ({agents['step_agents'][i]['num_agents']}): Temperature: {agents['step_agents'][i]['params'].temperature}, Top P: {agents['step_agents'][i]['params'].top_p}")

        method = AlgorithmHeterogenousFOA(
            model=api,
            agents=agents,
            env=EnvironmentHumanEval,
            num_agents=config.het_foa.num_agents,
            num_steps=config.het_foa.num_steps,
            k=config.het_foa.k,
            backtrack=config.het_foa.backtrack,
            resampling=config.het_foa.resampling,
            origin=config.het_foa.origin,
            min_steps=config.het_foa.min_steps,
            num_evaluations=config.het_foa.num_evaluations,
        )
    elif method_name == "foa":
        agents = AgentDictFOA(
            step=AgentActHumanEval,
            evaluate=AgentEvaluateHumanEval,
            step_params=params,
            eval_params=params,
        )
        method = AlgorithmFOA(
            model=api,
            agents=agents,
            env=EnvironmentHumanEval,
            num_agents=config.foa.num_agents,
            num_steps=config.foa.num_steps,
            k=config.foa.k,
            backtrack=config.foa.backtrack,
            resampling=config.foa.resampling,
            origin=config.foa.origin,
            min_steps=config.foa.min_steps,
            num_evaluations=config.foa.num_evaluations,
        )
    elif method_name == "tot_bfs":
        agents = AgentDictTOT(
            step=AgentBfsHumanEval,
            evaluate=AgentEvaluateHumanEval,
            step_params=params,
            eval_params=params,
        )
        method = AlgorithmTOT(
            model=api,
            agents=agents,
            env=EnvironmentHumanEval,
            num_selections=config.tot_bfs.num_selections,
            num_steps=config.tot_bfs.num_steps,
            num_evaluations=config.tot_bfs.num_evaluations,
        )
    elif method_name == "got":
        agents = AgentDictGOT(
            step=AgentActHumanEval,
            aggregate=AgentAggregateHumanEval,
            evaluate=AgentEvaluateHumanEval,
            step_params=params,
            aggregate_params=params,
            eval_params=params,
        )
        method = AlgorithmGOT(
            model=api,
            agents=agents,
            env=EnvironmentHumanEval,
            num_selections=config.got.num_selections,
            num_steps=config.got.num_steps,
            num_generate=config.got.num_generate,
            num_best=config.got.num_best,
            num_evaluations=config.got.num_evaluations,
        )
    else:
        raise NotImplementedError(f"Method {method_name} is not implemented yet.")
    return method

async def run(args, trial, cache_path):
    # Cache to be used
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache = Cache(cache_path)

    # LLM Provider
    if args.provider == "openai":
        if args.base_url and "localhost" in args.base_url:
            # For local vLLM servers, use a dummy API key
            client = AsyncOpenAI(base_url=args.base_url, api_key="dummy-key")
        else:
            client = AsyncOpenAI(base_url=args.base_url) if args.base_url else AsyncOpenAI()
    elif args.provider == "together":
        client = AsyncTogether()
    elif args.provider == "local":
        raise NotImplementedError("Local client is not implemented yet.")
    else:
        raise ValueError("Invalid provider. Choose 'openai', 'together', or 'local'.")
    
    # CacheSaver model layer
    if args.provider in ["openai", "together"]:
        model = OnlineLLM(client=client)
    else:
        raise NotImplementedError("Local model is not implemented yet.")
    
    # CacheSaver Pipeline: Batcher -> Reorderer -> Deduplicator -> Cache -> Model
    pipeline = OnlineAPI(
                    model=model,
                    cache=cache,
                    batch_size=args.batch_size,
                    timeout=args.timeout,
                    allow_batch_overflow=True,
                    correctness=bool(args.correctness)
                    )
    
    # Cachesaver additional layer for wrapping: API -> Pipeline
    api = API(
        pipeline=pipeline,
        model=args.model
    )

    # Decoding parameters
    params = DecodingParameters(
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        top_p=args.top_p,
        stop=args.stop,
        logprobs=args.logprobs
    )

    # Config for framework hyperpaarameters
    config = OmegaConf.load(args.conf_path)

    # Build the method
    method = build_method(args.method, params, api, config)

    # Load the dataset
    benchmark = BenchmarkHumanEval(path=args.dataset_path, split=args.split)

    # Run the method
    start = time.time()
    results = await method.benchmark(
        benchmark=benchmark,
        share_ns=True,
        cache=args.value_cache,
    )
    end = time.time()

    finished = []
    correct = []
    for result in results:
        evaluations = sorted([EnvironmentHumanEval.evaluate(state) for state in result], key=lambda x: x[1])
        finished.append(False if len(evaluations) == 0 else evaluations[-1][0])
        correct.append(1.0 if len(evaluations) == 0 else evaluations[-1][1])
    perc_finished = sum(finished) / len(finished)
    perc_correct = sum(correct) / len(correct)
    costs = {key:tokens2cost(api.tokens[key], args.model)["total"] for key in api.tokens.keys()}
    latency = {
        "mean": np.mean(api.latencies), 
        "std": np.std(api.latencies),
        "max": np.max(api.latencies), 
        "min": np.min(api.latencies), 
        "total": np.sum(api.latencies)
        }
    reuse = {
        "mean": np.mean(list(api.reuse.values())),
        "std": np.std(list(api.reuse.values())),
        "max": np.max(list(api.reuse.values())),
        "min": np.min(list(api.reuse.values())),
        "median": np.median(list(api.reuse.values())),
    }
    run_time = end - start
    throughput = len(benchmark) / run_time

    logger.info(f"Finished: {perc_finished:.2f} (trial {trial})")
    logger.info(f"Correct: {perc_correct:.2f} (trial {trial})")
    logger.info(f"Costs: {costs} (trial {trial})")
    logger.info(f"Latency: {latency['mean']} (trial {trial})")
    logger.info(f"Run time: {run_time:.2f} seconds (trial {trial})")
    logger.info(f"Throughput: {throughput:.2f} puzzles/second (trial {trial})")

    logger.info(f"Correct (deailed): {correct} (trial {trial})")
    logger.info(f"Tokens (detailed): {api.tokens} (trial {trial})")
    logger.info(f"Calls (detailed): {api.calls} (trial {trial})")
    logger.info(f"Reuse (detailed): {reuse} (trial {trial})")
    
    print("All good.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve Human Eval using LLMs.")
    parser.add_argument("--provider", type=str,default="openai", help="LLM provider")
    parser.add_argument("--base_url", type=str,default=None, help="Base URL for the API")
    parser.add_argument("--model", type=str,default="gpt-3.5-turbo", help="LLM model")
    parser.add_argument("--batch_size", type=int,default=10, help="CacheSaver's batch size")
    parser.add_argument("--timeout", type=float, default=1,help="CacheSaver's timeout")
    parser.add_argument("--temperature", type=float,default=0.7, help="Temperature for the model")
    parser.add_argument("--max_completion_tokens", type=int, help="Max completion tokens")
    parser.add_argument("--top_p", type=float,default=1, help="Top P for the model")
    parser.add_argument("--stop", type=str, nargs="+", help="Stop sequence for the model")
    parser.add_argument("--logprobs", action="store_true", help="Logprobs for the model")
    parser.add_argument("--dataset_path",default="./datasets/dataset_humaneval.csv.gz" ,type=str, help="Path to the dataset")
    parser.add_argument("--split", type=str,default="mini", help="Split of the dataset")
    parser.add_argument("--method", type=str,default="foa", help="Method to use")
    parser.add_argument("--conf_path", type=str, default="./scripts/frameworks/humaneval/humaneval.yaml",help="Path to corresponding config")
    parser.add_argument("--value_cache", action="store_true", help="Use value cache")
    parser.add_argument("--correctness", type=int, help="Use original ('correct') implementation")
    args = parser.parse_args()

    filename = f"logs/correctness/{args.model.split('/')[-1]}/humaneval/{args.method}.log"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=filename, filemode="a")
    logger.info("#"*50)

    # Load previous content
    with open(filename, "r") as f:
        contents = f.read()
    
    if args.batch_size == 1:
        previous_trials = [int(num) for num in re.findall(r"Shared Namespace \(trial (\d+)\)", contents)]
        trial = max(previous_trials) + 1 if previous_trials else 1
        logger.info(f"Shared Namespace (trial {trial})")
        cache_path = f"caches/frameworks/{args.method}/sns_{trial}"
    else:
        previous_trials = [int(num) for num in re.findall(r"Shared Namespace and Batch \(trial (\d+)\)", contents)]
        trial = max(previous_trials) + 1 if previous_trials else 1
        logger.info(f"Shared Namespace and Batch (trial {trial})")
        cache_path = f"caches/frameworks/humaneval"

    logger.info(f"Dataset Split: {args.split}")

    asyncio.run(run(args, trial=trial, cache_path=cache_path))
    logger.info("\n"*3)
    clean_log(filename)
