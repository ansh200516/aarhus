import re
import random
from typing import List
import numpy as np
import asyncio
import itertools
from dataclasses import replace

from . import prompts as prompts
from .state import StateGame24
from ...typedefs import Request, Agent, AgentDict, ValueFunctionRequiringAgent, StateReturningAgent, Model, DecodingParameters

from .environment import EnvironmentGame24

act_cache = {}
env = EnvironmentGame24

class AgentActGame24(Agent):
    """ """

    async def act(
        model: Model,
        state: StateGame24,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        # Format the prompt
        if state.current_state == "24":
            prompt = (
                prompts.cot.format(input=state.puzzle, context_str='')
                + "\nSteps:\n"
                + "\n".join(state.steps)
                + "\nAnswer: "
            )
        else:
            context_str = get_context(state)
            current_numbers = get_current_numbers(state)
            prompt = prompts.bfs.format(input=current_numbers, context_str=context_str)

        if prompt in act_cache:
            proposals = act_cache[prompt][:n]
            act_cache[prompt] = act_cache[prompt][n:]
        else:
            proposals = []
            act_cache[prompt] = []

        while len(proposals) < n:
            # Generate the response
            response = await model.request(
                prompt=prompt,
                n=1,
                request_id=request_id,
                namespace=namespace,
                params=params,
            )
            # Parse the response
            if state.current_state != "24":
                response = [response[0].rpartition(")")[0] + ")"]
            proposals.extend(r.strip() for r in response[0].split("\n"))
            if "Possible next steps:" in proposals:
                proposals.remove("Possible next steps:")

        random.seed(state.randomness)
        random.shuffle(proposals)
        act_cache[prompt].extend(proposals[n:])
        actions = proposals[:n]
        return actions


class AgentAggregateGame24(Agent):

    @staticmethod
    async def act(
        model: Model,
        state: StateGame24,
        actions: List[str],
        k: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns the aggregated actions for the Game of 24 task.
        """
        if len(actions) == 0:
            return []
        
        if len(state.current_state.split(" ")) == 1:
            return actions

        # Format the prompt
        proposals = ""
        for idx, action in enumerate(actions):
            proposals += f"({idx + 1}) " + action + "\n"

        context_str = get_context(state)
        prompt = prompts.aggregate.format(
            state=state.current_state, proposal=proposals.strip(), n_select_sample=k,
            context_str=context_str
        )

        responses = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the response
        try:
            selected_indexes = [int(i.strip()) - 1 for i in re.findall(r"\d+", responses[0])]
            selected_actions = [actions[i] for i in selected_indexes if i < len(actions)]
        except:
            selected_actions = []
        return selected_actions


class AgentBfsGame24(Agent):

    @staticmethod
    async def act(
        model: Model,
        state: StateGame24,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns a list of actions for the Game of 24 task.
        """
        # Format the prompt
        if len(state.current_state.strip().split(" ")) == 1:
            prompt = (
                prompts.cot.format(input=state.puzzle, context_str='')
                + "\nSteps:\n"
                + "\n".join(state.steps).strip()
                + "\nAnswer: "
            )

        else:
            context_str = get_context(state)
            current_numbers = get_current_numbers(state)
            prompt = prompts.bfs.format(input=current_numbers, context_str=context_str)

        # Generate the response
        response = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the response
        if state.current_state != "24":
            response = [response[0].rpartition(")")[0] + ")"]
        proposals = [r.strip() for r in response[0].split("\n")]
        if "Possible next steps:" in proposals:
            proposals.remove("Possible next steps:")
        return proposals


class AgentEvaluateGame24(Agent):

    @staticmethod
    async def act(
        model: Model,
        state: StateGame24,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
        cache: dict = None,
    ) -> float:
        """
        Returns a value for the given state
        """

        # Check if the state is already in the cache
        if cache is not None and state.current_state in cache:
            return cache[state.current_state]

        # Format the prompt
        if state.steps and "left" not in state.steps[-1]:
            formula = get_formula(state)
            prompt = prompts.evaluate_answer.format(input=state.puzzle, answer=formula)
        else:
            prompt = prompts.evaluate.format(input=state.current_state)

        # Format the request
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the response
        codes = [r.split("\n")[-1].lower().strip() for r in responses]
        code_map = {r"impossible": 0.001, r"likely": 1, r"sure": 20}
        value = 0
        for pattern, weight in code_map.items():
            matches = [code for code in codes if re.search(pattern, code)]
            value += weight * len(matches)

        # Cache the value
        if cache is not None:
            cache[state.current_state] = value
        return value


class AgentEvaluateObjectiveDifficultyGame24(Agent):

    @staticmethod
    async def act(
        model: Model,
        state: StateGame24,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> float:
        ways = count_make_24(list(map(int, state.puzzle.split()))) / 1000
        
        # 4 to 15, 6.75-->5, 9.5-->3, 12.25--> 1
        dist_1 = abs(13 - ways)
        dist_2 = abs(9.5 - ways)
        dist_3 = abs(6 - ways)

        if dist_1 == min(dist_1, dist_2, dist_3):
            diff = 1
        elif dist_2 == min(dist_1, dist_2, dist_3):
            diff = 3
        else:
            diff = 5

        return diff


class AgentEvaluateDifficultyGame24(Agent):

    @staticmethod
    async def act(
        model: Model,
        state: StateGame24,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> float:
        """
        Returns a difficulty value for the given state
        """
        prompt = prompts.evaluate_difficulty.format(input=state.puzzle)

        # Format the request
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        values = []
        pattern = r"Difficulty(?:\s+Score)?\s*:\s*([0-9]*\.?[0-9]+)"

        for response in responses:
            match = re.search(pattern, response, re.IGNORECASE)
            if match and match.group(1):
                value = float(match.group(1))
            else:
                value = 3
            values.append(value)
        value = sum(values) / len(values) if values else 3

        d1 = abs(1-value)
        d3 = abs(3-value)
        d5 = abs(5-value)

        rating = 0
        if d1 == min(d1, d3, d5):
            rating = 1
        elif d3 == min(d1, d3, d5):
            rating = 3
        else:
            rating = 5

        return rating


class AgentReactGame24(Agent):
    """
    Agent for React algorithm
    """

    @staticmethod
    async def act(
        model: Model,
        state: StateGame24,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        # Format the prompt
        if state.current_state == "24":
            prompt = (
                prompts.cot.format(input=state.puzzle, context_str='')
                + "\nSteps:\n"
                + "\n".join(state.steps)
                + "\nAnswer: "
            )
        else:
            context_str = get_context(state)
            current_numbers = get_current_numbers(state)
            prompt = prompts.react.format(input=current_numbers, context_str=context_str)

        # Generate the response
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the response
        proposals = [r.split("Possible next step:")[-1].strip() for r in responses]
        return proposals


class AgentRapGame24(Agent):
    """
    Agent for React algorithm
    """

    @staticmethod
    async def act(
        model: Model,
        state: StateGame24,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        if state.current_state == "24":
            prompt = (
                prompts.cot.format(input=state.puzzle, context_str='')
                + "\nSteps:\n"
                + "\n".join(state.steps)
                + "\nAnswer: "
            )
        else:
            current_numbers = get_current_numbers(state)
            context_str = get_context(state)
            prompt = prompts.react.format(input=current_numbers, context_str=context_str)

        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        proposals = [r.strip() for r in responses]
        return proposals


class AgentSelfEvaluateGame24(Agent):
    """
    Agent that performs self-evaluation of reasoning steps for Game24.
    Uses the LLM's own estimation of correctness by evaluating each reasoning step.
    Uses the probability of "Yes" as a reward signal for correct reasoning.
    """

    @staticmethod
    async def act(
        model: Model,
        state: StateGame24,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
        cache: dict = None,
    ) -> float:

        if cache is not None and state.current_state in cache:
            return cache[state.current_state]

        # Format the prompt based on whether we're evaluating a final answer or intermediate step
        if state.steps and "left" not in state.steps[-1]:
            # Evaluating a final answer
            formula = get_formula(state)
            prompt = prompts.self_evaluate_answer.format(
                input=state.puzzle, answer=formula, steps="\n".join(state.steps)
            )
        else:
            # Evaluating intermediate reasoning steps
            current_numbers = get_current_numbers(state)
            last_step = state.steps[-1] if state.steps else ""
            prompt = prompts.self_evaluate_step.format(
                input=current_numbers,
                step=last_step,
                previous_steps=(
                    "\n".join(state.steps[:-1]) if len(state.steps) > 1 else ""
                ),
            )

        eval_params = DecodingParameters(
            temperature=params.temperature,
            max_completion_tokens=params.max_completion_tokens,
            top_p=params.top_p,
            stop=params.stop,
            logprobs=True,
        )

        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=eval_params,
        )

        # Calculate the average probability of "Yes" across all responses
        yes_probabilities = []
        for response in responses:
            # Get the logprobs for the first token after the prompt
            if hasattr(response, "logprobs") and response.logprobs:
                first_token_logprobs = response.logprobs[0]
                # Look for Yes token probability
                yes_prob = next(
                    (
                        prob
                        for token, prob in first_token_logprobs.items()
                        if token.lower() in ["yes", "yes.", "yes!"]
                    ),
                    0.0,
                )
                yes_probabilities.append(
                    np.exp(yes_prob)
                )  # Convert logprob to probability

        if yes_probabilities:
            value = sum(yes_probabilities) / len(yes_probabilities)
            value = value * 20
        else:
            value = 0.001

        if cache is not None:
            cache[state.current_state] = value

        return value


class AgentReflectGame24(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateGame24,
        n: int, # Number of responses/actions to generate
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ):
        num_examples = min(2, len(prompts.examples_reflect))
        examples_str = "(Example Reflection)\n" + "\n\n(Example Reflection)\n".join(
        [example for example in prompts.examples_reflect[:num_examples]]
        )
        
        prompt = prompts.reflect.format(
            examples=examples_str,
            problem=state.puzzle,
            steps='\n'.join(state.steps)
        )
        
        responses = await model.request(
            prompt=prompt,
            n=1, 
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        reflection_text = responses[0].strip()
        return reflection_text
        

class AgentTerminalReflectHotpotQA(StateReturningAgent):
    @staticmethod
    async def act(
        model: Model,
        state: StateGame24,
        n: int, # Number of responses/actions to generate
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        actions = await AgentActGame24.act(
            model=model,
            state=state,
            n=n,
            namespace=namespace,
            request_id=request_id,
            params=params,
        )

        # additional terminal reflection logic
        states = [EnvironmentGame24.step(state, action) for action in actions]

        reflection_coroutines = []
        reflected_state_idxs = []
        for i, s in enumerate(states):
            if not EnvironmentGame24.is_final(s):
                continue

            # found a successful state
            if EnvironmentGame24.evaluate(s)[1] == 1:
                return [s]
            
            # if the state has failed, we need to reflect on it
            reflection_coroutines.append(
                AgentReflectGame24.act(
                    model=model,
                    state=s,
                    n=1,
                    namespace=namespace,
                    request_id=f"{request_id}-reflect-{i}",
                    params=params,
                )
            )

            reflected_state_idxs.append(i)
        
        if len(reflection_coroutines) == 0:
            return states
        
        thoughts = await asyncio.gather(*reflection_coroutines)
        
        for i in reflected_state_idxs:
            states[i] = StateGame24(
                puzzle=state.puzzle,
                current_state=state.puzzle,
                steps=[],
                t=0,
                answer=state.answer,
                docstore=state.docstore,
                randomness=state.randomness,
                reflections=[thoughts.pop(0)] + state.reflections,
                parent=state,
            )

        return states


class AgentValueReduceReflectHotpotQA(StateReturningAgent, ValueFunctionRequiringAgent):
    @staticmethod
    async def act(
        model: Model,
        state: StateGame24,
        n: int, # Number of responses/actions to generate
        namespace: str,
        request_id: str,
        params: DecodingParameters,
        value_agent: AgentDict
    ) -> List[str]:
        actions = await AgentActGame24.act(
            model=model,
            state=state,
            n=n,
            namespace=namespace,
            request_id=request_id,
            params=params,
        )

        # additional reflection logic: when value of new states is lower than current state
        states = [EnvironmentGame24.step(state, action) for action in actions]
        failed_states = []
        non_terminal_states = []
        value_reduced_states = []
        new_states = []

        for s in states:
            # return the winning state if it exists
            if EnvironmentGame24.evaluate(s)[0] == 1:
                return [s]
            
            # add failing states to the list
            if EnvironmentGame24.is_final(s):
                failed_states.append(s)
            else:
                non_terminal_states.append(s)        
        
        # get values for non terminal states
        if len(non_terminal_states) > 0:
            value_coroutines = [
                value_agent["agent"].act(
                    model=value_agent.get('model') or model,
                    state=s,
                    n=value_agent['num_agents'],
                    namespace=namespace,
                    request_id=f"{request_id}-value-{i}",
                    params=value_agent['params'],
                ) for i, s in enumerate(non_terminal_states)
            ]
            values = await asyncio.gather(*value_coroutines)
            for i in range(len(non_terminal_states)):
                if values[i] >= state.value:
                    # if the value of the new state is higher than the current state, keep it
                    new_states.append(replace(non_terminal_states[i], value=values[i]))
                else:
                    value_reduced_states.append(replace(non_terminal_states[i], value=values[i]))

        # get values for failed states
        if len(failed_states) > 0:
            for i in range(len(failed_states)):
                value_reduced_states.append(replace(failed_states[i], value=0))
        
        if len(value_reduced_states) == 0:
            # if no states were reduced, return the new states
            return new_states
        

        # reflect on the value reduced states
        reflection_coroutines = []
        for i, s in enumerate(value_reduced_states):
            reflection_coroutines.append(
                AgentReflectGame24.act(
                    model=model,
                    state=s,
                    n=1,
                    namespace=namespace,
                    request_id=f"{request_id}-reflect-{i}",
                    params=params,
                )
            )

        thoughts = await asyncio.gather(*reflection_coroutines)
        num_thoughts = len(thoughts)

        for i in range(num_thoughts):
            old_state_with_thought = state.clone()
            old_state_with_thought.reflections.insert(0, thoughts.pop(0))
            new_states.append(replace(old_state_with_thought, value=None))

        return new_states


def get_current_numbers(state: StateGame24) -> str:
    """
    Returns the current numbers in the state.
    """
    last_line = state.current_state.strip().split("\n")[-1]
    return last_line.split("left: ")[-1].split(")")[0]


def get_context(state: StateGame24) -> str:
    context_str = ''
    if state.reflections:
        reflections = '\n'.join(state.reflections)
        context_str = f"\nUse the given context as a guideline to solve the problem. Do not mention any usage of the context. Strictly follow the output format guidelines.\nContext: {reflections}\n\n(END OF CONTEXT)\n"
    return context_str


def get_formula(state: StateGame24) -> str:
    if state.steps:
        formula = state.steps[-1].lower().replace("answer: ", "")
        return formula
    else:
        # Should do some error handling here but for the moment we'll take it as it is
        return ""


def count_make_24(nums):
    if len(nums) < 2 or len(nums) > 4:
        return 0

    seen_expressions = set()

    def dfs(numbers):
        if len(numbers) == 1:
            if abs(numbers[0] - 24) < 1e-6:
                return 1
            return 0

        count = 0
        for i in range(len(numbers)):
            for j in range(len(numbers)):
                if i == j:
                    continue

                a, b = numbers[i], numbers[j]
                rest = [numbers[k] for k in range(len(numbers)) if k != i and k != j]

                operations = []
                operations.append(a + b)
                operations.append(a - b)
                operations.append(b - a)
                operations.append(a * b)
                if b != 0:
                    operations.append(a / b)
                if a != 0:
                    operations.append(b / a)

                for result in operations:
                    count += dfs(rest + [result])

        return count

    total_count = 0
    unique_permutations = set(itertools.permutations(nums))
    for perm in unique_permutations:
        total_count += dfs(list(perm))

    return total_count
