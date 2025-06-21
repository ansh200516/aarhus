import random
import logging
import asyncio
import numpy as np
from typing import TypedDict, Optional
from dataclasses import replace
from ..typedefs import Algorithm, Model, AgentDict, Agent, StateReturningAgent, ValueFunctionRequiringAgent, ValueFunctionUsingAgent, Environment, DecodingParameters, State, Benchmark, MAX_SEED
from ..utils import *

logger = logging.getLogger('new_algo_logger')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/new_algo_logs.log')
handler.setLevel(logging.INFO)
logger.addHandler(handler)

DEBUG = False  # Set to True to enable debug prints


class AgentDictHeterogenousFOA(TypedDict):
    evaluate: Agent
    eval_params: DecodingParameters
    step_agents: list[AgentDict]



class AlgorithmNewAlgo(Algorithm):
    def __init__(self,
                 model: Model,
                 agents: AgentDictHeterogenousFOA,
                 env: Environment,
                 width: int,
                 num_steps: int,
                 max_value: float,
                 k: int,
                 alpha: float,
                 backtrack: float,
                 resampling: str,
                 origin: float,
                 min_steps: int,
                 num_evaluations: int
                ):
        super().__init__(model, agents, env)

        self.eval_agent = agents["evaluate"]
        self.eval_params = agents["eval_params"]

        self.step_agents = agents["step_agents"]

        self.width = width
        self.num_steps = num_steps
        self.max_value = max_value
        self.k = k
        self.alpha = alpha
        self.backtrack = backtrack
        self.resampling = resampling
        self.origin = origin
        self.min_steps = min_steps
        self.num_evaluations = num_evaluations
        self.input_state_values = []
        # self.priors = np.ones((self.num_steps, len(self.step_agents))) / len(self.step_agents)
        # self.priors = np.array([[0.25032784, 0.54608981, 0.20358235],
        #             [0.15549521, 0.65412951, 0.19037528],
        #             [0.14660598, 0.36594092, 0.48745309],
        #             [0.1080735,  0.18180222, 0.71012428],
        #             [0.10565252, 0.21038833, 0.68395915],
        #             [0.27423906, 0.35757344, 0.3681875 ]])
        self.priors = np.array([[0.14744174, 0.61509742, 0.23746084],
            [0.3783879,  0.22943522, 0.39217688],
            [0.24950015, 0.39282898, 0.35767088],
            [0.24926366, 0.15897673, 0.5917596 ],
            [0.21156607, 0.16818497, 0.62024896],
            [0.14960874, 0.17279655, 0.67759471]])

        logger.info('#################################################################')


    def _get_ith_agent_dict(self, i) -> AgentDict:
        """Get the i-th agent in the fleet and its parameters."""
        if i >= self.width or i < 0:
            raise IndexError(f"Agent index {i} out of range.")

        agent_dict_index = -1
        while i >= 0:
            agent_dict_index += 1
            i -= self.step_agents[agent_dict_index]["num_agents"]

        return self.step_agents[agent_dict_index]
    

    def _wrap_agent_in_env(self, agent_class):
        if issubclass(agent_class, StateReturningAgent):
            return agent_class
                
        class EnvWrappedAgent(agent_class, StateReturningAgent):
            @staticmethod
            async def act(model: Model, state: State, n: int, namespace: str, request_id: str, params: DecodingParameters):
                actions = await agent_class.act(model=model, state=state, n=n, namespace=namespace, request_id=request_id, params=params)
                new_states = [self.env.step(state, action) for action in actions]

                return new_states
        
        EnvWrappedAgent.__name__ = f"EnvWrapped{agent_class.__name__}"
        EnvWrappedAgent.__qualname__ = f"EnvWrapped{agent_class.__qualname__}"
        EnvWrappedAgent.__module__ = agent_class.__module__
        EnvWrappedAgent.__doc__ = f"EnvWrapped version of {agent_class.__name__} to work with the environment."
        EnvWrappedAgent.__annotations__ = agent_class.__annotations__.copy()

        return EnvWrappedAgent


    def _wrap_agent_in_value_function(self, agent_class):
        if not issubclass(agent_class, ValueFunctionRequiringAgent) or issubclass(agent_class, ValueFunctionUsingAgent):
            return agent_class
                
        class ValueFunctionWrappedAgent(agent_class, ValueFunctionUsingAgent):
            @staticmethod
            async def act(model: Model, state: State, n: int, namespace: str, request_id: str, params: DecodingParameters):
                value_agent = AgentDict(
                    agent=self.eval_agent,
                    params=self.eval_params,
                    model=self.model,
                    num_agents=self.num_evaluations
                )

                return await agent_class.act(
                    model=model, 
                    state=state, 
                    n=n, 
                    namespace=namespace, 
                    request_id=request_id, 
                    params=params, 
                    value_agent=value_agent
                )
        
        ValueFunctionWrappedAgent.__name__ = f"ValueFunctionWrapped{agent_class.__name__}"
        ValueFunctionWrappedAgent.__qualname__ = f"ValueFunctionWrapped{agent_class.__qualname__}"
        ValueFunctionWrappedAgent.__module__ = agent_class.__module__
        ValueFunctionWrappedAgent.__doc__ = f"ValueFunctionWrapped version of {agent_class.__name__} to work with the value function."
        ValueFunctionWrappedAgent.__annotations__ = agent_class.__annotations__.copy()

        return ValueFunctionWrappedAgent


    def _wrap_agent(self, agent_class):
        agent_class = self._wrap_agent_in_env(agent_class)
        agent_class = self._wrap_agent_in_value_function(agent_class)
        return agent_class


    def sample_from_priors(self, state: State):
        return np.random.choice(
            len(self.step_agents),
            p=self.priors[state.t]
        )

    
    def update_priors(self, agents_used, states, new_states):
        for i, state in enumerate(states):
            t = state.t
            delta = new_states[i].value - state.value

            for k in range(-self.k, self.k+1):
                if t+k < 0 or t+k >= self.num_steps:
                    continue
                self.priors[t+k][agents_used[i]] += (self.backtrack**abs(k)) * self.priors[t][agents_used[i]] * self.alpha * delta
                self.priors[t+k][agents_used[i]] = min(0.85, max(0.15, self.priors[t+k][agents_used[i]]))
        
        self.priors /= self.priors.sum(axis=-1, keepdims=True)


    async def mutate_states(self, states: List[State], namespace: str, idx: int, step: int):
        # choose each agent individually using the priors
        agent_idxs_used = []
        agent_coroutines = []

        for i, state in enumerate(states):
            t = state.t
            agent_idx = self.sample_from_priors(state)

            agent_idxs_used.append(agent_idx)
            agent_coroutines.append(
                self._get_ith_agent_dict(agent_idx)['agent'].act(
                    model=self._get_ith_agent_dict(agent_idx).get('model') or self.model,
                    state=state,
                    n=1,
                    namespace=namespace,
                    request_id=f"idx{idx}-step{step}-{hash(state)}-agent{agent_idx+100*i}",
                    params=self._get_ith_agent_dict(agent_idx)['params']
                )
            )
        
        agent_responses = await asyncio.gather(*agent_coroutines)
        states = []
        for resp in agent_responses:
            states.extend(resp)
        
        return states, agent_idxs_used
    

    async def evaluate_states(self, states: List[State], value_cache, namespace: str, idx: int, step: int):
        solved_idxs = []
        terminal_idxs = []
        value_coroutines = []

        for i, state in enumerate(states):
            terminal = False
            if self.env.is_final(state):
                terminal_idxs.append(i)
                terminal = True

            if self.env.evaluate(state)[1] == 1:
                solved_idxs.append(i)
                terminal = True

            if not terminal:
                value_coroutines.append(
                    self.eval_agent.act(
                        model=self.model,
                        state=state,
                        n=self.num_evaluations,
                        namespace=namespace,
                        request_id=f"idx{idx}-evaluation{step}-{hash(state)}-agent{i}",
                        params=self.eval_params,
                        cache=value_cache
                    )
                )
        
        calculated_values = await asyncio.gather(*value_coroutines)

        values = []
        for i in range(len(states)):
            if i in solved_idxs:
                values.append(self.max_value)
            elif i in terminal_idxs:
                values.append(0)
            else:
                values.append(calculated_values.pop(0))
        
        new_states = []
        for value, state in zip(values, states):
            new_states.append(replace(state, value=value))

        return new_states, terminal_idxs, solved_idxs


    async def evaluate_uncertainty_in_states(self, states: List[State], namespace: str, idx: int, step: int):
        solved_idxs = []
        value_coroutines = []

        for i, state in enumerate(states):
            if self.env.evaluate(state)[1] == 1:
                solved_idxs.append(i)
            else:
                value_coroutines.append(
                    self.eval_agent.act(
                        model=self.model,
                        state=state,
                        n=self.num_evaluations,
                        namespace=namespace,
                        request_id=f"idx{idx}-uncertainty{step}-{hash(state)}-agent{i}",
                        params=self.eval_params,
                        cache=None
                    )
                )
        
        calculated_values = await asyncio.gather(*value_coroutines)

        values = []
        for i in range(len(states)):
            if i in solved_idxs:
                values.append(0)
            else:
                values.append(calculated_values.pop(0))

        new_states = []
        for i in range(len(values)):
            new_states.append(replace(states[i], uncertainty=values[i]))
        
        return new_states


    async def solve(self, idx: int, state: State, namespace: str, value_cache: dict = None):
        randomness = idx
        random.seed(randomness)
        resampler = Resampler(randomness)

        # set the value of inital state
        # state = replace(state, value=self.origin*self.backtrack*self.backtrack, uncertainty=10)
        state = replace(state, value=self.origin*self.backtrack*self.backtrack)

        # log the puzzle to stdout
        print('initial problem state:')
        print(state.puzzle)

        # Records of previously visited states (state_identifier, state_value, state)
        visited_states = [("INIT", self.origin, state)]

        # Initialize state for each agent
        states = [state.clone(randomness=random.randint(0, MAX_SEED)) for _ in range(self.width)]

        # Wrap action agents in the environment
        for i in range(len(self.step_agents)):
            agent_class = self.step_agents[i]['agent']
            self.step_agents[i]['agent'] = self._wrap_agent(agent_class)


        solved = False
        terminal_states = []
        for step in range(self.num_steps):
            print(f"Step {step} ({idx})")

            # mutate using agents
            new_states, agents_used = await self.mutate_states(states, namespace, idx, step)

            # evaluate all the new states
            new_states, terminal_idxs, solved_idxs = await self.evaluate_states(new_states, value_cache, namespace, idx, step)
            # new_states = await self.evaluate_uncertainty_in_states(new_states, namespace, idx, step)

            # learn the priors
            self.update_priors(agents_used, states, new_states)

            # log the failed states
            if len(terminal_idxs) != len(solved_idxs):
                print(f'Terminated incorrectly {len(terminal_idxs) - len(solved_idxs)} times for ({idx}) at step {step}')

            # early exit
            if len(solved_idxs) > 0:
                solved = True
                print(f'Solved {idx} at step {step}')
                break

            # append to terminal states list
            terminal_states.extend([new_states[i] for i in terminal_idxs])

            # Filter previously visited states records
            remaining_steps = self.num_steps - (step + 1)
            visited_states = [(identifier, value*self.backtrack, state) for identifier, value, state in visited_states]
            visited_states = [state for state in visited_states if  remaining_steps >= self.min_steps - len(state[2].steps)]

            # replace the terminal states with old ones
            if visited_states != []:
                replacements, _ = resampler.resample(visited_states.copy(), len(terminal_idxs), self.resampling)
            else:
                replacements, _ = resampler.resample([("", state.value, state) for state in states], len(terminal_idxs), resampling_method="linear")
            states = [replacements.pop(0) if i in terminal_idxs else state for i, state in enumerate(new_states)]

            # resampling
            for i in range(len(new_states)):
                if i not in terminal_idxs:
                    visited_states.append((f"{i}.{step}", states[i].value, states[i]))
            states, _ = resampler.resample(visited_states, self.width, self.resampling)


        return new_states


    async def benchmark(self, benchmark: Benchmark, share_ns: bool=False, cache: bool=True):
        cache = {} if cache else None
        solve_coroutines = [
            self.solve(
                idx=index,
                state=state,
                namespace="benchmark" if share_ns else f"benchmark_{index}",
                value_cache=cache
            )
            for index, state in benchmark
        ]
        results = await asyncio.gather(*solve_coroutines)


        # TODO: REMOVE

        print(self.priors)

        return results



class AlgorithmNewAlgoV2(AlgorithmNewAlgo):
    def __init__(self,
                 model: Model,
                 agents: AgentDictHeterogenousFOA,
                 env: Environment,
                 width: int,
                 num_steps: int,
                 max_value: float,
                 k: int,
                 alpha: float,
                 backtrack: float,
                 resampling: str,
                 origin: float,
                 min_steps: int,
                 num_evaluations: int
                ):
        
        super().__init__(model, agents, env, width, num_steps, max_value, k, alpha, backtrack, resampling, origin, min_steps, num_evaluations)
        self.priors = [[1, 1, 1, 1] for _ in range(len(self.step_agents))] # (alpha_e, beta_e, alpha_x, beta_x)
        self.lambda_t = lambda t: 0.8*(1-(t/self.num_steps))
    

    def sample_from_priors(self, state):
        value_s = state.value
        uncertainty_s = self.max_value - state.value

        agent = 0
        max_utility = 0

        for i in range(len(self.step_agents)):
            exploration_prior = np.random.beta(
                self.priors[i][0],
                self.priors[i][1],
            )

            expliotation_prior = np.random.beta(
                self.priors[i][2],
                self.priors[i][3],
            )

            utility = self.lambda_t(state.t) * uncertainty_s * exploration_prior + \
                        (1 - self.lambda_t(state.t)) * value_s * expliotation_prior
            
            if utility > max_utility:
                max_utility = utility
                agent = i
        
        return agent



    def update_priors(self, agents_used, states, new_states):
        for i in range(len(states)):
            value_s_t = states[i].value
            uncertainty_s_t = states[i].uncertainty

            value_s_t_plusone = new_states[i].value
            uncertainty_s_t_plusone = new_states[i].uncertainty

            if uncertainty_s_t_plusone < uncertainty_s_t:
                self.priors[agents_used[i]][0] += self.alpha * (uncertainty_s_t - uncertainty_s_t_plusone)
            else:
                self.priors[agents_used[i]][1] += self.alpha * (uncertainty_s_t_plusone - uncertainty_s_t)

            if value_s_t_plusone > value_s_t:
                self.priors[agents_used[i]][2] += self.alpha * (value_s_t_plusone - value_s_t)
            else:
                self.priors[agents_used[i]][3] += self.alpha * (value_s_t - value_s_t_plusone)
