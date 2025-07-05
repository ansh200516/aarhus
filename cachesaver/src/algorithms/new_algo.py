import random
import logging
import json
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


class AgentDictNewAlgo(TypedDict):
    evaluate: Agent
    eval_params: DecodingParameters
    step_agents: list[AgentDict]
    difficulty_agent: Optional[AgentDict]



class AlgorithmNewAlgo(Algorithm):
    def __init__(self,
                 model: Model,
                 agents: AgentDictNewAlgo,
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

        self.difficulty_agent = agents.get("difficulty_agent")

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
        self.priors = np.ones((self.num_steps, len(self.step_agents))) / len(self.step_agents)

        # turn features on and off by setting bool values
        self.features = {
            'updating_priors': True,
            'difficulty_based_width_init': False,
            'runtime_width_adaptation': True,
            'skewed_state_detection': True
        }

        logger.info('#################################################################')


    def _get_ith_agent_dict(self, i) -> AgentDict:
        """Get the i-th agent in the fleet and its parameters."""
        if i >= sum([a['num_agents'] for a in self.step_agents]) or i < 0:
            raise IndexError(f"Agent index {i} out of range.")

        agent_dict_index = -1
        while i >= 0:
            agent_dict_index += 1
            i -= self.step_agents[agent_dict_index]["num_agents"]

        return self.step_agents[agent_dict_index]
    

    '''
    Agent Wrappers
    '''

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


    '''
    LOG METHODS
    '''

    def _log_init(self, idx, width):
        def get_model_name(model):
            name = 'Unknown Model'
            try:
                name = model.model
            except:
                pass

            return name

        log = json.dumps({
            f'{self.env.__name__}-{idx}-init': {
                'env': self.env.__name__,
                'idx': idx,
                'features': self.features,
                'agents': [
                    {
                        'agent': agent_dict['agent'].__name__.replace('EnvWrapped', '').replace('ValueFunctionWrapped', ''),
                        'params': agent_dict['params'],
                        'model': get_model_name(agent_dict.get('model', self.model)),
                        'num_agents': agent_dict['num_agents']
                    }
                    for agent_dict in self.step_agents
                ],
                'width': width
            }
        })

        logger.info(log)

    
    def _log_step(self, idx, step, states, new_states, agents_used, terminal_idxs, solved_idxs):
        def get_model_name(model):
            name = 'Unknown Model'
            try:
                name = model.model
            except:
                pass
            return name
        
        log = json.dumps({
            f'{self.env.__name__}-{idx}-step': {
                'env': self.env.__name__,
                'idx': idx,
                'step': step,
                'input_states': [s.serialize() for s in states],
                'priors': self.priors.tolist(),
                'agents_used': [
                    {
                        'agent': agent_dict['agent'].__name__.replace('EnvWrapped', '').replace('ValueFunctionWrapped', ''),
                        'params': agent_dict['params'],
                        'model': get_model_name(agent_dict.get('model', self.model)),
                        'num_agents': agent_dict['num_agents']
                    }
                    for agent_dict in map(self._get_ith_agent_dict, agents_used)
                ],
                'output_states': [s.serialize() for s in new_states],
                'terminal_idxs': terminal_idxs,
                'solved_idxs': solved_idxs
            }
        })

        logger.info(log)


    def _log_end(self):
        log = json.dumps({
            f'{self.env.__name__}-end': {
                'priors': self.priors.tolist()
            }
        })

        logger.info(log)


    '''
    PRIOR (the prob. dist, not the english word 'prior') RELATED METHODS
    '''

    def sample_from_priors(self, state: State):
        self.priors = np.nan_to_num(self.priors, nan=0.0)
        self.priors /= self.priors.sum(axis=-1, keepdims=True)

        return np.random.choice(
            len(self.step_agents),
            p=self.priors[state.t]
        )

    
    def update_priors(self, agents_used, states, new_states):
        if not self.features['updating_priors']:
            return
        
        for i, state in enumerate(states):
            t = state.t
            delta = new_states[i].value - state.value

            for k in range(-self.k, self.k+1):
                if t+k < 0 or t+k >= self.num_steps:
                    continue
                self.priors[t+k][agents_used[i]] += (self.backtrack**abs(k)) * self.priors[t][agents_used[i]] * self.alpha * delta
                self.priors[t+k][agents_used[i]] = min(1, max(0.001, self.priors[t+k][agents_used[i]]))
        
        self.priors = np.nan_to_num(self.priors, nan=0.0)
        self.priors /= self.priors.sum(axis=-1, keepdims=True)


    '''
    MAIN GENETIC ALGO. ABSTRACTIONS
    '''

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


    def resample_states(self, resampler, states, new_states, visited_states, terminal_idxs, step, width):
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
        states, _ = resampler.resample(visited_states, width, self.resampling)

        return states, visited_states


    '''
    POPULATION SIZE METHODS
    '''

    def update_width(self, states, new_states, visited_states, resampler, terminal_idxs, step):
        width = len(states)
        if not self.features['runtime_width_adaptation']:
            return width
        
        resampled_out_states, _ = self.resample_states(resampler, states, new_states, visited_states.copy(), terminal_idxs, step, width)
        # resampled_out_states = new_states
        
        in_values = [state.value for state in states]
        out_values = [state.value for state in resampled_out_states]

        if len(out_values) == 0:
            out_values = [0] * width

        in_avg = sum(in_values) / len(in_values)
        out_avg = sum(out_values) / len(out_values)

        if in_avg < out_avg:
            new_w = max(self.width - self.width//2, width - 1)
        else:
            new_w = min(self.width + self.width//2, width + 1)
        
        if new_w < self.width // 2:
            new_w = self.width // 2
        elif new_w > self.width + self.width//2:
            new_w = self.width + self.width//2

        return new_w


    async def get_width(self, state, idx):
        if self.difficulty_agent is None or not self.features['difficulty_based_width_init']:
            return self.width
        
        rating = await self.difficulty_agent['agent'].act(
            self.difficulty_agent.get('model') or self.model,
            state,
            n=1,
            namespace=f"diff-randomnamespace-{idx}",
            request_id=f"diff-randomnrequestid-{idx}",
            params=self.difficulty_agent['params'],
        )

        # diff = 0 --> w = 2
        # diff = 10 --> w = self.w

        w = {
            1: self.width - self.width//2,
            3: self.width,
            5: self.width + self.width//2
        }

        return w[rating]



    '''
    SKEWED STATE DETECTION
    '''

    def filter_states(self, input_state, states, new_states, visited_states, terminal_idxs):
        if not self.features['skewed_state_detection']:
            return new_states, visited_states

        freq_in_states = {}
        skewed_states = []

        for i in range(len(states)):
            state = states[i]
            cs = state.current_state

            if cs == input_state.current_state:
                continue

            if cs not in freq_in_states:
                freq_in_states[cs] = []
            
            freq_in_states[cs].append(i)

        for idxs in freq_in_states.values():
            if len(idxs) <= 1: # TODO: Change this threshold
                continue

            # check if skewed state and filter if it is
            parent_values = [states[i].value for i in idxs]
            avg_parent_value = sum(parent_values) / len(parent_values)
            children = [new_states[i] for i in idxs]
            # if all([child.value <= (avg_parent_value-10) for child in children]):
            if all([child.value <= (avg_parent_value/2) for child in children]):
                # mark as skewed state
                skewed_states.append(states[idxs[0]].current_state)
                skewed_states.extend([child.current_state for child in children])
        

        # remove the skewed states from visited states and new states
        updated_vis_states = []
        updated_new_states = []
        for idf, s_val, s in visited_states:
            if s.current_state in skewed_states:
                continue
            updated_vis_states.append((idf, s_val, s))
        
        for s in new_states:
            if s.current_state in skewed_states:
                continue
            updated_new_states.append(s)
        
        # TODO: REMOVE (Debug purposes)
        if len(set(skewed_states)) > 0:
            print(f'Removed {len(set(skewed_states))} skewed states!')

        return updated_new_states, updated_vis_states


    '''
    SOLVEEEEE
    '''

    async def solve(self, idx: int, state: State, namespace: str, value_cache: dict = None):
        randomness = idx
        random.seed(randomness)
        resampler = Resampler(randomness)

        # set the value of inital state
        state = replace(state, value=self.origin*self.backtrack*self.backtrack)
        input_state = state.clone()

        # log the puzzle to stdout
        print('initial problem state:')
        print(state.puzzle)

        # Records of previously visited states (state_identifier, state_value, state)
        visited_states = [("INIT", self.origin, state)]

        # set problem width
        width = self.width
        width = await self.get_width(state, idx)
        print(f'Using width {width} for {idx}') # Debug purposes

        # Initialize state for each agent
        states = [state.clone(randomness=random.randint(0, MAX_SEED)) for _ in range(width)]

        # Wrap agents in respective wrappers
        for i in range(len(self.step_agents)):
            agent_class = self.step_agents[i]['agent']
            self.step_agents[i]['agent'] = self._wrap_agent(agent_class)

        # log init stuff
        self._log_init(idx, width)

        solved = False
        terminal_states = []
        for step in range(self.num_steps):
            print(f"Step {step} ({idx})")

            # mutate using agents
            new_states, agents_used = await self.mutate_states(states, namespace, idx, step)
            # evaluate all the new states
            new_states, terminal_idxs, solved_idxs = await self.evaluate_states(new_states, value_cache, namespace, idx, step)

            # log step
            self._log_step(idx, step, states, new_states, agents_used, terminal_idxs, solved_idxs)

            # runtime updates for width
            width = self.update_width(states, new_states, visited_states, resampler, terminal_idxs, step)
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

            # Skewed State Detection
            new_states, visited_states = self.filter_states(input_state, states, new_states, visited_states, terminal_idxs)

            # Resample new states
            states, visited_states = self.resample_states(resampler, states, new_states, visited_states, terminal_idxs, step, width)

            if len(states) == 0:
                break

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

        # log the priors
        self._log_end()
        print(self.priors) # Debug purposes

        return results
