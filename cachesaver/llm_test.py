import asyncio
from openai import AsyncOpenAI
from cachesaver.typedefs import Request, Batch,Response, SingleRequestModel, BatchRequestModel
from src.models import OnlineLLM
from src.tasks.game24.agents import AgentTerminalReflexionGame24, AgentActGame24, AgentReactGame24
from src.tasks.game24.state import StateGame24
from diskcache import Cache
from typing import List
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)  


class LLM(SingleRequestModel, BatchRequestModel):
    def __init__(self,client:AsyncOpenAI,model:str='gpt-4'):
        self.client=client
        self.model=model
    
    async def request(self, prompt, n, request_id, namespace, params)->Response:
        completion=await client.chat.completions.create(
            messages=[{"role":"user","content":prompt}],
            model='gpt-4.1-nano',
            n=1,
            temperature=0.7,
            max_completion_tokens=256
        )
        response=Response([choice.message.content for choice in completion.choices])
        return response.data

    async def batch_request(self,batch:Batch)->List[Response]:
        requests=[self.request(request) for request in batch.requests]
        completions=await asyncio.gather(*requests)
        return completions


async def main()->List[any]:     
    model=LLM(
        client=client,
        model='gpt-4.1-nano'
    )
    new_state = await AgentReactGame24.act(
        model=model,
        state=StateGame24(
            puzzle="4 5 7 8",
            current_state="16 8",
            steps=['4 + 5 = 9 (left: 9 7 8)', '9 + 7 = 16 (left: 16 8)'],
            randomness=42,
            context=""
        ),
        n=1,
        namespace="game24",
        request_id="game24_1",
        params=None
    )

    new_state = new_state[0]
    return new_state

if __name__ == "__main__":
    response=asyncio.run(main())
    print(response)



'''



4 + 4 = 8 (left: 8 8 8)  
4 - 4 = 0 (left: 8 8 0)  
8 - 4 = 4 (left: 4 8 8)  
8 + 4 = 12 (left: 4 8 12)  
8 / 4 = 2 (left: 8 2 8)













'''
