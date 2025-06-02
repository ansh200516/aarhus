import asyncio
from openai import AsyncOpenAI
from src.cachesaver.typedefs import Request, Batch,Response, SingleRequestModel, BatchRequestModel
from src.cachesaver.pipelines import OnlineAPI
from diskcache import Cache
from typing import List
import os
from dotenv import load_dotenv
load_dotenv

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)  


class LLM(SingleRequestModel, BatchRequestModel):
    def __init__(self,client:AsyncOpenAI,model:str='gpt-4'):
        self.client=client
        self.model=model
    async def request(self,request:Request)->Response:
        completion=await client.chat.completions.create(
            messages=[{"role":"user","content":request.prompt}],
            model=self.model,
            n=request.n,
            max_completion_tokens=10
        )
        response=Response([choice.message.content for choice in completion.choices])
        return response
    async def batch_request(self,batch:Batch)->List[Response]:
        requests=[self.request(request) for request in batch.requests]
        completions=await asyncio.gather(*requests)
        return completions


async def main()->List[any]:     
    model=LLM(
        client=client,
        model='gpt-4'
    )  
    
    request1=Request(
        prompt="What is the meaning of life",
        n=3,
        request_id="sth1",
        namespace="sth"
    )
    
    request2=Request(
        prompt="What is 2+2",
        n=2,
        request_id="abc_09",
        namespace="trial"
    )
    
    cache = Cache(f"./caches/test")
    
    batch=Batch(requests=[request1,request2])
    
    pipline=OnlineAPI(
        model=model,
        cache=cache,
        batch_size=2,
        timeout=1
    )
    
    # response=await pipline.batch_request(batch=batch)
    response=await pipline.request(request=request1)
    
    return response

if __name__ == "__main__":
    response=asyncio.run(main())
    print(response)
