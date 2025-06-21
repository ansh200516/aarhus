import asyncio
from openai import AsyncOpenAI
from cachesaver.typedefs import Request, Batch,Response, SingleRequestModel, BatchRequestModel
from src.models import OnlineLLM
from src.tasks.hotpotqa.agents import *
from src.tasks.hotpotqa.state import StateHotpotQA
from src.tasks.hotpotqa.prompts import *
from diskcache import Cache
from typing import List
import os
from dotenv import load_dotenv
from langchain import Wikipedia
from langchain.agents.react.base import DocstoreExplorer

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
            max_completion_tokens=512
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

    state0 = StateHotpotQA(
        puzzle="Ladislav Matejka's Michigan Slavic Publications has published volumes by which historian considered to be the founder of morphophonology?",
        current_state='',
        steps=[],
        answer='Nikolai Trubetzkoy',
        reflections=['The failure occurred because I searched for an unrelated figure (Henry George Liddell) instead of focusing on the specific historian associated with morphophonology or the founder of that field. To improve, I should refine my search strategy by directly looking for "morphophonology" or "founder of morphophonology" and consider the possibility that the relevant scholar might be well-known in linguistic circles. A more effective plan would be to search for "morphophonology" first, then identify prominent scholars associated with that field, and verify if Ladislav Matejka\'s publisher has published works by such a scholar. This targeted approach can help avoid unrelated results and focus on the relevant historian.'],
        value=None,
        parent=None,
        randomness=10,
        docstore=None
    )

    state1 = StateHotpotQA(
        puzzle="Ladislav Matejka's Michigan Slavic Publications has published volumes by which historian considered to be the founder of morphophonology?",
        current_state='Action 1: Search[André Martinet]\nObservation 1: André Martinet (French: [ɑ̃.dʁe maʁ.ti.nɛ]; 12 April 1908 - 16 July 1999) was a French linguist, influential due to his work on structural linguistics. In linguistic theory, Martinet is known especially for his studies on linguistic economy and double articulation.',
        steps=['Action 1: Search[André Martinet]\nObservation 1: André Martinet (French: [ɑ̃.dʁe maʁ.ti.nɛ]; 12 April 1908 - 16 July 1999) was a French linguist, influential due to his work on structural linguistics. In linguistic theory, Martinet is known especially for his studies on linguistic economy and double articulation.'],
        answer='Nikolai Trubetzkoy',
        reflections=[],
        value=None,
        parent=None,
        randomness=10,
        docstore=None
    )

    # state2 = StateHotpotQA(
    #     puzzle='What surrounding river in a Canadian province was Erin Wall born near?',
    #     current_state='Action 1: Search[Erin Wall]\nObservation 1: Erin Wall (4 November 1975 – 8 October 2020) was a Canadian-American operatic soprano who had an active international career from 2001 until her death of breast cancer in 2020. Chiefly associated with the Lyric Opera of Chicago, the Vancouver-based soprano appeared with the company in fourteen productions from 2001 through 2018. She was particularly admired for her performances in the operas of Mozart and Strauss, and performed leading roles as a guest artist at important houses, including the Metropolitan Opera, La Scala, and the Paris Opera.',
    #     steps=['Action 1: Search[Erin Wall]\nObservation 1: Erin Wall (4 November 1975 – 8 October 2020) was a Canadian-American operatic soprano who had an active international career from 2001 until her death of breast cancer in 2020. Chiefly associated with the Lyric Opera of Chicago, the Vancouver-based soprano appeared with the company in fourteen productions from 2001 through 2018. She was particularly admired for her performances in the operas of Mozart and Strauss, and performed leading roles as a guest artist at important houses, including the Metropolitan Opera, La Scala, and the Paris Opera.'],
    #     answer='Bow River',
    #     reflections=['The failure in this trial occurred because I searched only Erin Wall’s biography without investigating the specific geographical or contextual clues needed to identify the surrounding river. To improve efficiency, I should formulate a targeted plan: after the initial search, I will look for any mentions of her birthplace or Canadian provinces, then search for prominent rivers associated with that province. This focused approach will help me find the relevant river more directly, reducing unnecessary searches and increasing the likelihood of correctly answering the question.'],
    #     value=None,
    #     parent=None,
    #     randomness=69,
    #     docstore=None
    # )

    s1 = StateHotpotQA(
        puzzle='Camp Courtney was named after the Major killed in action at the battle codenamed what?',
        current_state='''
Action 1: Thought 1: I need to find information about Camp Courtney and its naming history, specifically the Major killed in action and the battle codenamed what.
Action 1: Search[Camp Courtney]
Observation 1: Camp Courtney is a United States Marine Corps base located in Okinawa, Japan. It is named after Major William F. Courtney, who was killed during World War II.
Thought 2: The description mentions Major William F. Courtney but does not specify the battle codenamed what. I should look up Major William F. Courtney for more details.
Action 2: Lookup[Major William F. Courtney]
Observation 2: Major William F. Courtney was awarded the Silver Star for his actions in World War II. The specific battle codenamed is not provided here, so I need to find which battle he was killed in.
Thought 3: To find the battle codenamed, I should look for the battle or operation associated with Major William F. Courtney's death.
Action 3: Search[Major William F. Courtney battle]
Observation 3: There is limited information directly associating Major Courtney with a specific battle codenamed. However, considering the location and timeframe, he was likely killed during the Battle of Okinawa.
Thought 4: The Battle of Okinawa was a major WWII battle in Okinawa, Japan. It is well-known and significant, and often associated with US military operations in that region.
Action 4: Finish[Battle of''',
        steps=["Action 1: Thought 1: I need to find information about Camp Courtney and its naming history, specifically the Major killed in action and the battle codenamed what.\nAction 1: Search[Camp Courtney]\nObservation 1: Camp Courtney is a United States Marine Corps base located in Okinawa, Japan. It is named after Major William F. Courtney, who was killed during World War II.\nThought 2: The description mentions Major William F. Courtney but does not specify the battle codenamed what. I should look up Major William F. Courtney for more details.\nAction 2: Lookup[Major William F. Courtney]\nObservation 2: Major William F. Courtney was awarded the Silver Star for his actions in World War II. The specific battle codenamed is not provided here, so I need to find which battle he was killed in.\nThought 3: To find the battle codenamed, I should look for the battle or operation associated with Major William F. Courtney's death.\nAction 3: Search[Major William F. Courtney battle]\nObservation 3: There is limited information directly associating Major Courtney with a specific battle codenamed. However, considering the location and timeframe, he was likely killed during the Battle of Okinawa.\nThought 4: The Battle of Okinawa was a major WWII battle in Okinawa, Japan. It is well-known and significant, and often associated with US military operations in that region.\nAction 4: Finish[Battle of\nObservation 1: Invalid Action. Valid Actions are Lookup[<topic>], Search[<topic>] and Finish[<answer>]."],
        answer='Operation Iceberg',
        reflections=[],
        value=None,
        parent=None,
        randomness=10,
        docstore=None
    )

    s2 = StateHotpotQA(
        puzzle='Camp Courtney was named after the Major killed in action at the battle codenamed what?',
        current_state='Action 1: Search[Battle of Bougainville]\nObservation 1: The Bougainville campaign was a series of land and naval battles of the Pacific campaign of World War II between Allied forces and the Empire of Japan, named after the island of Bougainville, one of the Solomon Islands. It was part of Operation Cartwheel, the Allied grand strategy in the South Pacific. \nThe campaign took place in the Northern Solomons in two phases. The first phase, in which American troops landed and held the perimeter around the beachhead at Torokina, lasted from November 1943 through November 1944. The second phase, in which primarily Australian troops went on the offensive, mopping up pockets of starving, isolated but still-determined Japanese, lasted from November 1944 until August 1945, when the last Japanese soldiers on the island surrendered. Operations during the final phase of the campaign saw the Australian forces advance north towards the Bonis Peninsula and south towards the main Japanese stronghold around Buin, although the war ended before these two enclaves were completely destroyed.',
        steps=['Action 1: Search[Battle of Bougainville]\nObservation 1: The Bougainville campaign was a series of land and naval battles of the Pacific campaign of World War II between Allied forces and the Empire of Japan, named after the island of Bougainville, one of the Solomon Islands. It was part of Operation Cartwheel, the Allied grand strategy in the South Pacific. \nThe campaign took place in the Northern Solomons in two phases. The first phase, in which American troops landed and held the perimeter around the beachhead at Torokina, lasted from November 1943 through November 1944. The second phase, in which primarily Australian troops went on the offensive, mopping up pockets of starving, isolated but still-determined Japanese, lasted from November 1944 until August 1945, when the last Japanese soldiers on the island surrendered. Operations during the final phase of the campaign saw the Australian forces advance north towards the Bonis Peninsula and south towards the main Japanese stronghold around Buin, although the war ended before these two enclaves were completely destroyed.'],
        answer='Operation Iceberg',
        reflections=[],
        value=None,
        parent=None,
        randomness=10,
        docstore=None
    )

    # val, resp = await AgentEvaluateHotpotQA.act(
    #     model=model,
    #     state=state2,
    #     n=1,
    #     namespace='foonamespace',
    #     request_id='req_foo',
    #     params=DecodingParameters(
    #         max_completion_tokens=256,
    #         temperature=0.7,
    #         top_p=1,
    #         stop=None,
    #         logprobs=None,
    #     ),
    #     cache=None
    # )


    prompt = contrast_evaluate.format(
        examples='\n\n'.join(examples_contrast_evaluate),
        question=s1.puzzle,
        cs_1=s1.current_state,
        reflections_1='\n'.join(s1.reflections),
        cs_2=s2.current_state,
        reflections_2='\n'.join(s2.reflections),
    )

    resp = await model.request(
        prompt=prompt,
        n=1,
        request_id="game24_1",
        namespace="game24",
        params=None
    )

    print(resp)




if __name__ == "__main__":
    response=asyncio.run(main())


