###################
###---Prompts---###
###################
act = """Solve a question answering task with sequential Action steps. Action can be three types:

(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.

Below some examples are given. The examples also include the observations after each action, which you should not use in your answer.

{examples}

(END OF EXAMPLES)

Remember, your task is to find the immediate next action. Answer in the format given by the examples and mention nothing more.

Question: {question}
{current_state}"""

react = """Solve a question answering task with interleaving Thought and Action steps. Thought can reason about the current situation, and Action can be three types:

(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.

Below some examples are given. The examples also include the observations after each action, which you should not use in your answer.

{examples}

(END OF EXAMPLES)

Remember, your task is to find the immediate next thought and action. Answer them in the format given by the examples and mention nothing more. Limit your response to only one thought and one action.

Question: {question}
{current_state}"""

react_with_reflect = """Solve a question answering task with interleaving Thought and Action steps. Thought can reason about the current situation, and Action can be three types:

(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.

{reflections_header}{reflections}

Below some examples are given. The examples also include the observations after each action, which you should not use in your answer.

{examples}

(END OF EXAMPLES)

Remember, your task is to find the immediate next thought and action. Answer them in the format given by the examples and mention nothing more.

Question: {question}
{current_state}"""

bfs = """We're solving a question answering task with sequential Action steps. Your task is to propose multiple possible next actions given the current trajectory. Action can be three types:

(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task. When you provide your answer, only state the essential information, without full sentences or explanations.


You may take as many steps as necessary.

Below some examples are given. The examples also include the observations after each action, which you should not use in your answer.

{examples}

(END OF EXAMPLES)

Remember, your task is to propose multiple immediate next actions. Answer in the format given by the examples and mention nothing more.

Question: {question}
{current_state}

Possible Actions:
"""

evaluate_with_reflect = '''Analyze the trajectories of a solution to a question answering
task. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions that can be three types: 
(1) Search[entity]: In this case, your evaluation should be influenced based on whether useful information is found in the resulting observation.
(2) Lookup[keyword]: ]: In this case, your evaluation should be influenced based on whether useful information is found in the resulting observation.
(3) Finish[answer]: In this case, your evaluation should be influenced based on whether the answer is correct or not which will be presented in the resulting observation.

Given a question and a trajectory, evaluate its correctness and provide your reasoning and analysis in detail. Focus on the latest available thought, action, and observation. Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. Then at the last line conclude with your value estimation which can be an integer number from 1 to 10.

Below some examples are give.

{examples}

(END OF EXAMPLES)

Remember, your task is to evaluate the correctness of the available thoughts, action, and observation based on your reasoning analysis. Answer in the format given by the examples and mention nothing more. Make sure to indicate the correctness score at the end of your answer in the following format: "Correctness score : <score>".

Question: {question}
{current_state}

Thoughts:
{reflections}
(END OF THOUGHTS)

Evaluation:
'''

evaluate = '''Analyze the trajectories of a solution to a question answering
task. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions that can be three types: 
(1) Search[entity]: In this case, your evaluation should be influenced based on whether useful information is found in the resulting observation.
(2) Lookup[keyword]: ]: In this case, your evaluation should be influenced based on whether useful information is found in the resulting observation.
(3) Finish[answer]: In this case, your evaluation should be influenced based on whether the answer is correct or not which will be presented in the resulting observation.

Given a question and a trajectory, evaluate its correctness and provide your reasoning and analysis in detail. Focus on the latest available thought, action, and observation. Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Do not generate additional thoughts or actions. Then at the last line conclude with your value estimation which can be an integer number from 1 to 10.

Below some examples are give.

{examples}

(END OF EXAMPLES)

Remember, your task is to evaluate the correctness of the latest available thought (if available), action, and observation based on your reasoning analysis. Answer in the format given by the examples and mention nothing more. Make sure to indicate the correctness score at the end of your answer in the following format: "Correctness score : <score>".

Question: {question}
{current_state}

Evaluation:
'''

aggregate = '''Analyze the trajectories of a solution to a question answering
task. The trajectories are labeled by environmental observations about the situation and actions that can be three types:
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.

Given a question, trajectories and possible actions, select {k} actions that you believe are the best and most relevant to the question. Focus on the latest available action and observation, where you should only select actions from the possible actions. Do not generate additional thoughts or actions. Return only the selected actions in the format given by the examples.

Below some examples are given.

{examples}

(END OF EXAMPLES)

Remember, your task is to select the {k} best actions from the possible actions. Answer in the format given by the examples and mention nothing more.

Question: {question}
{current_state}
possible actions:
{actions}

Selected actions:
'''

reflect = """You are an advanced reasoning agent that can improve based on self reflection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps, or your actions were inefficient. In a few sentences, Diagnose a possible reason for failure or inefficiency and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.
Here are some examples:

{examples}

(END OF EXAMPLES)

Previous trial:
Question: {question}
{scratchpad}

Reflection:"""

self_evaluate_step = '''You are evaluating a reasoning step in a question answering task. Given the current state and the proposed step, determine if this step is correct and logical. Consider:
1. Is the search/lookup action relevant to finding the answer?
2. Is the thought process logical and focused on the question?
3. Does it follow the rules of using Search, Lookup, and Finish actions appropriately?

Current state: {current_state}
Proposed step: {step}

Is this reasoning step correct? Answer with a single word: Yes or No.
'''

self_evaluate_answer = '''You are evaluating a complete solution to a question answering task. Given the question, the steps taken, and the final answer, determine if the solution is correct. Consider:
1. Does it use appropriate search and lookup actions to find relevant information?
2. Are all actions logically connected and relevant to the question?
3. Does it correctly answer the question based on the information found?
4. Are the steps taken efficient and focused?

Question: {question}

Steps taken:
{steps}

Final answer: {answer}

Is this solution correct? Answer with a single word: Yes or No.
'''

act_with_reflect = """Solve a question answering task with sequential Action steps. Action can be three types:

(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.

{reflections_header}{reflections}

Below some examples are given. The examples also include the observations after each action, which you should not use in your answer.

{examples}

(END OF EXAMPLES)

Remember, your task is to find the immediate next action. Answer in the format given by the examples and mention nothing more.

Question: {question}
{current_state}"""

bfs_with_reflect = """We're solving a question answering task with sequential Action steps. Your task is to propose multiple possible next actions given the current trajectory. Action can be three types:

(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task. When you provide your answer, only state the essential information, without full sentences or explanations.

{reflections_header}{reflections}

You may take as many steps as necessary.

Below some examples are given. The examples also include the observations after each action, which you should not use in your answer.

{examples}

(END OF EXAMPLES)

Remember, your task is to propose multiple immediate next actions. Answer in the format given by the examples and mention nothing more.

Question: {question}
{current_state}

Possible Actions:
"""

################################
###---Examples for fewshot---###
################################
REFLECTION_HEADER = 'You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'

examples_bfs = [
"""Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?

Possible Actions:
Search[Adam Clayton Powell (film)]
Search[The Saimaa Gesture (film)]
Search[Finish rock music]
Search[Finish documentaries]
Search[Juice Leskinen]
Search[Documentary film]
""",

"""Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.

Possible Actions:
Lookup[named after]
Lookup[Allie Goertz]
Lookup[Matt Groening]
Lookup[name]
Search[Allie Goertz]
Search[The Simpsons]
Search[Allie Goertz Simspons]
""",

"""Question: What profession does Nicholas Ray and Elia Kazan have in common?
Action 1: Search[Nicholas Ray]
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Action 2: Search[Elia Kazan]
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.

Possible Actions:
Finish[director, screenwriter, actor]
Finish[film director, screenwriter, actor]
Finish[director, screenwriter and actor]
Lookup[Nicholas Ray]
Lookup[profession]
Lookup[producer]""",
]


examples_act = [
"""Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
Action 2: Lookup[eastern sector]
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
Action 3: Search[High Plains]
Observation 3: High Plains refers to one of two distinct land regions:
Action 4: Search[High Plains (United States)]
Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]
Action 5: Finish[1,800 to 7,000 ft]""",

"""Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 
Action 3: Finish[Richard Nixon]"""

"""Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Action 1: Search[Adam Clayton Powell]
Observation 1: Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].
Action 2: Search[Adam Clayton Powell (film)]
Observation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.
The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.
Action 3: Finish[The Saimaa Gesture]""",

"""Question: What profession does Nicholas Ray and Elia Kazan have in common?
Action 1: Search[Nicholas Ray]
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Action 2: Search[Elia Kazan]
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Action 3: Finish[director, screenwriter, actor]""",

"""Question: Which magazine was started first Arthur's Magazine or First for Women?
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century. 
Action 2: Search[First for Women]
Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989. 
Action 3: Finish[Arthur's Magazine]""",

"""Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Action 1: Search[Pavel Urysohn]
Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 â August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.
Action 2: Search[Leonid Levin]
Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist. 
Action 3: Finish[yes]"""]

examples_react = [
"""Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
Thought 2: It does not mention the eastern sector. So I need to look up eastern sector.
Action 2: Lookup[eastern sector]
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
Thought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.
Action 3: Search[High Plains]
Observation 3: High Plains refers to one of two distinct land regions:
Thought 4: I need to instead search High Plains (United States).
Action 4: Search[High Plains (United States)]
Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]
Thought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action 5: Finish[1,800 to 7,000 ft]""",

"""Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 
Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action 3: Finish[Richard Nixon]"""

"""Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Thought 1: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.
Action 1: Search[Adam Clayton Powell]
Observation 1: Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].
Thought 2: To find the documentary, I can search Adam Clayton Powell (film).
Action 2: Search[Adam Clayton Powell (film)]
Observation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.
The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.
Thought 3: Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.
Action 3: Finish[The Saimaa Gesture]""",

"""Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought 1: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.
Action 1: Search[Nicholas Ray]
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Thought 2: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.
Action 2: Search[Elia Kazan]
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Thought 3: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Action 3: Finish[director, screenwriter, actor]""",

"""Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century. 
Thought 2: Arthur's Magazine was started in 1844. I need to search First for Women next.
Action 2: Search[First for Women]
Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989. 
Thought 3: First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.
Action 3: Finish[Arthur's Magazine]""",

"""Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.
Action 1: Search[Pavel Urysohn]
Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 â August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.
Thought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.
Action 2: Search[Leonid Levin]
Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist. 
Thought 3: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work. 
Action 3: Finish[yes]"""]

examples_evaluate = [
"""Question: Which other Mexican Formula One race car driver has held the podium besides the Force India driver born in 1990?
Thought 1: I need to search for a list of Mexican Formula One race car drivers, then eliminate the one who drove for Force India and was born in 1990.
Action 1: Search[Formula One drivers from Mexico]
Observation 1: There have been six Formula One drivers from Mexico who have taken part in races since the championship began in 1950. Sergio P\u00e9rez is the most successful, having won six Grands Prix. Since 2021, P\u00e9rez drives for the Red Bull team.. P\u00e9rez is the only Mexican F1 driver to achieve a pole position. He took his first pole position at the 2022 Saudi Arabian Grand Prix, in his 219th Grand Prix meeting (a record for the most Grands Prix before a pole).
Thought 2: Sergio P\u00e9rez, born in 1990 and drove for Force India, is not the Mexican driver we are looking for. The passage does not give me enough information about other Mexican drivers who have achieved a podium finish. I need to look up more information.
Action 2: Lookup[Formula One drivers from Mexico]
Observation 2: No more results.
Thought 3: The available information doesn't allow me to determine which other Mexican Formula One race car driver, besides Sergio P\u00e9rez, has held the podium. With the search function failing to provide a full list of Mexican Formula One drivers, I need to return different search results. Let's try searching for Mexican Grand Prix winners, as they would have held the podium.
Action 3: Search[Mexican Grand Prix winners]
Observation 3: Could not find Mexican Grand Prix winners. Similar: ['Mexican Grand Prix', 'List of Formula One Grand Prix winners', '1990 Mexican Grand Prix', '2018 Mexican Grand Prix', '2019 Mexican Grand Prix'].
Thought 4: Given the difficulties in finding a direct list of Mexican Grand Prix winners, I should search for the Mexican Grand Prix to get a broader picture of the race's history. This might include winners.
Action 4: Search[Mexican Grand Prix]

Evaluation:
This trajectory is incorrect as my search should be related to Mexican Formula One race car drivers, not winners of the Mexican Grand Prix, a seperate event. A better search would have been for the List of Formula One Grand Prix winners, as suggested.
Thus the correctness score is 3""",

"""Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844\u00e2\u0080\u00931846) was an American literary periodical published in Philadelphia in the 19th century. Edited by Timothy Shay Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others.[1][2] In May 1846 it was merged into Godey's Lady's Book.[3]

Evaluation:
This trajectory is correct as it is reasonable to search for the first magazine provided in the question. It is also better to have simple searches corresponding to a single entity, making this the best action.
Thus the correctness score is 10""",

"""Question: The football manager who recruited David Beckham managed Manchester United during what timeframe?Question: The football manager who recruited David Beckham managed Manchester United during what timeframe?
Thought 1: I need to search for the football manager who recruited David Beckham, and then find out the timeframe in which he managed Manchester United.
Action 1: Search[David Beckham's football manager]
Observation 1: Could not find David Beckham's football manager. Similar: ['David Beckham', 'Inter Miami CF (redirect from Miami Beckham United)', 'Victoria Beckham', 'The Beckham Experiment', 'Miami Freedom Park'].
Thought 2: I couldn't find the information on David Beckham's football manager. I need to search Manchester United and find out the timeframe of the manager.
Action 2: Search[Manchester United]
Observation 2: Manchester United Football Club, commonly referred to as Man United (often stylised as Man Utd), or simply United, is a professional football club based in Old Trafford, Greater Manchester, England. The club competes in the Premier League, the top division in the English football league system. Nicknamed the Red Devils, they were founded as Newton Heath LYR Football Club in 1878, but changed their name to Manchester United in 1902. After a spell playing in Clayton, Manchester, the club moved to their current stadium, Old Trafford, in 1910.. Domestically, Manchester United have won a record 20 league titles, 12 FA Cups, six League Cups and a record 21 FA Community Shields.
Thought 3: The information about David Beckham's football manager is not available. I will search for Manchester United's managers to find the timeframe.
Action 3: Search[Manchester United managers]
Observation 3: Manchester United Football Club is a professional football club based in Old Trafford, Greater Manchester, England. The club was formed in Newton Heath in 1878 as Newton Heath LYR F.C., and played their first competitive match in October 1886, when they entered the First Round of the 1886–87 FA Cup. The club was renamed Manchester United F.C. in 1902, and moved to Old Trafford in 1910.. From the beginning of the club's official managerial records in 1892 to the start of the 2022–23 season, Manchester United have had 23 full-time managers.
Thought 4: The information about David Beckham's football manager is not available. The question asks for the timeframe when the manager managed Manchester United, so I need to find a list of Manchester United managers.
Action 4: Search[list of Manchester United managers]
Observation 4: Manchester United Football Club is a professional football club based in Old Trafford, Greater Manchester, England. The club was formed in Newton Heath in 1878 as Newton Heath LYR F.C., and played their first competitive match in October 1886, when they entered the First Round of the 1886–87 FA Cup. The club was renamed Manchester United F.C. in 1902, and moved to Old Trafford in 1910.. From the beginning of the club's official managerial records in 1892 to the start of the 2022–23 season, Manchester United have had 23 full-time managers.

Evaluation:
This trajectory is mostly incorrect as I was not able to retrieve the relevant information to answer the question. The searches are too broad and not relevant to the specific player recruited, David Beckham. The last two actions are also too similar. It would be better to first search for David Beckham and lookup information related to his manager.
Thus the correctness score is 4""",

"""Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 
Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action 3: Finish[President Richard Nixon]

Evaluation:
This trajectory is correct as all of my thoughts and actions are correct. It makes sense to search for Milhouse first as it is the central subject of the question. It is also correct to directly look up the relevant information in the article, instead of trying another search. 
Thus the correctness score is 10"""
]

examples_aggregate = [
"""
Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?

Possible Actions:
Search[Adam Clayton Powell (film)]
Search[The Saimaa Gesture (film)]
Search[Finish rock music]
Search[Finish documentaries]
Search[Juice Leskinen]
Search[Documentary film]

Selected actions:
Search[Adam Clayton Powell (film)]
Search[The Saimaa Gesture (film)]
Search[Finish documentaries]
""",

"""Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.

Possible Actions:
Lookup[named after]
Lookup[Allie Goertz]
Lookup[Matt Groening]
Lookup[name]
Search[Allie Goertz]
Search[The Simpsons]
Search[Allie Goertz Simspons]

Selected actions:
Lookup[named after]
Lookup[name]
Search[The Simpsons]
""",

"""Question: What profession does Nicholas Ray and Elia Kazan have in common?
Action 1: Search[Nicholas Ray]
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Action 2: Search[Elia Kazan]
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.

Possible Actions:
Finish[director, screenwriter, actor]
Finish[film director, screenwriter, actor]
Finish[director, screenwriter and actor]
Lookup[Nicholas Ray]
Lookup[profession]
Lookup[producer]

Selected actions:
Finish[director, screenwriter, actor]
Finish[film director, screenwriter, actor]
Finish[director, screenwriter and actor]""",
]

examples_reflect = [
"""Previous Trial:
Question: The Rome Protocols were signed by three Prime Ministers one of which was assassinated as part of what?
Action 1: Search[Rome Protocols]
Observation 1: The Rome Protocols were a series of three international agreements concluded in Rome on 17 March 1934 between the governments of Austria, Hungary and Italy. They were signed by Italian Prime Minister Benito Mussolini, Austrian Chancellor Engelbert Dollfuss and Hungarian Prime Minister Gyula Gömbös. All of the protocols went into effect on 12 July 1934 and were registered in League of Nations Treaty Series on 12 December 1934.
Action 2: Search[Benito Mussolini]
Observation 2: Benito Amilcare Andrea Mussolini (...) was an Italian politician (...) summary execution in 1945 by Italian partisans. (...)
Action 3: Finish[World War II]

Reflection: I searched one of the prime ministers involved in the signing then attempted to answer right away. I should have searched each of the prime ministers, then looked up 'death' or 'assassinated' on each of their pages to get more specific information before answering, or checked if all three were assassinated. The question specified only one was assassinated. I should focus on Engelbert Dollfuss or Gyula Gömbös. My final answer "World War II" was too broad. I need to verify the specific assassination event for the correct prime minister.
""",

"""Previous Trial:
Question: Kam Heskin plays Paige Morgan in a 2004 film directed by who?
Action 1: Search[Kam Heskin]
Observation 1: Kam Erika Heskin (born May 8, 1973) is an American actress. She began her career playing Caitlin Deschanel on the NBC daytime soap opera Sunset Beach (1998–1999), before appearing in films Planet of the Apes (2001) and Catch Me If You Can (2002). Heskin went on in 2003 to play Elizabeth Bennet in an independent film Pride & Prejudice: A Latter-Day Comedy, Robin in Charmed (2004), and Paige Morgan in The Prince and Me film franchise (2006–2010).
Action 2: Search[The Prince and Me film franchise]
Observation 2: Could not find [The Prince and Me film franchise]. Similar: ['Kam Heskin', 'Prince of Persia', 'Harry Potter and the Half-Blood Prince (film)', ...]
Action 3: Search[The Prince and Me (2004 film)]
Observation 3: Could not find [The Prince and Me (2004 film)]. Similar: ['The Prince & Me', 'The Prince & Me 2: The Royal Wedding', ...]
Action 4: Search[The Prince & Me (2004 film)]
Observation 4: Could not find [The Prince & Me (2004 film)]. Similar: ['The Prince & Me', ...]
Action 5: Search[The Prince & Me (2004 film)]
Observation 5: Could not find [The Prince & Me (2004 film)]. Similar: ['The Prince & Me', ...]
Action 6: Search[The Prince & Me (2004 film)]
Observation 6: Could not find [The Prince & Me (2004 film)]. Similar: ['The Prince & Me', ...]

Reflection: I got stuck in a loop trying to search for 'The Prince and Me (2004 film)' which repeatedly failed. The initial search for Kam Heskin mentioned she played Paige Morgan in "The Prince and Me film franchise (2006-2010)" and "Robin in Charmed (2004)". The question specifically asks about a 2004 film. I missed that "Charmed (2004)" was mentioned for Kam Heskin. I should have investigated "Charmed (2004)" or, if "The Prince and Me" was indeed a 2004 film, I should have used the similar search result 'The Prince & Me' that was offered and seems more likely to be the correct film title. I need to be more careful with dates and explore provided similar search results when direct searches fail.
"""
]

examples_contrast_evaluate = [
'''Question: Who is the current drummer of the band who did the song "What Lovers Do"?

EXAMPLE_TRAJECTORY_1:
Action 1: Thought 1: I need to identify the band that performed "What Lovers Do" and then find out who the current drummer of that band is.
Action 1: Search["What Lovers Do" song]
Observation 1: "What Lovers Do" is a song by American singer Maroon 5, featuring SZA.
Thought 2: The band is Maroon 5. Now, I should search for the current drummer of Maroon 5.
Action 2: Search[Maroon 5 band members]
Observation 2: As of the latest information, the band Maroon 5's current drummer is Matt Flynn.
Observation 1: Invalid Action. Valid Actions are Lookup[<topic>], Search[<topic>] and Finish[<answer>].

(END OF EXAMPLE_TRAJECTORY_1)

EXAMPLE_TRAJECTORY_2:
Action 1: Search[Maroon 5]
Observation 1: Maroon 5 is an American pop rock band from Los Angeles, California. It consists of lead vocalist and rhythm guitarist Adam Levine, rhythm guitarist and keyboardist Jesse Carmichael, lead guitarist James Valentine, drummer Matt Flynn, keyboardist PJ Morton, and bassist and sampler Sam Farrar. Original members Levine, Carmichael, bassist Mickey Madden, and drummer Ryan Dusick first came together as Kara's Flowers in 1994, while they were in high school.
After self-releasing their independent album We Like Digging?, the band signed to Reprise Records and released the album The Fourth World in 1997. The album garnered a tepid response, after which the record label dropped the band and the members focused on college. In 2001, the band re-emerged as Maroon 5, pursuing a different direction and adding guitarist Valentine. The band signed with Octone Records, an independent record label with a separate joint venture relationship with J Records and released their debut album Songs About Jane in June 2002. Aided by the hit singles "Harder to Breathe", "This Love" and "She Will Be Loved", the album peaked at number six on the Billboard 200 chart and went quadruple platinum in 2005. In the same year, the band won the Grammy Award for Best New Artist. In 2006, Dusick left the band after suffering from serious wrist and shoulder injuries and was replaced by Matt Flynn.
The band's second album It Won't Be Soon Before Long was released in May 2007. It debuted at number one on the US Billboard 200 chart and the lead single "Makes Me Wonder", became the band's first number-one single on the Billboard Hot 100. In 2010, the band released their third album Hands All Over, to favorable reviews, re-releasing a year later to include the single "Moves like Jagger", which topped the Billboard Hot 100. In 2012, Carmichael left the group and was replaced by musician PJ Morton, as the band released the fourth album Overexposed, with the song "One More Night", topping the Billboard Hot 100 chart for nine consecutive weeks.
In 2014, Carmichael rejoined the band alongside Morton to record the fifth album V (roman numeral pronounced "five"), with the band signed to Interscope Records and Levine's own label 222 Records. Following the release of V, it reached number one on the Billboard 200. In 2016, Maroon 5 recruited their long-time collaborator Sam Farrar, as the band continued for the sixth studio album Red Pill Blues, which was released in November 2017. With the addition of Morton and Farrar, the band's lineup increased to seven members. The successful singles of both albums "Sugar" and "Girls Like You" peaked at numbers two and one on the Hot 100 chart respectively. Madden announced his departure from the band in 2020 following his arrest on domestic violence charges, with Farrar becoming their new bassist. The band's seventh album, Jordi, was released in June 2021. Maroon 5 has sold more than 120 million records, making them one of the world's best-selling music artists. In January 2023, it was announced "Moves like Jagger" is the second most downloaded song of all time.

(END OF EXAMPLE_TRAJECTORY_2)


Evaluation:
While both trajectories mention drummers in the band Maroon 5, EXAMPLE_TRAJECTORY_1 also mentions that Matt Flynn is the current drummer, which is correct and is relevant to the question. Hence, EXAMPLE_TRAJECTORY_1 is slightly more promising to deliver the correct answer as compared to EXAMPLE_TRAJECTORY_2.
Promising Trajectory: EXAMPLE_TRAJECTORY_1
''',

'''Question: The Walkley Heights suburb is located on land that was the home of a prison built in what year?

EXAMPLE_TRAJECTORY_1:
Action 1: Thought 1: I need to find information about Walkley Heights and its history, specifically regarding the prison and the year it was built.
Action 1: Search[Walkley Heights]
Observation 1: (Result 1 / 1) Walkley Heights is a suburb in Adelaide, South Australia, developed in the early 2000s. There is no mention of a prison there.
Thought 2: Since the suburb's development is recent, I should look into historical prisons in the area that might have been located there previously.
Action 2: Search[Prison in Walkley Heights]
Observation 2: No relevant results found. I will try a broader search.
Action 3: Search[Prison in Adelaide, South Australia]
Observation 3: The South Australian state prison was established in 1838, located in Adelaide. The original prison was the South Australian House of Correction.
Thought 4: The prison built in 1838 is significant, and perhaps the land was previously used for that prison before the suburb was developed.
Action 4: Lookup[South Australian House of Correction]
Observation 4: The South Australian House of Correction was built in 1838 and operated until 1862, after which a new prison was built.
Thought 5: The original prison was built in 1838, which aligns with the question about the prison's construction year.

(END OF EXAMPLE_TRAJECTORY_1)

EXAMPLE_TRAJECTORY_2:
Action 1: Search[Walkley Heights]
Observation 1: Walkley Heights is a suburb of Adelaide, South Australia. The suburb is located on land formerly comprising the prison farm for Yatala Labour Prison, and includes fifty-five hectares of land formerly owned by R. M. Williams which was compulsorily acquired during the time of former State Premier Sir Thomas Playford. The suburb (and one adjacent main road) is named after John Walkley, an early pioneer in South Australia
The suburb had a population, in 2001, of only 713 increasing to 3,497 by 2016.

(END OF EXAMPLE_TRAJECTORY_2)

Evaluation:
EXAMPLE_TRAJECTORY_1 follows a good logical deduction sequence but incorrect inferences are made. The trajectory assumes that the South Australian state prison is the only possible option without correlating it back to Walkey Heights, unlike EXAMPLE_TRAJECTORY_2. In EXAMPLE_TRAJECTORY_2, we observe factual data about Walkey Heights which clearly mentions that it is located on the land formerly comprising of Yatala Labour Prison. This is a much more grounded inference and will give very promising results after looking more into the Yatala Labour Prison. Hence, EXAMPLE_TRAJECTORY_2 is much more promising than EXAMPLE_TRAJECTORY_1.
Promising Trajectory: EXAMPLE_TRAJECTORY_2
'''
]

################################
###--Contrastive Evaluation--###
################################

contrast_evaluate = '''Analyze the given trajectories of a solution to a question answering task. The trajectories are labeled by environmental observations about the situation, thoughts that can reason about the current situation and actions that can be three types: 
(1) Search[entity]: In this case, your evaluation should be influenced based on whether useful information is found in the resulting observation.
(2) Lookup[keyword]: In this case, your evaluation should be influenced based on whether useful information is found in the resulting observation.
(3) Finish[answer]: In this case, your evaluation should be influenced based on whether the answer is correct or not which will be presented in the resulting observation.

Given a question and two trajectories, evaluate their logical correctness against each other and provide your reasoning and analysis in detail. Focus on the available thoughts, actions, and observations. Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet. Prioritize ground truths over anything else. Trajectories that lay a correct roadmap to solve the question without providing much information may be more promising compared to trajectories that provide a lot of information but don't link it back to the question and/or create incorrect inferences/assumptions. Do not generate additional thoughts or actions. Then at the last line conclude with your evaluation and analysis by stating which trajectory is more promising.

Below some examples are given.

{examples}

(END OF EXAMPLES)

Remember, your task is to evaluate the correctness of the latest available thoughts (if available), actions, and observations for both trajectories based on your reasoning analysis and provide a comparison between the two. Answer in the format given by the examples and mention nothing more. Do not generate additional thoughts or actions. Make sure to report the more promising trajectory at the end of your answer in the following format: "Promising Trajectory : <trajectory_id>".

Question: {question}

TRAJECTORY_1 (ID=1):
{cs_1}
Thoughts generated:
{reflections_1}

(END OF TRAJECTORY_1)

TRAJECTORY_2 (ID=2):
{cs_2}
Thoughts generated:
{reflections_2}

(END OF TRAJECTORY_2)

Evaluation:
'''

certainty = '''You are an expert at analyzing reasoning traces. Analyze the given trace for a task and report the information score for it, on a scale of 1-10. The information score is the amount of information present in the reasoning trace that is grounded and relevant to solving the problem. If you think there is enough relevant, grounded information in the trace in order to solve the problem then give a high score. If there is abscence of grounded relevant information then give a low score. If there is some relevant information available, but more information is needed to solve the problem then give it a moderate score. Information irrelevant to the task at hand should incurr a penlty to the score. I will also give you grounded information to verify the trace against.
The traces are labeled by environmental observations about the situation (which need to be verified against the grounded information that will also be provided), thoughts that can reason about the current situation and also reflect back and steer the trace into the correct direction, and actions that can be three types: 
(1) Search[entity]: In this case, your evaluation should be influenced based on whether useful information is found in the resulting observation.
(2) Lookup[keyword]: ]: In this case, your evaluation should be influenced based on whether useful information is found in the resulting observation.
(3) Finish[answer]: In this case, your evaluation should be influenced based on whether the answer is correct or not which will be presented in the resulting observation.

Given a question, a trace and grounded information, evaluate if there is enough information in the trace to solve the problem and assign it a score from 1-10 along with detailed analysis. Do not generate additional thoughts or actions. Then at the last line conclude with your information score estimation which can be an integer number from 1 to 10. Explicily state what the information score is by saying "the information score is ..."

Grounded Information:
{grounded_observations}
(END OF GROUNDED INFORMATION)

Question: {question}
{current_state}

Evaluation:
'''


evaluate_difficulty='''Evaluate how difficult it is to reach to the anser using the given puzzle by giving a difficulty rating from 1-5, 5 being the most difficult. Analyze difficulty based on high level plan and number of steps required to solve the problem, uncertainty in the problem and difficulty of intermediate steps. Follow the format given below:

Format:
Thought: <your thoughts>
Difficulty: <difficulty score between 1-5>

Please note that you are not supposed to inlucde the "<" and ">". Strictcly follow the output format. Be accurate and precise in your thoughts.

Input: {input}
'''