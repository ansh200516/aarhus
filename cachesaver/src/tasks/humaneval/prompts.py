SIMPLE_CHAT_INSTRUCTION = "You are a programming assistant. You will be given a function signature and docstring. You should fill in the following text of the missing function body. For example, the first line of the completion should have 4 spaces for the indendation so that it fits syntactically with the preceding signature."
SIMPLE_CHAT_INSTRUCTION_V2 = """You are an AI that only responds with only {lang} code. You will be given a function signature and its docstring by the user. Write your full implementation (restate the function signature)."""

aggregate_prompt = """You are a programming assistant, who is helping user to write efficient and correct codes. You will be given multiple implementations of the same function. You should choose the {k} best implementation based on the following criterias:
1. Correctness: The implementation should return the correct output.
2. Efficiency: The implementation should be efficient in terms of time and space complexity.
3. Readability: The implementation should be readable and understandable.
4. Style: The implementation should follow the style guide of the language.
5. Testability: The implementation should be testable.

Remember your task is to choose the {k} best implementation based on the above criterias. Make sure to return only the indexes of the selected implementations, separated by commas. Do not include any other explanations, introduction, conclusions or thoughts. Just return the indexes of the selected implementations.
Function signature and docstring:
{prompt}
Implementations:
{implementations}
Chosen implementation:
"""

evaluation_prompt = """You are a programming assistant, who is helping the user to evaluate a generated code. You will be given a single implementation of a function, and you should evaluate it based on the following criteria:

1. **Correctness**: Does the implementation return the correct output for different inputs?
2. **Efficiency**: Is the implementation efficient in terms of time and space complexity?
3. **Readability**: Is the code readable and understandable? Is it easy to follow?
4. **Style**: Does the implementation follow the style guide of the language (naming conventions, indentation, etc.)?
5. **Testability**: Is the implementation testable? Can it be easily tested with unit tests?

Evaluate the code on each criterion with a score from 1 to 10 (integers only, no fractions). Then give an overall score as the sum of all scores.

Function signature and docstring:
{prompt}

Implementation:
{implementation}

Evaluation scores:
- Correctness: <score>
- Efficiency: <score>
- Readability: <score>
- Style: <score>
- Testability: <score>

Overall Score: <final score>

Do not include any further thoughts or reasoning, just the evaluation scores and the final overall score."""


SIMPLE_CHAT_INSTRUCTION_BFS = """
You are an AI that only responds with {lang} code. You will be given a function signature and its docstring by the user.
Write multiple full implementations (at least two), each restating the function signature. Use a different approach for each.
Mark the start and end of each implementation using triple backticks, like this:
\`\`\`
<Code implementation here>
\`\`\`
Each implementation should be fully contained within its own set of backticks, without any additional markers.
"""

react = """You are a programming assistant solving a coding task. Think step by step and plan your implementation carefully. You will be given a function signature and docstring, and you need to implement the function.

For each step:
1. Think about what needs to be done
2. Write code to implement that step
3. Consider edge cases and error handling

Example:
Function signature and docstring:
def add_numbers(a: int, b: int) -> int:
    '''Add two numbers and return the result.'''

Thought: I need to implement a simple addition function. The function takes two integers and returns their sum. I should handle basic input validation.

Action: ```python
def add_numbers(a: int, b: int) -> int:
    '''Add two numbers and return the result.'''
    # Input validation
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("Both arguments must be integers")
    return a + b
```

Thought: The implementation looks good. It includes:
1. Type hints for parameters and return value
2. Input validation to ensure both arguments are integers
3. Simple and efficient addition operation
4. Proper docstring preservation

Action: Finish[The implementation is complete and correct]

Function signature and docstring:
{prompt}

Current implementation:
{current_state}

Remember to think step by step and write clear, efficient code."""

self_evaluate_step = '''You are evaluating a reasoning step in a code generation task. Given the function signature, current implementation, and the proposed step, determine if this step is correct and logical. Consider:
1. Is the code syntactically correct?
2. Does it follow the function's requirements?
3. Is it a logical next step in the implementation?
4. Does it handle edge cases appropriately?

Function signature and docstring:
{prompt}

Current implementation:
{current_state}

Proposed step:
{step}

Is this reasoning step correct? Answer with a single word: Yes or No.
'''

self_evaluate_answer = '''You are evaluating a complete solution to a code generation task. Given the function signature, the implementation steps, and the final code, determine if the solution is correct. Consider:
1. Does the implementation match the function signature and docstring?
2. Is the code syntactically correct and follows language style guidelines?
3. Does it handle all edge cases and error conditions?
4. Is it efficient and readable?
5. Does it include appropriate tests or validation?

Function signature and docstring:
{prompt}

Implementation steps:
{steps}

Final code:
{answer}

Is this solution correct? Answer with a single word: Yes or No.
'''


reflect = """You are a {lang} programming assistant, who is helping the user to improve their code. You will be given a function implementation and a series of unit tests. Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. You will need this as a hint when you try again later. Only provide the few sentence description in your answer, not the implementation.":

{examples}

(END OF EXAMPLES)

Previous trial:
function impl: {function_impl}
failed tests: {failed_tests}
Reflection:"""

examples_reflect_py = [
'''Previous trial:
function impl:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 >= max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```
failed tests: 
assert longest_subarray_with_sum_limit([5, 6, 7, 8, 9], 4) == [] # output: [5]

reflection:
The implementation failed the where no subarray fulfills the condition. The issue in the implementation is due to the use of >= instead of > in the condition to update the result. Because of this, it returns a subarray even when the sum is greater than the target, as it still updates the result when the current subarray length is equal to the previous longest subarray length. To overcome this error, we should change the condition to only update the result when the current subarray length is strictly greater than the previous longest subarray length. This can be done by replacing >= with > in the condition.''',
'''Previous trial:
function impl:
```python
def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:
    n = len(nums)
    left, right = 0, 0
    max_length = 0
    current_sum = 0
    result = []
    while current_sum + nums[right] <= target:
        current_sum += nums[right]
        right += 1
    while right < n:
        current_sum += nums[right]
        while current_sum > target:
            current_sum -= nums[left]
            left += 1
        if right - left + 1 > max_length:
            max_length = right - left + 1
            result = nums[left:right+1]
        right += 1
    return result
```
failed tests:
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 8) == [1, 2, 3] # output: list index out of range
assert longest_subarray_with_sum_limit([1, 2, 3, 4, 5], 15) == [1, 2, 3, 4, 5] # output: list index out of range
assert longest_subarray_with_sum_limit([5, 6, 7, 8, 9], 4) == [] # output: list index out of range
assert longest_subarray_with_sum_limit([1, -1, 2, -2, 3, -3], 2) == [1, -1, 2, -2, 3] # output: list index out of range

reflection:
The implementation failed 4 out of the 7 test cases due to an IndexError. The issue stems from the while loop while current_sum + nums[right] <= target:, which directly accesses nums[right] without checking if right is within the bounds of the list. This results in a runtime error when right goes beyond the list length. To overcome this error, we need to add a bounds check for the right variable in the mentioned while loop. We can modify the loop condition to while right < len(nums) and current_sum + nums[right] <= target:. This change will ensure that we only access elements within the bounds of the list, thus avoiding the IndexError.
''',
'''Previous trial:
function impl:
```python
def add(a: int, b: int) -> int:
    """
    Given integers a and b, return the total value of a and b.
    """
    return a - b
```

failed tests:
assert add(1, 2) == 3 # output: -1
assert add(1, 2) == 4 # output: -1

reflection:
The implementation failed the test cases where the input integers are 1 and 2. The issue arises because the code does not add the two integers together, but instead subtracts the second integer from the first. To fix this issue, we should change the operator from `-` to `+` in the return statement. This will ensure that the function returns the correct output for the given input.''',

'''Previous trial:
function impl:
```python
from typing import *
def fullJustify(words: List[str], maxWidth: int) -> List[str]:
    """
    Given an array of words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.
    You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces `' '` when necessary so that each line has exactly maxWidth characters.
    Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.
    For the last line of text, it should be left justified and no extra space is inserted between words.
    Note:
    A word is defined as a character sequence consisting of non-space characters only.
    Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
    The input array `words` contains at least one word.
    """
    res = []
    cur_line = []
    cur_len = 0

    for word in words:
        if cur_len + len(word) + len(cur_line) > maxWidth:
            if len(cur_line) == 1:
                res.append(cur_line[0] + ' ' * (maxWidth - cur_len))
            else:
                spaces = maxWidth - cur_len
                space_between = spaces // (len(cur_line) - 1)
                extra_spaces = spaces % (len(cur_line) - 1)
                line = ''
                for i, w in enumerate(cur_line[:-1]):
                    line += w + ' ' * (space_between + (i < extra_spaces))
                line += cur_line[-1]
                res.append(line)
            cur_line = []
            cur_len = 0
        cur_line.append(word)
        cur_len += len(word)

    last_line = ' '.join(cur_line)
    last_line += ' ' * (maxWidth - len(last_line))
    res.append(last_line)

    return res
```

failed tests:
assert fullJustify([], 10) == [] # output: ['          ']
assert fullJustify([], 0) == [] # output: ['']

reflection:
The implementation failed the test cases where the input list of words is empty. The issue arises because the code does not handle the case where there are no words to process. As a result, it still appends a line with spaces to the result list, even when there are no words. To fix this issue, we should add a condition at the beginning of the function to check if the input list is empty, and return an empty list if it is. This will ensure that the function returns the correct output for empty input lists.
END EXAMPLES

'''

]




examples_reflect_rs = [
'''Previous trial:
function impl:
```rust
fn add(a: i32, b: i32) -> i32 {
    // Given integers a and b, return the total value of a and b.
    a - b
}
```

failed tests:
assert_eq!(add(1, 2), 3); // output: -1
assert_eq!(add(1, 2), 4); // output: -1

reflection:
The implementation failed the test cases where the input integers are 1 and 2. The issue arises because the code does not add the two integers together, but instead subtracts the second integer from the first. To fix this issue, we should change the operator from `-` to `+` in the return statement. This will ensure that the function returns the correct output for the given input.
''',

'''Previous trial:
function impl:
```rust
pub fn group_anagrams(strs: Vec<String>) -> Vec<Vec<String>> {
// Given an array of strings strs, group the anagrams together. You can return the answer in any order.
// An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.
  use std::collections::HashMap;
  let mut map: HashMap<[u8;26], Vec<String>> = HashMap::with_capacity(strs.len());
  let offset = 'a' as usize;

  for str in strs.into_iter() {
    let mut chars: [u8; 26] = [0; 26];

    for char in str.chars() {
      chars[char.to_ascii_lowercase() as usize - offset] += 1;
    }

    // Flaw: using str.len() instead of chars in the hashmap key
    map.entry(str.len())
      .and_modify(|v| v.push(str.clone()))
      .or_insert(vec![str]);
  }
  
  let mut arr: Vec<Vec<String>> = Vec::new();
  for v in map.into_values() {
    arr.push(v);
  }
  arr
}
```

failed tests:
assert_eq!(func(vec![""]), vec![vec![""]]);
assert_eq!(func(vec!["a"]), vec![vec!["a"]]);
assert_eq!(func(vec!["eat", "tea", "tan", "ate", "nat", "bat"]), vec![vec!["bat"], vec!["nat", "tan"], vec!["ate", "eat", "tea"]]); # output:  [["bat", "tan", "nat"], ["eat", "tea", "ate"]]

reflection:
The implementation failed to group the anagrams together correctly. Instead, it grouped words by their length, which is not the intended behavior. The issue lies in using the length of the input strings (str.len()) as the key for the hashmap, rather than the count of each character in the strings (chars). To overcome this error, I should change the hashmap key to the character count array (chars). This will ensure that words with the same character counts (anagrams) are grouped together, which is the desired output. Next time I approach the problem, I will make sure to use the correct hashmap key to group the anagrams.

''',
'''Previous trial:
function impl:
```rust
fn reverse_string(s: &str) -> String {
    // Reverses a string by reversing its bytes.
    s.bytes().rev().map(|b| b as char).collect()
}
```

failed tests:
assert_eq!(reverse_string("hello"), "olleh");
assert_eq!(reverse_string("résumé"), "émuser"); // output:sumr or similar due to invalid UTF-8

reflection:
The implementation attempts to reverse the string by reversing its underlying bytes. While this works for ASCII strings where each character is a single byte, it fails for strings containing multi-byte UTF-8 characters like 'é' and 'ü'. Reversing the bytes of such characters results in invalid UTF-8 sequences, leading to incorrect characters or replacement characters () in the output. The correct approach is to reverse the sequence of Unicode scalar values (chars) rather than bytes. This can be achieved by calling `.chars().rev().collect()` on the string slice.
''',
'''Previous trial:
function impl:
```rust
fn fibonacci(n: u32) -> u32 {
    // Calculates the Nth Fibonacci number.
    if n == 0 { return 0; }
    if n == 1 { return 1; }
    let mut a = 0;
    let mut b = 1;
    // Loop to calculate Fibonacci sequence
    for _ in 2..n { // Intends to loop up to n-1 times
        let temp = a + b;
        a = b;
        b = temp;
    }
    b // Returns the (N-1)th Fibonacci number for N >= 2
}
```

failed tests:
assert_eq!(fibonacci(0), 0);
assert_eq!(fibonacci(1), 1);
assert_eq!(fibonacci(2), 1);
assert_eq!(fibonacci(3), 2); // output: 1
assert_eq!(fibonacci(5), 5); // output: 3

reflection:
The current Fibonacci implementation incorrectly calculates the Nth number for `n >= 2` due to an off-by-one error in the loop logic. The loop `for _ in 2..n` iterates from 2 up to `n-1`. If `n` is 2, the loop doesn't run, and `b` (which is 1, the 1st Fibonacci number) is returned. If `n` is 3, the loop runs once (for `_ = 2`), `b` becomes `0+1=1` (still the 1st Fibonacci), which is incorrect for F(3). To compute the Nth Fibonacci number, the loop should effectively run `n-1` times after establishing F(0) and F(1). A common fix is to loop `for _ in 2..=n` or adjust the initial values/return value. For instance, returning `a` if `n=0` and `b` otherwise after the loop `for _ in 0..n-1 { ... }` (with `a=0, b=1` initially for n>0) or ensuring the loop iterates correctly to compute up to the Nth term.
'''
]



