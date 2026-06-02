# This file is generated automatically through parse_prompt.py
SCORE_LOGIT = """Here are two images: the original and the edited version. Please evaluate the edited image based on the following editing instruction and requirement. 
Instruction: {prompt}
You need to rate the editing result from 0 to 5 based on the accuracy and quality of the edit. 
0: The wrong object was edited, or the edit completely fails to meet the requirements. 
5: The correct object was edited, the requirements were met, and the visual result is high quality.
Response Format (Directly response the score number): 
0-5"""

_context_no_delimit_reasoning_first = """You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

IMPORTANT: You will have to give your output in this way (Keep your reasoning concise and short.):
{
"reasoning" : "...",
"score" : [...]
}
"""

_prompts_0shot_two_image_edit_rule = """RULES:

Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.
The objective is to evaluate how successfully the editing instruction has been executed in the second image.

Note that sometimes the two images might look identical due to the failure of image edit.
"""

_prompts_0shot_tie_rule_SC = """
From scale 1 to 10: 
A score from 1 to 10 will be given based on the success of the editing. (1 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)
A second score from 1 to 10 will rate the degree of overediting in the second image. (1 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.

Editing instruction: <instruction>
"""

_prompts_0shot_rule_PQ = """RULES:

The image is an AI-generated image.
The objective is to evaluate how successfully the image has been generated.

From scale 1 to 10: 
A score from 1 to 10 will be given based on image naturalness. 
(
    1 indicates that the scene in the image does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting. 
    10 indicates that the image looks natural.
)
A second score from 1 to 10 will rate the image artifacts. 
(
    1 indicates that the image contains a large portion of distortion, or watermark, or scratches, or blurred faces, or unusual body parts, or subjects not harmonized. 
    10 indicates the image has no artifacts.
)
Put the score in a list such that output score = [naturalness, artifacts]
"""