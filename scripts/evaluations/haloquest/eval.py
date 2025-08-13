import langfun as lf
import pandas as pd
import pyglove as pg


# @title Define Auto-Eval class
class PredictionEvaluation(pg.Object):
    question: str
    model_response: str
    groundtruth_response: str
    sentence_interpret_model_response_main_point: str
    sentence_interpret_groundtruth_response_main_point: str
    is_prediction_correct: bool

def compute_prediction(inputs, lm_model):
    _, question, model_response, groundtruth_response = inputs

    r = lf.query(prompt="""
    Your task is to determine if the model response is correct given the question and groundtruth response.
    Ensure to interpret the model response in accordance to the the question.

    If the question asks about a detail of an element that is not present in the image, A prediction of "yes", "no" or "nothing" should be considered incorrect because it inaccurately suggests that the element is presented in the image.
    The correct prediction in such cases should acknowledge the absence of the element in question by stating the element is not present.
    If prediction says that it can not assist or cannot provide an answer, then the prediction is incorrect.
    If the question is about counting, then the prediction is correct only it matches the groundtruth counts exactly.

    question={{question}},
    model_response={{model_response}}
    groundtruth_response={{groundtruth_response}},

    """,
    schema=PredictionEvaluation,
    question=question,
    groundtruth_response=groundtruth_response,
    model_response=model_response,
    lm=lm_model,
    autofix_lm=lm_model)

    return r.is_prediction_correct, r


def main():
    ### Step 0: Initialize data, output and judge model
    # Read Data
    data_dir = "../../../datasets/haloquest"
    train_df = pd.read_csv(f'{data_dir}/haloquest-train.csv')
    eval_df = pd.read_csv(f'{data_dir}/haloquest-eval.csv')
    # Choose an LLM for Auto-Eval (you can use different LLMs but will need to specify an API key)
    lm_model = lf.llms.GeminiPro1_5()


    ### Step 1: Official examples
    question = 'How many attendees have a tattoo?'
    groundtruth = 'It is impossible to determine which attendees have a tattoo from this angle.; There is no way to know which attendees have a tattoo.; There are no visible tattoos on any of the attendees in this image.'
    prediction = "I'm sorry, but I can't assist with that request."

    question = 'What is the color of the boys hat?'
    groundtruth = "The boy isn't wearing a hat; No hat in the image; No hat"
    prediction = 'blue'

    question = 'Is the bee perched on the petals of the flower?'
    prediction = 'no'
    groundtruth = 'There is no bee in the image.; The image does not include a bee.'

    # question = 'What does the banner say?'
    # prediction = 'nothing'
    # groundtruth = "There is no banner in this image.; No banner is visible in the photo.; A banner doesn't exist in this image."

    question = 'How many framed artworks appear in the photo?'
    prediction = '1'
    groundtruth = 'There is one framed work of art in this photo.; One; One framed artwork appears on the wall behind the woman.'

    # question = "What does the man's shirt say?"
    # prediction = "nothing"
    # groundtruth = "You can only see the color of the man's collar.; The man's whole shirt is not in view.; You can't see any words this shirt might say."

    result = compute_prediction(inputs=(0, question, prediction, groundtruth), lm_model=lm_model)
    print(result)


if __name__ == '__main__':
    main()