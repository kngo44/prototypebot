import os
from dotenv import load_dotenv
from colorama import Fore, Back, Style
from openai import OpenAI

# load values from the .env file if it exists
load_dotenv()

# configure OpenAI
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INSTRUCTIONS = """You are an AI assistant who is an expert recruiter and interviewer in the United States who works with hyper-competitive companies to hire top talent. 
You have been assessing applicants' compatibility for various roles for 10 years, with an impressive track record of matching the right job seekers with the right jobs.
You can offer guidance on effectively answering various categories of interview questions, including behavioral, communication, personality, technical, and brain teaser questions.
If you cannot provide an answer to a question, please respond with, "I'm unable to help with that. My current role is to assist only with topics related to the job application and interviewing process."
Please aim to be as helpful and friendly as possible in your response. Do not use any external URLs in your answers. Do not refer to any blogs in your answers.
Format the response by providing the user with an interview question tailored to their desired category and job position.
Please wait until the user responds to the question before offering professional tips on how to effectively answer the question and excel in the interview."""

TEMPERATURE = 0.5
MAX_TOKENS = 500
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0.6
# limits how many questions we include in the prompt
MAX_CONTEXT_QUESTIONS = 10


def get_response(instructions, previous_questions_and_answers, new_question):
    """Get a response from ChatCompletion

    Args:
        instructions: The instructions for the chat bot - this determines how it will behave
        previous_questions_and_answers: Chat history
        new_question: The new question to ask the bot

    Returns:
        The response text
    """
    # build the messages
    messages = [
        { "role": "system", "content": instructions },
    ]
    # add the previous questions and answers
    for question, answer in previous_questions_and_answers[-MAX_CONTEXT_QUESTIONS:]:
        messages.append({ "role": "user", "content": question })
        messages.append({ "role": "assistant", "content": answer })
    # add the new question
    messages.append({ "role": "user", "content": new_question })

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=1,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )
    return completion.choices[0].message.content


def get_moderation(question):
    """
    Check the question is safe to ask the model

    Parameters:
        question (str): The question to check

    Returns a list of errors if the question is not safe, otherwise returns None
    """

    errors = {
        "hate": "Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.",
        "hate/threatening": "Hateful content that also includes violence or serious harm towards the targeted group.",
        "self-harm": "Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.",
        "sexual": "Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).",
        "sexual/minors": "Sexual content that includes an individual who is under 18 years old.",
        "violence": "Content that promotes or glorifies violence or celebrates the suffering or humiliation of others.",
        "violence/graphic": "Violent content that depicts death, violence, or serious physical injury in extreme graphic detail.",
    }
    response = client.moderations.create(input=question)
    categories = response.results[0].categories
    if response.results[0].flagged:
        # Get the categories that are flagged and generate a message
        result = [
            error
            for category, error in errors.items()
            if categories.get(category, False)
        ]
        return result
    return None

def main():
    os.system("cls" if os.name == "nt" else "clear")

    print(Fore.GREEN + Style.BRIGHT + "What job position are you applying to? Which category of interview questions would you like to practice?")

    # keep track of previous questions and answers
    previous_questions_and_answers = []
    while True:
        # ask the user for their question
        new_question = input(
            Fore.GREEN + Style.BRIGHT + "You: " + Style.RESET_ALL
        )
        # check the question is safe
        errors = get_moderation(new_question)
        if errors:
            print(
                Fore.RED
                + Style.BRIGHT
                + "Sorry, your question didn't pass the moderation check:"
            )
            for error in errors:
                print(error)
            print(Style.RESET_ALL)
            continue
        response = get_response(INSTRUCTIONS, previous_questions_and_answers, new_question)

        # add the new question and answer to the list of previous questions and answers
        previous_questions_and_answers.append((new_question, response))

        # print the response
        print(Fore.CYAN + Style.BRIGHT + "interviewbot: " + Style.NORMAL + response)


if __name__ == "__main__":
    main()