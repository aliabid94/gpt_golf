import gradio as gr
import json
import random
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2", max_length=60)

with open("wordlist.json") as wordlist_json:
    wordlist = json.load(wordlist_json)


def autocomplete(text):
    end_text = " ".join(text.split(" ")[-30:-1])
    generated_text = generator(
        end_text, return_full_text=False, clean_up_tokenization_spaces=True
    )[0]["generated_text"]
    generated_text = generated_text.replace("\n", "")
    return generated_text


with gr.Blocks() as demo:
    gr.Markdown(
        """
    # GPT Golf

    How many turns will it take you to get GPT to say the target word?
    Here are the rules of the game: 
    - Your goal is to get GPT to say a target word in as few turns as possible.
    - Each turn, you add up to 5 words to its dialogue. 
    - When you click submit, your prompt will be added to the dialogue. Then GPT will also add to the dialogue.
    - You can't say the target word, but as soon as GPT does, you win!
    """
    )
    error_box = gr.Textbox(label="Error", elem_id="error", visible=False)
    dialogue_var = gr.Variable(value=[])

    start_btn = gr.Button("Start", variant="primary")
    with gr.Column(visible=False) as game:
        with gr.Row() as stats:
            target_word_box = gr.Textbox(
                label="Target Word", elem_id="target", interactive=False
            )
            num_turns_box = gr.Number(0, label="# of Turns so Far", elem_id="num_turns")
        dialogue_box = gr.HighlightedText(label="Dialogue")
        with gr.Column() as prompt_set:
            prompt_box = gr.Textbox(label="Prompt", placeholder="Enter Next 5 Words...")
            submit_btn = gr.Button("Submit").style(full_width=True)
        win = gr.HTML(
            "<div style='width: 100%; padding: 3rem; font-size: 4rem; color: green; text-align: center; font-weight: bold'>You Won!</div>",
            visible=False,
        )

    def start_game():
        return {
            start_btn: gr.update(visible=False),
            game: gr.update(visible=True),
            target_word_box: random.choice(wordlist),
        }

    start_btn.click(start_game, inputs=None, outputs=[start_btn, game, target_word_box])

    def submit(prompt, target_word, dialogue, num_turns):
        if len(prompt.split(" ")) > 5:
            return {
                error_box: gr.update(
                    visible=True, value="Prompt must be a maximum of 5 words!"
                )
            }
        if target_word in prompt:
            return {
                error_box: gr.update(
                    visible=True, value="You can't use the target word in the prompt!"
                )
            }
        dialogue.append(prompt)
        response = autocomplete(" ".join(dialogue))
        dialogue.append(response)
        labeled_dialogue = [
            (text, None if i % 2 == 0 else "gpt") for i, text in enumerate(dialogue)
        ]
        if target_word in response:
            return {
                dialogue_box: labeled_dialogue,
                prompt_set: gr.update(visible=False),
                win: gr.update(visible=True),
                num_turns_box: num_turns + 1,
                dialogue_var: dialogue,
                error_box: gr.update(visible=False),
            }
        else:
            return {
                dialogue_box: labeled_dialogue,
                prompt_box: "",
                num_turns_box: num_turns + 1,
                dialogue_var: dialogue,
                error_box: gr.update(visible=False),
            }

    submit_btn.click(
        submit,
        inputs=[prompt_box, target_word_box, dialogue_var, num_turns_box],
        outputs=[
            dialogue_var,
            dialogue_box,
            prompt_box,
            num_turns_box,
            error_box,
            prompt_set,
            win,
        ],
    )


demo.launch()
