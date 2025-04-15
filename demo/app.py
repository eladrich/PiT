import gradio as gr
import spaces
from pit import PiTDemoPipeline

BLOCK_WIDTH = 300
BLOCK_HEIGHT = 360
FONT_SIZE = 3.5

pit_pipeline = PiTDemoPipeline(
    prior_repo="kfirgold99/Piece-it-Together", prior_path="models/characters_ckpt/prior.ckpt"
)


@spaces.GPU
def run_character_generation(part_1, part_2, part_3, seed=None):
    crops_paths = [part_1, part_2, part_3]
    image = pit_pipeline.run(crops_paths=crops_paths, seed=seed, n_images=1)[0]
    return image


with gr.Blocks(css="style.css") as demo:
    gr.HTML(
        """<div style="text-align: center;"><h1>Piece it Together: Part-Based Concepting with IP-Priors</h1></div>"""
    )
    gr.HTML(
        '<div style="text-align: center;"><h3><a href="https://eladrich.github.io/PiT/">https://eladrich.github.io/PiT/</a></h3></div>'
    )
    gr.HTML(
        '<div style="text-align: center;">Piece it Together (PiT) combines different input parts to generate a complete concept in a prior domain.</div>'
    )
    with gr.Row(equal_height=True, elem_classes="justified-element"):
        with gr.Column(scale=0, min_width=BLOCK_WIDTH):
            part_1 = gr.Image(label="Upload part 1", type="filepath", width=BLOCK_WIDTH, height=BLOCK_HEIGHT)
        with gr.Column(scale=0, min_width=BLOCK_WIDTH):
            part_2 = gr.Image(label="Upload part 2", type="filepath", width=BLOCK_WIDTH, height=BLOCK_HEIGHT)
        with gr.Column(scale=0, min_width=BLOCK_WIDTH):
            part_3 = gr.Image(label="Upload part 3", type="filepath", width=BLOCK_WIDTH, height=BLOCK_HEIGHT)
        with gr.Column(scale=0, min_width=BLOCK_WIDTH):
            output_eq_1 = gr.Image(label="Output", width=BLOCK_WIDTH, height=BLOCK_HEIGHT)
    with gr.Row(equal_height=True, elem_classes="justified-element"):
        run_button = gr.Button("Create your character!", elem_classes="small-elem")
        run_button.click(fn=run_character_generation, inputs=[part_1, part_2, part_3], outputs=[output_eq_1])
    with gr.Row(equal_height=True, elem_classes="justified-element"):
        pass

    with gr.Row(equal_height=True, elem_classes="justified-element"):
        with gr.Column(scale=1):
            examples = [
                [
                    "assets/characters_parts/part_a.jpg",
                    "assets/characters_parts/part_b.jpg",
                    "assets/characters_parts/part_c.jpg",
                ]
            ]
            gr.Examples(
                examples=examples,
                inputs=[part_1, part_2, part_3],
                outputs=[output_eq_1],
                fn=run_character_generation,
                cache_examples=False,
            )

demo.queue().launch(share=True)
