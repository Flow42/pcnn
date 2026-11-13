"""Gradio presentation layer. Run with `uv run python -m visualization.app`."""

from pathlib import Path
import argparse

import gradio as gr

from .algorithms import ALGORITHMS
from .processing import cleanup_result, inspect_frame, process_image


def material_theme():
    return gr.themes.Base(
        primary_hue="violet", secondary_hue="teal", neutral_hue="slate",
        font=["Roboto", "Arial", "sans-serif"],
    ).set(
        body_background_fill="#f7f2fa", body_background_fill_dark="#141218",
        block_background_fill="#fffbfe", block_background_fill_dark="#211f26",
        body_text_color="#1d1b20", body_text_color_dark="#e6e0e9",
        body_text_color_subdued="#49454f", body_text_color_subdued_dark="#cac4d0",
        block_label_text_color="#49454f", block_label_text_color_dark="#cac4d0",
        block_info_text_color="#49454f", block_info_text_color_dark="#cac4d0",
        block_border_width="0px", block_radius="24px",
        block_shadow="0 1px 3px rgba(29, 27, 32, 0.12)",
        button_primary_background_fill="#6750a4", button_primary_text_color="#ffffff",
        button_primary_background_fill_hover="#7962b6",
        button_large_radius="999px", button_small_radius="999px",
        input_radius="12px", section_header_text_size="*text_lg",
    )


def build_app(algorithms=None):
    registry = ALGORITHMS if algorithms is None else algorithms
    if not registry:
        raise ValueError("Register at least one algorithm.")
    default = next(iter(registry))
    with gr.Blocks(title="Pulse Studio", analytics_enabled=False,
                   delete_cache=(600, 3600)) as demo:
        result = gr.State(None, time_to_live=3600, delete_callback=cleanup_result)
        gr.HTML('<header class="studio-header"><span class="eyebrow">PCNN EXPLORER</span>'
                '<h1>Pulse Studio</h1><p>Turn an image into a sequence of neural pulses. '
                'Tune, play, and explore each iteration.</p></header>')
        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=300, elem_classes="surface", elem_id="upload-card"):
                source = gr.Image(type="filepath", image_mode="RGB", sources=["upload"],
                                  label="Upload an image", height=360, interactive=True)
            with gr.Column(scale=7, min_width=320, elem_classes="surface", elem_id="settings-card"):
                algorithm = gr.Dropdown([(adapter.name, key) for key, adapter in registry.items()],
                                        value=default, label="Algorithm", interactive=True)
                controls_slot = gr.Column()
        with gr.Column(elem_classes="surface", elem_id="results-card"):
            gr.Markdown("## Explore the result")
            status = gr.Markdown("Upload an image, adjust the parameters, then select **Run algorithm**.",
                                 elem_id="run-status")
            with gr.Tabs():
                with gr.Tab("Video"):
                    video = gr.Video(label="Pulse sequence", interactive=False, height=400,
                                     format="mp4")
                with gr.Tab("Frame inspector"):
                    frame_image = gr.Image(label="Current frame", interactive=False,
                                           height=400, elem_classes="pixel-preview")
                    frame_slider = gr.Slider(1, 2, value=1, step=1, label="Frame", interactive=False)
                    with gr.Row():
                        previous = gr.Button("← Previous frame", interactive=False)
                        following = gr.Button("Next frame →", interactive=False)
                    frame_info = gr.Markdown("Run an image to inspect its frames.")
                with gr.Tab("Selected result"):
                    best = gr.Image(label="Highest-entropy frame", interactive=False, height=400,
                                    elem_classes="pixel-preview")
                    gr.Markdown("The earliest frame with maximum pixel entropy is selected. "
                                "For PCNN's binary pulses, this favors a balance of firing and silent neurons.")

        with controls_slot:
            @gr.render(inputs=[algorithm, source])
            def render_controls(key, image_path):
                adapter = registry[key]
                gr.Markdown(adapter.description, elem_classes="plain-copy")
                if not image_path:
                    gr.Markdown("Upload an image to configure the algorithm.", elem_classes="plain-copy")
                    return
                parameters = [parameter for parameter in adapter.parameters if parameter.visible]
                components = []
                with gr.Row(elem_classes="parameter-grid"):
                    for parameter in parameters:
                        with gr.Column(scale=1, min_width=240, key=f"{key}-{parameter.name}-cell"):
                            if isinstance(parameter.default, bool):
                                component = gr.Checkbox(value=parameter.default, label=parameter.label,
                                                        info=parameter.info, key=f"{key}-{parameter.name}",
                                                        interactive=True, scale=1, min_width=240)
                            else:
                                component = gr.Slider(parameter.minimum, parameter.maximum,
                                                      value=parameter.default, step=parameter.step,
                                                      label=parameter.label, info=parameter.info,
                                                      key=f"{key}-{parameter.name}", interactive=True,
                                                      scale=1, min_width=240)
                        components.append(component)
                run = gr.Button("Run algorithm", variant="primary", size="lg")

                def run_algorithm(path, old_result, *values, progress=gr.Progress()):
                    progress(0, desc="Computing pulse frames…")
                    params = dict(zip((p.name for p in parameters), values))
                    try:
                        current = process_image(
                            adapter, path, params,
                            progress=lambda count: progress(None, desc=f"Computed {count} frames…"),
                        )
                    except (ValueError, OSError, RuntimeError) as error:
                        raise gr.Error(str(error)) from error
                    cleanup_result(old_result)
                    image, _, info = inspect_frame(current, 1)
                    height, width = image.shape
                    return (current, current.video_path, image,
                            gr.Slider(minimum=1, maximum=max(2, len(current.frames)), value=1,
                                      interactive=len(current.frames) > 1),
                            info, current.frames[current.best_index],
                            f"**{adapter.name}** · {len(current.frames)} frames · {width} × {height} px · "
                            f"selected frame **{current.best_index + 1}**. "
                            "Run again to apply changed inputs or settings.",
                            gr.Button(interactive=len(current.frames) > 1),
                            gr.Button(interactive=len(current.frames) > 1))

                run.click(run_algorithm, [source, result, *components],
                          [result, video, frame_image, frame_slider, frame_info, best, status, previous, following],
                          concurrency_id="viewer", concurrency_limit=1)

        def navigate(current, index, offset=0):
            image, index, info = inspect_frame(current, index + offset)
            return image, index, info

        navigation_outputs = [frame_image, frame_slider, frame_info]
        frame_slider.input(navigate, [result, frame_slider], navigation_outputs,
                           concurrency_id="viewer")
        previous.click(lambda current, index: navigate(current, index, -1),
                       [result, frame_slider], navigation_outputs, concurrency_id="viewer")
        following.click(lambda current, index: navigate(current, index, 1),
                        [result, frame_slider], navigation_outputs, concurrency_id="viewer")
    return demo.queue(max_size=16)


def main():
    parser = argparse.ArgumentParser(description="Launch the Gradio PCNN visualization")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app().launch(server_name=args.host, server_port=args.port,
                       theme=material_theme(), css_paths=[Path(__file__).with_name("material.css")])


if __name__ == "__main__":
    main()
