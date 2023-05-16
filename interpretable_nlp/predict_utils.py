import json
from typing import List
import torch

from captum.attr import LayerIntegratedGradients

from colour import Color

SENTIMENT_TO_COLOR = {
    "радость": "#ff80cc",
    "печаль": "#00FFFF",
    "злость": "#3434b3",
    "неопред": "#72e2b3",
    "нейтр": "#909090",
}
COLOR_RANGE = list(Color("red").range_to(Color("white"), 10)) + list(
    Color("white").range_to(Color("green"), 10)
)


def predict_sentiment(tokenized: list, model: torch.nn.Module, device: torch.device):
    """
    Predict Sentiment score (between 0 and 1)
    :param text:
    :param model: Model in eval mode
    :param device: cuda or cpu torch.device
    :return:
    """
    tokens_idx = tokenized.input_ids[:512]
    x = torch.tensor([tokens_idx], dtype=torch.long)

    x = x.to(device)

    with torch.no_grad():
        y_hat = model(x).logits.cpu()
        _, predicted = torch.max(y_hat, 1)
        predicted_class = predicted.item()
    return {
        0: "радость",
        1: "печаль",
        2: "злость",
        3: "неопред",
        4: "нейтр"
            }[predicted_class]


def join_split_tokens(tokens: list,
                      attributions: list) -> tuple[list, list]:
    new_tokens = []
    new_attributions = []
    for index, token in enumerate(tokens):
        attribution = attributions[index]
        if token.startswith('##'):
            token = new_tokens[-1] + token[2:]
            attribution = new_attributions[-1] + attribution
            new_tokens[-1] = token
            new_attributions[-1] = attribution
        else:
            new_tokens.append(token)
            new_attributions.append(attribution)
    return new_tokens, new_attributions


def attribution_to_html(tokens: List, attributions: List, sentiment: str):
    tokens, attributions = join_split_tokens(tokens, attributions)
    color = {
        "радость": lambda alpha: f"rgba(255, 255, 128, .{alpha})",
        "печаль": lambda alpha: f"rgba(30,129,176, .{alpha})",
        "злость": lambda alpha: f"rgba(228,52,52, .{alpha})",
        "неопред": lambda alpha: f"rgba(200, 114, 226, .{alpha})",
        "нейтр": lambda alpha: f"rgba(149, 142, 146, .{alpha})",
    }
    html = ""
    for token, attribution in zip(tokens, attributions):
        max_attr = max(map(abs, attributions))
        alpha = int(abs(attribution / max_attr) * 10) - 1
        bg_color = color[sentiment](alpha)
        print(token, alpha, 'orig attr', round(attribution, 4), 'out of', max_attr)
        html += f""" <span style="background-color: {bg_color}">{token}</span>"""

    return html





def compose_annotation_file(text, tokens, attributions):
    tokens, attributions = join_split_tokens(tokens, attributions)
    tokens_with_positions = [f'{position}: {token}'
                             for position, token in zip(list(range(len(tokens))),
                                                        tokens)]
    processed_attributions = list(
        map(lambda value: round(value, 3),
            map(abs, attributions)))
    attributions_dict = dict(zip(tokens_with_positions, processed_attributions))
    annotation = {'текст': text} | attributions_dict
    return json.dumps(annotation, ensure_ascii=False)


def attribution_fun(tokenized: str, model, embedding, device: torch.device,
                    special_tokens: dict[str, str]):
    tokens_idx = tokenized.input_ids[:512]
    x = torch.tensor([tokens_idx], dtype=torch.long)
    ref = torch.tensor(
        [[special_tokens['CLS_IDX']] + [special_tokens['PAD_IDX']]
         * (len(tokens_idx) - 2) + [special_tokens['SEP_IDX']]], dtype=torch.long
    )

    x = x.to(device)
    ref = ref.to(device)

    base_class = 2

    def forward_callable(inputs):
        return model(inputs).logits

    lig = LayerIntegratedGradients(
        forward_callable,
        embedding(model),
    )

    attributions_ig, delta = lig.attribute(
        x, ref, n_steps=500, return_convergence_delta=True, target=base_class
    )
    attributions_ig = attributions_ig[0, 1:-1, :].sum(dim=-1).cpu()
    # attributions_ig = attributions_ig / attributions_ig.abs().max()
    # attributions_ig = attributions_ig / torch.norm(attributions_ig)
    print('transformed attribution', attributions_ig.shape)
    return tokenized, attributions_ig.tolist()
