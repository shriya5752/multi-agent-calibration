"""
Agent: wraps a single LLM call for one round of the debate.

Each agent must return a structured triple:
    {"answer": "A", "confidence": 0-100, "reasoning": "..."}

We ask the model to emit a strict tag format and parse it with regex,
failing explicitly on malformed output so invalid predictions cannot enter metrics.
"""

import re
import random


ANSWER_RE = re.compile(r"^ANSWER:[ \t]*([A-D])[ \t]*$", re.IGNORECASE | re.MULTILINE)
CONFIDENCE_RE = re.compile(r"^CONFIDENCE:[ \t]*(\d{1,3})[ \t]*%?[ \t]*$", re.IGNORECASE | re.MULTILINE)


def build_prompt(question, options, other_responses, round_num):
    """Construct the user-turn prompt for a given round."""
    opt_lines = "\n".join(f"{k}) {v}" for k, v in options.items())

    base = (
        f"Question:\n{question}\n\nOptions:\n{opt_lines}\n\n"
    )

    if round_num == 1 or not other_responses:
        base += (
            "Give your best answer.\n"
            "Respond in EXACTLY this format:\n"
            "REASONING: <2-3 sentences>\n"
            "ANSWER: <A/B/C/D>\n"
            "CONFIDENCE: <0-100, your certainty that this answer is correct>\n"
        )
    else:
        other_block = "\n".join(
            f"- {name}: answered {r['answer']} (confidence {r['confidence']}). "
            f"Reasoning: {r['reasoning']}"
            for name, r in other_responses.items()
        )
        base += (
            f"Other agents' round {round_num - 1} responses:\n{other_block}\n\n"
            "Reconsider the question given what the other agents said. "
            "You may change your answer or keep it â€” decide based on the "
            "arguments, not on peer pressure alone.\n"
            "Respond in EXACTLY this format:\n"
            "REASONING: <2-3 sentences, mention if/why you changed your mind>\n"
            "ANSWER: <A/B/C/D>\n"
            "CONFIDENCE: <0-100, your certainty that this answer is correct>\n"
        )
    return base


def parse_response(text):
    """Extract answer/confidence/reasoning from raw model text."""
    answer_match = ANSWER_RE.search(text)
    conf_match = CONFIDENCE_RE.search(text)

    answer = answer_match.group(1).upper() if answer_match else None
    confidence = int(conf_match.group(1)) if conf_match else None
    if answer is None or confidence is None or not 0 <= confidence <= 100:
        raise ValueError("Malformed model response: expected ANSWER A-D and CONFIDENCE 0-100. " + repr(text))

    reasoning = text.strip()
    if "REASONING:" in text.upper():
        # crude but robust slice between REASONING: and ANSWER:
        try:
            start = text.upper().index("REASONING:") + len("REASONING:")
            end = text.upper().index("ANSWER:")
            reasoning = text[start:end].strip()
        except ValueError:
            pass

    parsed_ok = True

    return {
        "answer": answer,
        "confidence": confidence,
        "reasoning": reasoning,
        "parsed_ok": parsed_ok,
        "raw_text": text,
    }


class Agent:
    def __init__(self, name, persona_key, persona_prompt, client=None,
                 model="llama-3.1-8b-instant", mock=False):
        self.name = name
        self.persona_key = persona_key
        self.persona_prompt = persona_prompt
        self.client = client
        self.model = model
        self.mock = mock

    def respond(self, question, options, other_responses, round_num):
        prompt = build_prompt(question, options, other_responses, round_num)

        if self.mock:
            text = self._mock_response(options, round_num)
        else:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.persona_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=300,
            )
            text = completion.choices[0].message.content

        return parse_response(text)

    def _mock_response(self, options, round_num):
        """Deterministic-ish fake response so the pipeline is testable
        without any API key or network access."""
        letter = random.choice(list(options.keys()))
        conf = random.randint(40, 95)
        return (
            f"REASONING: Mock reasoning for round {round_num}.\n"
            f"ANSWER: {letter}\n"
            f"CONFIDENCE: {conf}\n"
        )

