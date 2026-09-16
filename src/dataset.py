from typing import Dict, List

NAMES_SG = ["The cat", "The dog", "The teacher", "The scientist", "The artist",
            "The manager", "The doctor", "The student", "The engineer", "The nurse",
            "The pilot", "The chef", "The farmer", "The lawyer", "The painter"]
NAMES_PL = [n + "s" for n in NAMES_SG]

VERBS_SG = ["runs", "sleeps", "writes", "works", "sings", "arrives",
            "walks", "talks", "reads", "dances"]
VERBS_PL = ["run", "sleep", "write", "work", "sing", "arrive",
            "walk", "talk", "read", "dance"]

HEAD_NOUNS_SG = ["key", "letter", "result", "photo", "report", "message", "answer",
                 "cause", "effect", "solution", "idea", "plan", "request", "comment",
                 "note", "file", "document", "article", "story", "review"]
HEAD_NOUNS_PL = [n + "s" for n in HEAD_NOUNS_SG]

DISTRACTOR_NOUNS_SG = ["cabinet", "lawyer", "experiment", "wall", "manager", "client",
                       "question", "problem", "decision", "issue", "teacher", "student",
                       "committee", "board", "judge", "doctor", "engineer", "artist",
                       "scientist", "writer"]
DISTRACTOR_NOUNS_PL = [n + "s" for n in DISTRACTOR_NOUNS_SG]

PREPS = ["to the", "from the", "of the", "on the", "near the"]

NEG_SUBJECTS = ["The cat", "The teacher", "The manager", "The doctor", "The artist"]
NEG_VERBS = [("like", "likes"), ("agree", "agrees"), ("care", "cares"), ("worry", "worries")]

PRONOUNS_SG = ["He", "She", "It"]
PRONOUNS_PL = ["They", "We"]
PRONOUN_VERBS = [("walks", "walk"), ("runs", "run"), ("sings", "sing"), ("works", "work")]


def _item(prompt: str, correct: str, incorrect: str, setting: str) -> Dict[str, str]:
    return {"prompt": prompt, "correct": " " + correct, "incorrect": " " + incorrect,
            "setting": setting}


def build_dataset() -> List[Dict[str, str]]:
    data = []

    for i, subj in enumerate(NAMES_SG):
        data.append(_item(subj, VERBS_SG[i % 10], VERBS_PL[i % 10], "simple_sg"))
    for i, subj in enumerate(NAMES_PL):
        data.append(_item(subj, VERBS_PL[i % 10], VERBS_SG[i % 10], "simple_pl"))

    for i in range(len(HEAD_NOUNS_SG)):
        prep = PREPS[i % len(PREPS)]
        data.append(_item(f"The {HEAD_NOUNS_SG[i]} {prep} {DISTRACTOR_NOUNS_PL[i]}",
                          "is", "are", "distractor_sg"))
        data.append(_item(f"The {HEAD_NOUNS_PL[i]} {prep} {DISTRACTOR_NOUNS_SG[i]}",
                          "are", "is", "distractor_pl"))

    for subj in NEG_SUBJECTS:
        for base, inflected in NEG_VERBS:
            data.append(_item(f"{subj} does not", base, inflected, "negation"))
            data.append(_item(f"{subj}s do not", base, inflected, "negation"))

    for pron in PRONOUNS_SG:
        for sg, pl in PRONOUN_VERBS:
            data.append(_item(pron, sg, pl, "pronoun"))
    for pron in PRONOUNS_PL:
        for sg, pl in PRONOUN_VERBS:
            data.append(_item(pron, pl, sg, "pronoun"))

    return data


def build_patching_pairs(dataset: List[Dict[str, str]], seed: int = 0) -> List[Dict[str, str]]:
    import random
    rng = random.Random(seed)
    sg = [d for d in dataset if d["setting"] == "simple_sg"]
    pl = [d for d in dataset if d["setting"] == "simple_pl"]

    pairs = []
    for clean, pool in [(s, pl) for s in sg] + [(p, sg) for p in pl]:
        corrupted = rng.choice(pool)
        pairs.append({"clean": clean["prompt"], "corrupted": corrupted["prompt"],
                      "correct": clean["correct"], "incorrect": clean["incorrect"]})
    return pairs


if __name__ == "__main__":
    from collections import Counter
    ds = build_dataset()
    print(len(ds), Counter(d["setting"] for d in ds))
