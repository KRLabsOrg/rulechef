"""Observe GLiNER / GLiNER2 predictions and learn rules from them.

Demonstrates:
  - GLiNER NER observation with monkey-patching
  - GLiNER2 classification and structured extraction observation
  - AgenticCoordinator with rule pruning
  - Incremental learning with corrections
  - grex-powered regex suggestions

Requirements:
    pip install rulechef[gliner]   # for GLiNER
    pip install rulechef[gliner2]  # for GLiNER2

Usage:
    export OPENAI_API_KEY='your-key'
    python examples/gliner_observation.py
"""

import os

from openai import OpenAI

from rulechef import RuleChef
from rulechef.coordinator import AgenticCoordinator


def _make_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")

    client_kwargs = {"api_key": api_key}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


def gliner_ner():
    """Observe GLiNER NER predictions → learn rules → evaluate."""
    from gliner import GLiNER

    print("=" * 60)
    print("GLiNER NER Observation")
    print("=" * 60)

    client = _make_client()
    model_name = os.environ.get("RULECHEF_MODEL", "gpt-4o-mini")

    # Load GLiNER model
    gliner_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

    # Set up RuleChef with AgenticCoordinator + grex
    coordinator = AgenticCoordinator(
        llm_client=client,
        model=model_name,
        prune_after_learn=True,
    )
    chef = RuleChef(
        client=client,
        model=model_name,
        coordinator=coordinator,
        use_grex=True,
    )

    # Observe GLiNER predictions (auto-detects predict_entities → NER)
    chef.start_observing_gliner(gliner_model, auto_learn=False)

    labels = ["person", "company", "location"]
    texts = [
        "Apple was founded by Steve Jobs and Steve Wozniak in Cupertino.",
        "Elon Musk is the CEO of Tesla, headquartered in Austin, Texas.",
        "Microsoft, led by Satya Nadella, is based in Redmond, Washington.",
        "Jeff Bezos founded Amazon in Seattle in 1994.",
        "Sundar Pichai runs Google from Mountain View, California.",
        "Mark Zuckerberg created Facebook in Cambridge.",
        "Tim Cook took over Apple after Steve Jobs passed away in Palo Alto.",
        "Nvidia, led by Jensen Huang, is headquartered in Santa Clara.",
        "Larry Page and Sergey Brin started Google in Menlo Park.",
        "Satya Nadella transformed Microsoft from its headquarters in Redmond.",
    ]

    print("\nObserving 10 GLiNER predictions...")
    for text in texts:
        entities = gliner_model.predict_entities(text, labels, threshold=0.3)
        ents = ", ".join(f"{e['label']}:{e['text']}" for e in entities)
        print(f"  {text[:55]:58s} → {ents}")

    print(f"\nBuffer: {chef.get_buffer_stats()['new_examples']} examples")

    # Learn rules (full synthesis + refinement)
    print("\n--- Phase 1: Initial learning ---")
    result = chef.learn_rules(run_evaluation=True, max_refinement_iterations=3)
    if result:
        rules, eval_result = result
        print(f"\n  {len(rules)} rules, F1={eval_result.micro_f1:.0%}")

    # Add more data incrementally
    print("\n--- Phase 2: Incremental learning with 5 more examples ---")
    more_texts = [
        "Sam Altman leads OpenAI from San Francisco.",
        "Dario Amodei runs Anthropic from San Francisco, California.",
        "Intel, founded by Gordon Moore, is based in Santa Clara.",
        "Reed Hastings co-founded Netflix in Scotts Valley.",
        "Lisa Su is the CEO of AMD, based in Santa Clara.",
    ]

    for text in more_texts:
        gliner_model.predict_entities(text, labels, threshold=0.3)

    result = chef.learn_rules(
        run_evaluation=True,
        max_refinement_iterations=2,
        incremental_only=True,
    )
    if result:
        rules, eval_result = result
        print(f"\n  {len(rules)} rules after patch, F1={eval_result.micro_f1:.0%}")

    # Test on unseen data
    print("\n--- Held-out test ---")
    chef.stop_observing_gliner()

    test_texts = [
        "Pat Gelsinger was the CEO of Intel in Santa Clara.",
        "Andy Jassy runs Amazon from Seattle, Washington.",
    ]
    for text in test_texts:
        gliner_ents = gliner_model.predict_entities(text, labels, threshold=0.3)
        rule_result = chef.extract({"text": text})

        gliner_set = {(e["text"], e["label"]) for e in gliner_ents}
        rule_set = {(e["text"], e["type"]) for e in rule_result.get("entities", [])}
        overlap = len(gliner_set & rule_set)

        print(f"\n  {text}")
        print(f"    GLiNER: {sorted(gliner_set)}")
        print(f"    Rules:  {sorted(rule_set)}  ({overlap}/{len(gliner_set)} match)")


def gliner2_classification():
    """Observe GLiNER2 classification → learn rules."""
    from gliner2 import GLiNER2

    print("\n" + "=" * 60)
    print("GLiNER2 Classification Observation")
    print("=" * 60)

    client = _make_client()
    model_name = os.environ.get("RULECHEF_MODEL", "gpt-4o-mini")

    extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

    coordinator = AgenticCoordinator(
        llm_client=client,
        model=model_name,
        prune_after_learn=True,
    )
    chef = RuleChef(
        client=client,
        model=model_name,
        coordinator=coordinator,
        use_grex=True,
    )

    # Observe classify_text → CLASSIFICATION task
    chef.start_observing_gliner(extractor, method="classify_text", auto_learn=False)

    schema = {"sentiment": ["positive", "negative", "neutral"]}
    texts = [
        "I love this product! It's amazing and works perfectly.",
        "Terrible experience, the product broke after one day.",
        "The weather is okay today, nothing special.",
        "Best purchase I've ever made, highly recommend!",
        "Complete waste of money, worst quality ever.",
        "Average product, works fine for the price.",
        "Absolutely fantastic, exceeded all my expectations!",
        "Horrible customer service, will never buy again.",
        "It's a decent product, nothing extraordinary.",
        "Really happy with my purchase, fast delivery too!",
    ]

    print("\nObserving 10 classifications...")
    for text in texts:
        result = extractor.classify_text(text, schema)
        label = list(result.values())[0]
        print(f"  {text[:55]:58s} → {label}")

    result = chef.learn_rules(run_evaluation=True, max_refinement_iterations=3)
    if result:
        rules, eval_result = result
        print(f"\n  {len(rules)} rules, F1={eval_result.micro_f1:.0%}")

    # Test
    print("\n--- Held-out test ---")
    chef.stop_observing_gliner()

    for text in ["Great quality, very happy!", "Broke on day one, terrible."]:
        gliner_label = list(extractor.classify_text(text, schema).values())[0]
        rule_label = chef.extract({"text": text}).get("label", "")
        match = "✓" if gliner_label == rule_label else "✗"
        print(f"  {match} {text:45s} GLiNER2={gliner_label:10s} Rules={rule_label}")


def gliner2_extraction():
    """Observe GLiNER2 structured extraction → learn rules."""
    from gliner2 import GLiNER2

    print("\n" + "=" * 60)
    print("GLiNER2 Structured Extraction Observation")
    print("=" * 60)

    client = _make_client()
    model_name = os.environ.get("RULECHEF_MODEL", "gpt-4o-mini")

    extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

    chef = RuleChef(client=client, model=model_name, use_grex=True)

    # Observe extract_json → TRANSFORMATION task
    chef.start_observing_gliner(extractor, method="extract_json", auto_learn=False)

    schema = {
        "people": [
            "name::str::Person name",
            "age::str::Age",
            "role::str::Job role",
            "company::str::Company",
        ]
    }

    texts = [
        "John Smith, age 35, works as a software engineer at Google.",
        "Maria Garcia, 28, is a data scientist at Microsoft.",
        "Bob Johnson, age 42, is a product manager at Amazon.",
        "Alice Chen, 31, works as a UX designer at Apple.",
        "David Kim, age 45, is a VP of engineering at Meta.",
        "Sarah Wilson, 29, is a machine learning engineer at Nvidia.",
    ]

    print("\nObserving 6 extractions...")
    for text in texts:
        result = extractor.extract_json(text, schema)
        for p in result.get("people", []):
            print(f"  {text[:50]:53s} → {p}")

    result = chef.learn_rules(run_evaluation=True)
    if result:
        rules, eval_result = result
        print(f"\n  {len(rules)} rules, F1={eval_result.micro_f1:.0%}")

    # Test on unseen data
    print("\n--- Held-out test ---")
    chef.stop_observing_gliner()

    for text in [
        "Emily Brown, age 38, is a senior architect at Oracle.",
        "James Lee, 33, works as a DevOps engineer at Spotify.",
    ]:
        gliner_result = extractor.extract_json(text, schema)
        rule_result = chef.extract({"text": text})
        print(f"\n  {text}")
        print(f"    GLiNER2: {gliner_result.get('people', [])}")
        print(f"    Rules:   {rule_result.get('people', [])}")


if __name__ == "__main__":
    import sys

    # Run specific demo or all
    demos = {
        "ner": gliner_ner,
        "classify": gliner2_classification,
        "extract": gliner2_extraction,
    }

    if len(sys.argv) > 1 and sys.argv[1] in demos:
        demos[sys.argv[1]]()
    else:
        gliner_ner()
        gliner2_classification()
        gliner2_extraction()
