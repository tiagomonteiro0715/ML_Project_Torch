# -*- coding: utf-8 -*-
"""
Phonebook Lookup (Repeat After Me)

Problem: Proved transformers can copy exponentially long strings while
fixed-state SSMs fundamentally cannot. Phonebook lookup tests retrieval
from context.
"""

import random
import string


def generate_phonebook_task(
    n_entries=100,
    name_len=5,
    num_queries=10,
    seed=42
):
    """
    Generates a phonebook lookup task:
    - A context string listing name->number pairs
    - Query/answer pairs asking for a specific entry

    The paper proves Transformers can copy exponentially long strings
    with 2 layers, while fixed-memory models (SSMs) fundamentally cannot.
    """
    random.seed(seed)

    # Generate unique names and phone numbers
    names = set()
    while len(names) < n_entries:
        names.add("".join(random.choices(string.ascii_uppercase, k=name_len)))
    names = list(names)
    numbers = ["".join(random.choices(string.digits, k=10)) for _ in range(n_entries)]

    phonebook = dict(zip(names, numbers))

    # Build the context string (what the model sees)
    context_lines = [f"{name}: {number}" for name, number in phonebook.items()]
    context = "\n".join(context_lines)

    # Build query/answer pairs
    query_names = random.sample(names, min(num_queries, n_entries))
    queries = []
    for name in query_names:
        queries.append({
            "context": context,
            "query": f"What is the phone number for {name}?",
            "answer": phonebook[name]
        })

    return phonebook, queries


if __name__ == "__main__":
    phonebook, queries = generate_phonebook_task(n_entries=100, num_queries=5)

    print(f"Phonebook size: {len(phonebook)} entries")
    print(f"Context length:  ~{len(queries[0]['context'])} chars")
    print()
    for i, q in enumerate(queries):
        print(f"Q{i+1}: {q['query']}")
        print(f"A{i+1}: {q['answer']}\n")
