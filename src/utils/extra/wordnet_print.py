from nltk.corpus import wordnet as wn

def pretty_synset_info(synset_name):
    try:
        syn = wn.synset(synset_name)
    except:
        print(f"❌ Synset '{synset_name}' not found.")
        return

    print(f"\n🧠 Synset: {syn.name()}")
    print(f"📖 Definition: {syn.definition()}")

    # Examples
    examples = syn.examples()
    print("\n💬 Examples:")
    if examples:
        for ex in examples:
            print(f"  - {ex}")
    else:
        print("  (None)")

    # Lemmas
    lemmas = [lemma.name() for lemma in syn.lemmas()]
    print("\n🔤 Lemmas:")
    for lemma in lemmas:
        print(f"  - {lemma}")

    # Helper to format synset lists
    def format_synset_list(title, synsets):
        print(f"\n🔗 {title}:")
        if synsets:
            for s in synsets:
                print(f"  - {s.name()} :: {s.definition()}")
        else:
            print("  (None)")

    format_synset_list("Hypernyms", syn.hypernyms())
    format_synset_list("Hyponyms", syn.hyponyms())
    format_synset_list("Part Meronyms", syn.part_meronyms())
    format_synset_list("Substance Meronyms", syn.substance_meronyms())
    format_synset_list("Member Meronyms", syn.member_meronyms())
    format_synset_list("Part Holonyms", syn.part_holonyms())
    format_synset_list("Substance Holonyms", syn.substance_holonyms())
    format_synset_list("Member Holonyms", syn.member_holonyms())
    format_synset_list("Attributes", syn.attributes())
    format_synset_list("Root Hypernyms", syn.root_hypernyms())

    print("\n📏 Path Similarity with Itself:")
    print(f"  - {syn.path_similarity(syn):.2f}")

    # All synsets from lemmas of the original synset
    print("\n🌐 All Synsets Related to Lemmas of This Synset:")
    seen = set()
    for lemma in lemmas:
        lemma_synsets = wn.synsets(lemma, pos=syn.pos())  # Filter by same POS
        for related_syn in lemma_synsets:
            if related_syn.name() not in seen:
                seen.add(related_syn.name())
                print(f"  - {related_syn.name()} :: {related_syn.definition()}")

# 📥 Interactive loop
def main():
    print("🔍 Enter a WordNet synset name (e.g., 'dog.n.01'). Press Enter with no input to exit.")
    while True:
        user_input = input("\nSynset name: ").strip()
        if not user_input:
            print("👋 Goodbye!")
            break
        pretty_synset_info(user_input)

if __name__ == "__main__":
    main()
