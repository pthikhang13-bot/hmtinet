# Annotation guidelines (HMTI-Net verification subset)

## Task

For each cue, list every ontology concept that the cue genuinely refers to, using the English label as written in the ontology. Leave the cell empty if the cue refers to none. Separate multiple concepts with a semicolon.

Concepts to use (only these):

- Marco Polo Bridge Incident
- Battle of Pingxingguan
- Battle of Taierzhuang
- Nanjing Massacre
- Second Sino-Japanese War
- Eighth Route Army
- New Fourth Army
- National Revolutionary Army
- Imperial Japanese Army
- Chinese Communist Party
- KMT
- Yan'an Revolutionary Site
- Lugou Bridge
- Landmine warfare
- Guerrilla warfare
- National humiliation

## Rules

1. Code the *reference*, not the word. A cue mentioning the city of Nanjing does not refer to the Nanjing Massacre; a cue about the Nanjing Safety Zone does.
2. Code only what the cue itself states. Do not infer from the surrounding scene, which you cannot see.
3. Metaphor, idiom and unrelated homonyms do not count. Provide a note when you think a lexical matcher would misfire here.
4. A cue may carry several concepts; do not force a single choice.
5. If you are uncertain between two readings, record the one you would defend and add a note; these cells are exactly what the disagreement analysis needs.
6. Do not consult the model output or the lexicon before coding.

## Disagreement handling

Two annotators code the sheet independently. Disagreements on a concept are resolved by discussion; if unresolved, the cue is excluded from the scoring subset for that concept and the exclusion is reported. Cohen's kappa is computed per concept on the pre-discussion labels.
