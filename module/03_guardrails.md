Changelogs:

12/05: Updated "Strict Mode" for hallucination mitigation and section warning messages.


Module 3: Guardrails. For this module ensure there are no empty/missing sections, all sections are less than 150 words, do not fabricate any false information and ensure it's all real, if the paper is long, chunk it into sections.

evidence_mode = "strict" --> Only include claims, equations, and results that appear in the provided text. If not enough information can be found, issue message "Not enough detail to siummarize"

For sections that are missing/empty or too short (<50 words)--> issue error warning. Issue message "Section too short. Summary incomplete".
