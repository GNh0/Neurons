# Dataset attribution

MaleCNS v1.0, released by the MaleCNS collaboration and Janelia Research Campus.

- Project: https://male-cns.janelia.org/
- Download and terms: https://male-cns.janelia.org/download/
- Paper: Berg et al., *Sexual dimorphism in the complete Drosophila male central nervous system connectome*, Cell (2026). https://doi.org/10.1016/j.cell.2026.08.015
- Dataset release: 2026-06-08.
- Dataset license: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/

Raw Feather files are unmodified. Derived files select annotated neurons with nonempty superclass, retain all provided positive pair weights within that set, convert 8 nm voxel coordinates to micrometres, and store connectivity as CSR. LIF dynamics, transmitter effects, artificial memory and visual lines are application modeling choices, not additional observations by the dataset authors. No endorsement by the authors is implied.

Third-party dependency licenses accompany the Windows bundle. LLM weights and Ollama are not redistributed in the bundle.
