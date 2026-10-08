# Recipe: Reduce File Complexity

Target hotspot: `dirac_crystallography_suite.py`
(complexity 0.6, centrality 1.0)

1. Read dependents: `grep -n 'dirac_crystallography_suite.py' readmenator-agent/ARCHITECTURE*.md`
2. Extract functions/classes into new files in the same subsystem
3. Update imports
4. Regenerate: `readmenator .`
