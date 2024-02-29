# 🦜️🔗 LangChain {partner}

This repository contains 1 package with Datastax integrations with LangChain:

- [langchain-astradb](https://pypi.org/project/langchain-astradb/) integrates [AstraDB](https://www.datastax.com/products/datastax-astra).

## Initial Repo Checklist (Remove this section after completing)

This setup assumes that the partner package is already split. For those instructions,
see [these docs](https://python.langchain.com/docs/contributing/integrations#partner-packages).

- [ ] Fill out the readme above (for folks that follow pypi link)
- [ ] Copy package into /libs folder
- [ ] Update these fields in /libs/*/pyproject.toml

    - `tool.poetry.repository`
    - `tool.poetry.urls["Source Code"]`
    
- [ ] Add integration testing secrets in Github (ask Erick for help)
- [ ] Add secrets as env vars in .github/workflows/_release.yml
- [ ] Configure `LIB_DIRS` in .github/scripts/check_diff.py
- [ ] Add partner collaborators in Github (ask Erick for help)
- [ ] Add new repo to test-pypi and pypi trusted publishing (ask Erick for help)
- [ ] Populate .github/workflows/_release.yml with `on.workflow_dispatch.inputs.working-directory.default`
