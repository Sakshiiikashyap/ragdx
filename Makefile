PY ?= python

.PHONY: test exp lint

test:
	$(PY) -m pytest -q

# usage: make exp EXP=exp01
exp:
	$(PY) experiments/$(EXP).py

lint:
	$(PY) -m compileall -q src experiments tests
