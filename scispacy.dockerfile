# scispaCy baseline runtime (own deps, not the project lock). Used by scripts/baselines/scispacy_run.py.
#   docker build --platform linux/amd64 -f scispacy.dockerfile -t xmr4el-scispacy .
# KB / linker files (~1-2 GB for umls) download on first use into $SCISPACY_CACHE: mount a host dir there.
FROM python:3.10-slim

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir scispacy==0.6.2 \
 && python -c "from scispacy.candidate_generation import CandidateGenerator"

ENV SCISPACY_CACHE=/cache/scispacy
WORKDIR /app
